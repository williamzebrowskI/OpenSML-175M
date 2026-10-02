#!/usr/bin/env python3
"""Local pretrained-model playground; follow published best snapshots."""
# Generation helpers adapted from the released OpenSML-150M playground.
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import queue
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT_RUN=ROOT/'runs/research_20b_176m_b44222/playground'
DEFAULT_V1_TOKENIZER=ROOT.parent/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
ASSETS=ROOT/'playground'

@dataclass
class Checkpoint:
    bundle: Path
    metadata: dict
    manifest: dict
    family: str = 'v1'
    model_type: str = 'base'
    @property
    def config(self): return self.metadata['recipe']['model']
    def public(self):
        m=self.metadata
        return dict(checkpoint=self.bundle.name, step=m['step'], tokens=m['tokens'],
                    validation_loss=m['best_val_loss'], context_limit=self.config['max_seq_len'],
                    max_output_tokens=min(2048,self.config['max_seq_len']), tokenizer='Own 32K byte-BPE',
                    model_type='base', selection='best validation loss')

@dataclass
class ModelSource:
    model_id: str
    label: str
    run: Path
    tokenizer_dir: Path = DEFAULT_V1_TOKENIZER
    def checkpoint(self):
        pointer=json.loads((self.run/'best.json').read_text())
        name=pointer.get('bundle','')
        if not isinstance(name,str) or not re.fullmatch(r'step_\d+_[a-f0-9]+',name):
            raise ValueError('Invalid playground pointer')
        bundle=self.run/name
        if bundle.is_symlink(): raise ValueError('Snapshot must not be a symlink')
        manifest=json.loads((bundle/'manifest.json').read_text())
        if manifest.get('format')!='sml-playground-snapshot-v1': raise ValueError('Invalid snapshot')
        path=bundle/'model.safetensors.json'
        record=manifest['files'][path.name];raw=path.read_bytes()
        if path.is_symlink() or len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Metadata checksum mismatch')
        metadata=json.loads(raw)
        if metadata['step']!=manifest['step'] or 'v2_contract' not in metadata:
            raise ValueError('Invalid pretrained checkpoint metadata')
        return Checkpoint(bundle,metadata,manifest)

def checkpoint_tokenizer(checkpoint, tokenizer_dir=None):
    from sml_v1.tokenization import Tokenizer
    tokenizer=Tokenizer(tokenizer_dir or DEFAULT_V1_TOKENIZER)
    if tokenizer.fingerprint!=checkpoint.metadata['v2_contract']['tokenizer'] or tokenizer.vocab_size!=checkpoint.config['vocab_size']:
        raise ValueError('Checkpoint tokenizer mismatch')
    return V1TokenizerAdapter(tokenizer)
class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: str = Field(pattern='^(user|assistant)$')
    content: str = Field(max_length=12000)

class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str | None = Field(default=None, pattern='^[a-z0-9_-]{1,48}$')
    messages: list[Message] = Field(min_length=1, max_length=100)
    instruction: str = Field(default='', max_length=12000)
    max_tokens: int = Field(default=64, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=1000, strict=True)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0, le=2 ** 32 - 1, strict=True)

    @model_validator(mode='after')
    def check_messages(self):
        if self.messages[-1].role != 'user' or not self.messages[-1].content.strip():
            raise ValueError('Finish with a nonempty user prompt')
        if len(self.instruction) + sum((len(m.content) for m in self.messages)) > 32000:
            raise ValueError('Conversation is too long; clear it and start a new prompt')
        return self

def completion_text(messages):
    text = ''
    for message in messages:
        if message.role == 'user' and text:
            text += '\n\n'
        text += message.content
    return text

def penalize_repetition(logits, generated, penalty):
    """Reduce scores for distinct tokens in the last 64 response tokens.

    Apply the sign-aware repetition penalty before temperature/top-k selection.
    Prompt tokens are excluded; 1.0 preserves the original sampling path.
    """
    if penalty == 1.0 or not generated:
        return logits
    import mlx.core as mx
    tokens = mx.array(sorted(set(generated[-64:])), dtype=mx.int32)[None, :]
    scores = logits.astype(mx.float32)
    selected = mx.take_along_axis(scores, tokens, axis=-1)
    adjusted = mx.where(selected < 0, selected * penalty, selected / penalty)
    return mx.put_along_axis(scores, tokens, adjusted, axis=-1)

def prepare_prompt(tokenizer, options, context_limit, add_bos=False, preserve_prompt=False, output_limit=192):
    """Keep an optional instruction intact while trimming older conversation."""
    instruction = options.instruction.strip()
    prefix = tokenizer.encode(instruction + '\n\n', out_type=int) if instruction else []
    instruction_tokens = len(prefix)
    if add_bos and tokenizer.bos_id() >= 0:
        prefix.insert(0, tokenizer.bos_id())
    ids = tokenizer.encode(completion_text(options.messages), out_type=int)
    if not ids:
        raise ValueError('Prompt has no tokens')
    max_new = min(options.max_tokens, output_limit, context_limit - 1)
    if preserve_prompt:
        remaining = context_limit - len(prefix) - 1
        if remaining < 1:
            raise ValueError('Instruction is too long. Shorten it to leave room for your message and output.')
        trimmed = max(0, len(ids) - remaining)
        prompt = prefix + ids[-remaining:]
        return dict(ids=prompt, instruction_tokens=instruction_tokens, trimmed_tokens=trimmed, max_tokens=min(max_new, context_limit - len(prompt)))
    remaining = context_limit - max_new - len(prefix)
    if remaining < 1:
        raise ValueError('Instruction is too long for the selected output limit. Shorten it or lower Max new tokens to leave room for your message.')
    trimmed = max(0, len(ids) - remaining)
    return dict(ids=prefix + ids[-remaining:], instruction_tokens=instruction_tokens, trimmed_tokens=trimmed, max_tokens=max_new)

class V1TokenizerAdapter:
    """Expose the playground interface without changing V1's text encoding."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.fingerprint = tokenizer.fingerprint

    def encode(self, text, out_type=int):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def bos_id(self):
        return -1

    def eos_id(self):
        return self.tokenizer.eos

class InferenceEngine:
    """All MLX operations run on one dedicated worker thread."""

    def __init__(self, run=DEFAULT_RUN, sources=None, default_model='research'):
        self.run = Path(run)
        self.sources = sources or {'research': ModelSource('research', 'Research pretrained — best', self.run)}
        self.default_model = default_model
        self.model = self.tokenizer = self.identity = None

    def load(self, checkpoint, tokenizer_dir=None):
        import mlx.core as mx
        from sml_v1.model import TransformerConfig, TransformerLM
        from sml_v1.inference import _cast_model_floats
        identity = (str(checkpoint.bundle.resolve()), checkpoint.family, checkpoint.manifest['files']['model.safetensors']['sha256'], checkpoint.manifest['files']['model.safetensors.json']['sha256'], str(tokenizer_dir))
        if self.identity == identity:
            return
        self.model = self.tokenizer = self.identity = None
        mx.clear_cache()
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(64 * 1024 ** 2)
        mx.reset_peak_memory()
        limit = 2048 if checkpoint.family == 'v1' else 256
        if not 1 < checkpoint.config['max_seq_len'] <= limit:
            raise ValueError('Checkpoint exceeds the supported context limit')
        tokenizer = checkpoint_tokenizer(checkpoint, tokenizer_dir)
        record = checkpoint.manifest['files']['model.safetensors']
        budget_mib = 1200 if checkpoint.metadata.get('compute_precision') == 'float32' else 600
        if record['bytes'] > budget_mib * 1024 ** 2:
            raise ValueError('Checkpoint exceeds this playground’s model-weight budget')
        weights = checkpoint.bundle / 'model.safetensors'
        if weights.is_symlink():
            raise ValueError('Weights must not be a symlink')
        with tempfile.TemporaryDirectory(prefix='sml-playground-') as tmp:
            target = Path(tmp) / 'model.safetensors'
            digest = hashlib.sha256()
            size = 0
            with weights.open('rb') as src, target.open('wb') as dst:
                for block in iter(lambda: src.read(4 * 1024 ** 2), b''):
                    dst.write(block)
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != record['sha256'] or size != record['bytes']:
                raise ValueError('Model weights checksum mismatch')
            model = TransformerLM(TransformerConfig(**checkpoint.config))
            precision = mx.float32 if checkpoint.metadata.get('compute_precision') == 'float32' else mx.bfloat16
            _cast_model_floats(model, precision)
            model.load_weights(str(target), strict=True)
            model.eval()
            mx.eval(model.parameters())
        self.model, self.tokenizer, self.identity = (model, tokenizer, identity)

    def generate(self, options, emit, cancelled):
        import mlx.core as mx
        import numpy as np
        from sml_v1.inference import _sample_next_id
        source = self.sources[options.model_id or self.default_model]
        checkpoint = None
        for attempt in range(3):
            if cancelled.is_set():
                return
            try:
                checkpoint = source.checkpoint()
                emit(dict(type='status', message=f"Loading best checkpoint · step {checkpoint.metadata['step']:,}"))
                self.load(checkpoint, source.tokenizer_dir)
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise ValueError('Best checkpoint changed during loading; please retry')
        model, tokenizer = (self.model, self.tokenizer)
        prepared = prepare_prompt(tokenizer, options, model.cfg.max_seq_len, preserve_prompt=True, output_limit=2048)
        ids, max_new = (prepared['ids'], prepared['max_tokens'])
        emit(dict(type='start', **checkpoint.public(), model_id=source.model_id, model_label=source.label, run_name=source.run.name, prompt_tokens=len(ids), trimmed_tokens=prepared['trimmed_tokens'], max_tokens=max_new, instruction_tokens=prepared['instruction_tokens'], repetition_penalty=options.repetition_penalty))
        full_context = checkpoint.metadata.get('decode_mode') == 'full_context'
        generated, caches = ([], None)
        start = time.perf_counter()
        try:
            if cancelled.is_set():
                return
            logits, caches = model.logits(mx.array([ids], dtype=mx.int32))
            scores = logits[:, -1, :]
            rng = np.random.default_rng(options.seed)
            reason = 'length'
            for _ in range(max_new):
                if cancelled.is_set():
                    reason = 'stopped'
                    break
                sampling_scores = penalize_repetition(scores, generated, options.repetition_penalty)
                token = _sample_next_id(sampling_scores, options.temperature, options.top_k, rng)
                if token == tokenizer.eos_id():
                    reason = 'eos'
                    break
                generated.append(token)
                text = tokenizer.decode(generated)
                elapsed = time.perf_counter() - start
                emit(dict(type='text', text=text, generated_tokens=len(generated), seconds=elapsed, tokens_per_second=len(generated) / max(elapsed, 1e-06)))
                if len(generated) < max_new and (not cancelled.is_set()):
                    if full_context:
                        logits, _ = model.logits(mx.array([ids + generated], dtype=mx.int32))
                        scores = logits[:, -1, :]
                    else:
                        scores, caches = model.step(mx.array([[token]], dtype=mx.int32), caches=caches)
            elapsed = time.perf_counter() - start
            emit(dict(type='done', text=tokenizer.decode(generated), generated_tokens=len(generated), seconds=elapsed, tokens_per_second=len(generated) / max(elapsed, 1e-06), stop_reason=reason, peak_gib=mx.get_peak_memory() / 1024 ** 3))
        finally:
            caches = None
            mx.clear_cache()

class GenerationService:

    def __init__(self, engine):
        self.engine = engine
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sml-inference')
        self.busy = threading.Lock()
        self.cancelled = None
        self.follow_stop = threading.Event()
        self.follow_thread = None

    def start(self, options):
        if not self.busy.acquire(blocking=False):
            raise HTTPException(409, 'A generation is already running. Stop it or wait a moment.')
        events = queue.Queue(maxsize=32)
        cancelled, finished = (threading.Event(), threading.Event())
        self.cancelled = cancelled

        def emit(event):
            while not cancelled.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def work():
            try:
                self.engine.generate(options, emit, cancelled)
            except Exception as exc:
                emit(dict(type='error', message=str(exc)))
            finally:
                finished.set()
                self.busy.release()
        try:
            self.pool.submit(work)
        except Exception:
            self.busy.release()
            raise
        return (events, cancelled, finished)

    def close(self):
        self.follow_stop.set()
        if self.follow_thread is not None:
            self.follow_thread.join(timeout=10)
        if self.cancelled is not None:
            self.cancelled.set()
        self.pool.shutdown(wait=False, cancel_futures=True)

def create_app(run=DEFAULT_RUN, engine=None, tokenizer=DEFAULT_V1_TOKENIZER):
    sources={'research':ModelSource('research','Research pretrained — best',Path(run),Path(tokenizer))}
    service=GenerationService(engine or InferenceEngine(run,sources,'research'))
    app=FastAPI(docs_url=None,redoc_url=None);app.state.service=service
    @app.middleware('http')
    async def local_only(request,call_next):
        host=request.headers.get('host','').split(':')[0];origin=request.headers.get('origin')
        if host not in ('127.0.0.1','localhost','testserver') or (origin and origin!=str(request.base_url).rstrip('/')):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Local, same-origin requests only'},status_code=403)
        response=await call_next(request);response.headers['Cache-Control']='no-store'
        return response
    @app.get('/')
    def index(): return FileResponse(ASSETS/'index.html')
    @app.get('/app.js')
    def javascript(): return FileResponse(ASSETS/'app.js',media_type='text/javascript')
    @app.get('/style.css')
    def stylesheet(): return FileResponse(ASSETS/'style.css',media_type='text/css')
    def source_status():
        source=sources['research']
        info=dict(model_id=source.model_id,model_label=source.label,run_name=source.run.parent.name)
        try: return dict(available=True,**info,**source.checkpoint().public())
        except FileNotFoundError: return dict(available=False,**info,message='Waiting for the first 100M-token checkpoint publication.')
        except (OSError,ValueError,KeyError) as exc: return dict(available=False,**info,message=str(exc))
    @app.get('/api/models')
    def models(): return dict(default_model='research',models=[source_status()])
    @app.get('/api/status')
    def status():
        return dict(busy=service.busy.locked(),**source_status())
    @app.post('/api/generate')
    async def generate(options:GenerationRequest,request:Request):
        if options.model_id not in (None,'research'): raise HTTPException(404,'Unknown model')
        if not status().get('available'): raise HTTPException(503,status().get('message'))
        events,cancelled,finished=service.start(options)
        async def stream():
            try:
                while True:
                    if await request.is_disconnected(): break
                    try: event=events.get_nowait()
                    except queue.Empty:
                        if finished.is_set(): break
                        await asyncio.sleep(.015);continue
                    yield 'data: '+json.dumps(event,ensure_ascii=False)+'\n\n'
            finally: cancelled.set()
        return StreamingResponse(stream(),media_type='text/event-stream')
    return app

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN,help='Published playground snapshots, not training checkpoints')
    p.add_argument('--tokenizer',type=Path,default=DEFAULT_V1_TOKENIZER)
    p.add_argument('--port',type=int,default=8767)
    a=p.parse_args();app=create_app(a.run_dir.resolve(),tokenizer=a.tokenizer.resolve())
    import uvicorn
    print(f'Playground: http://127.0.0.1:{a.port}; waiting for best pretrained snapshot',flush=True)
    try: uvicorn.run(app,host='127.0.0.1',port=a.port,access_log=False)
    finally: app.state.service.close()
if __name__=='__main__': main()
