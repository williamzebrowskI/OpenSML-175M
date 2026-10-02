#!/usr/bin/env python3
"""Fresh five-Mac research pretraining; plan only unless --run is explicit."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.common import ROOT, file_sha256, fingerprint, load_corpus, lock, read_json
from sml_v1.recipe import learning_rate, tokens_per_update, validate
from sml_v1.sample_prompts import read_prompts


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT.parent/'configs/pretrain_20b.json')
    p.add_argument('--corpus', type=Path, default=ROOT.parent/'configs/corpus_20b.json')
    p.add_argument('--tokenizer', type=Path,
                   default=ROOT.parent/'reference/opensml-150m/tokenizer/bytebpe32k_v1')
    p.add_argument('--sample-prompts-file', type=Path,
                   help='Live file on Mac 1; overrides the recipe-relative prompt path')
    p.add_argument('--output', type=Path, default=ROOT/'runs/research_20b_176m_b44222')
    p.add_argument('--network', type=Path, help='Verified current five-Mac network plan')
    p.add_argument('--resume', type=Path, help='Explicit checkpoint from this output; otherwise fresh')
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--stop-after-steps', type=int, default=0,
                   help='Bounded smoke/pilot run; 0 means the full token target')
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    if a.stop_after_steps < 0:
        p.error('--stop-after-steps must be nonnegative')
    if a.run and a.network is None:
        p.error('--run requires --network with fresh network/correctness verification')
    return a


def main(argv=None):
    a = arguments(argv)
    config, output = a.config.resolve(), a.output.resolve()
    recipe = validate(read_json(config))
    if len(recipe['batches']) != 5 or recipe.get('data_mode') != 'hf_stream':
        raise ValueError('Research launcher requires five ranks and HF streaming')
    if any(k in recipe for k in ('continuation', 'stage_b_transition', 'topology_transition',
                                 'lr_trial_transition', 'scheduler_transition')):
        raise ValueError('Use a fresh research recipe, not a historical transition')
    corpus = load_corpus(a.corpus)
    prompts_file = (a.sample_prompts_file or config.parent/recipe['sample_prompts_file']).resolve()
    prompts, _ = read_prompts(prompts_file)
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Research outputs must stay inside pretraining/runs')
    if (output/'latest.json').exists() and a.resume is None:
        raise ValueError('Existing output requires explicit --resume; fresh weights never overwrite it')
    if a.resume and not a.resume.resolve().is_relative_to(output):
        raise ValueError('Resume must come from this research output')
    print(json.dumps(dict(mode='run' if a.run else 'plan only', initialization='resume' if a.resume else 'fresh',
        config=str(config), output=str(output), target_tokens=recipe['target_tokens'], model=recipe['model'],
        physical_mac_order=[1,2,3,5,4], batches=recipe['batches'], grad_accum=recipe['grad_accum'],
        tokens_per_update=tokens_per_update(recipe), peak_lr=recipe['peak_lr'],
        warmup_tokens=int(recipe['target_tokens']*recipe['warmup_fraction']),
        decay_start_tokens=int(recipe['target_tokens']*recipe['decay_start_fraction']),
        final_lr=learning_rate(recipe['target_tokens'],recipe),
        corpus_shares={s['label']:s['weight'] for s in corpus['sources']},
        tokenizer=str(a.tokenizer.resolve()), sample_prompts_file=str(prompts_file),
        sample_prompt_count=len(prompts), prompt_reload='each sampling event on coordinator',
        validation='fixed held-out data; editable prompts are qualitative samples only',
        stop_after_steps=a.stop_after_steps), indent=2), flush=True)
    if not a.run:
        print('No downloads, SSH, model initialization or training. Add --run only after preflight.')
        return
    from sml_v1.tokenization import Tokenizer
    from sml_v1.stream import stream_manifest
    tokenizer = Tokenizer(a.tokenizer)
    if tokenizer.vocab_size != recipe['model']['vocab_size']:
        raise ValueError('Frozen tokenizer vocabulary differs from model')
    manifest = stream_manifest(corpus, tokenizer, recipe['stream'])
    modules = {p.name:file_sha256(p) for p in (ROOT/'sml_v1').glob('*.py')}
    proof = read_json(a.network/'five_rank_correctness.json')
    receipts = proof.get('receipts', [])
    if (proof.get('modules') != modules or len(receipts) != 5 or
            {r.get('rank') for r in receipts} != set(range(5)) or
            any(r.get('status') != 'passed' for r in receipts)):
        raise ValueError('Run five-rank correctness checks against the current worker code first')
    if a.resume:
        from sml_v1.checkpoint_metadata import verify_bundle_metadata
        from sml_v1.continuation import check_resume
        meta = verify_bundle_metadata(a.resume)
        contract = dict(recipe=fingerprint(recipe), tokenizer=tokenizer.fingerprint,
                        data=fingerprint(manifest), code=fingerprint(modules))
        check_resume(meta, contract, recipe, manifest)
    from five_mac_cluster import Cluster, PYTHON
    from sml_v1.launch import stage
    cluster = Cluster(a.network)
    with lock(output/'.launcher.lock'):
        if (output/'STOP').exists() and not a.clear_stop:
            raise ValueError('STOP exists; use --clear-stop explicitly to resume')
        job = output/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster.preflight(job)
        source = stage(job, recipe, a.tokenizer, a.resume, corpus)
        cluster.stage(source)
        if a.clear_stop:
            (output/'STOP').unlink(missing_ok=True)
        worker = ['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v1.pretrain',
            '--config',str(source/'recipe.json'),'--tokenizer',str(source/'tokenizer'),
            '--corpus',str(source/'corpus.json'),'--save-dir',str(output),'--job-dir',str(job),
            '--ring-control','native','--stop-after-steps',str(a.stop_after_steps),
            '--sample-prompts-file',str(prompts_file)]
        if a.resume:
            worker += ['--resume',str(source/'checkpoint')]
        cluster.launch(job, source, worker, timeout=None, stop_dir=output)
        result = read_json(job/'result.json')
        if result.get('status') not in ('stopped','complete'):
            raise RuntimeError('Missing clean worker completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}', flush=True)


if __name__ == '__main__':
    main()
