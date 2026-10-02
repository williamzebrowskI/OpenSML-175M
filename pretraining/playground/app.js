'use strict';
const $ = id => document.getElementById(id);
let history = [], controller = null, running = false;
let selectedModel = null;
const defaults = { 'max-tokens': 64, temperature: 0.7, 'top-k': 40, 'repetition-penalty': 1.1, seed: 2026 };
const promptPresets = [
  ['Ocean tides', 'Ocean tides are the regular rise and fall of sea levels. They are mainly caused by'],
  ['Rainbows', 'A rainbow appears when sunlight passes through drops of water in the air. The colors form because'],
  ['Seeds', 'A seed contains a young plant and a supply of stored food. When it receives enough water,'],
  ['Magnets', 'A magnet has a north pole and a south pole. When two north poles are brought close together,'],
  ['Bread', 'Yeast helps bread dough rise by'],
  ['Earth’s rotation', 'Earth rotates on its axis approximately once every 24 hours. This rotation causes'],
  ['The heart', 'The human heart is a muscular organ. Its main function is to'],
  ['Photosynthesis', 'Photosynthesis is the process by which plants use sunlight to'],
  ['Ancient Egypt', 'The ancient Egyptians built pyramids primarily to serve as'],
  ['Continents', 'Africa is the second-largest continent by area. The largest continent is'],
];
for (const [label, prompt] of promptPresets) {
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'prompt-pill'; button.textContent = label;
  button.title = prompt; button.setAttribute('aria-label', `${label}: ${prompt}`);
  button.onclick = () => {
    if (running) return;
    $('prompt').value = prompt;
    $('prompt').dispatchEvent(new Event('input', { bubbles: true }));
    $('prompt').focus(); $('prompt').setSelectionRange(prompt.length, prompt.length);
  };
  $('prompt-presets').append(button);
}
function notice(text = '') { $('notice').textContent = text; }
function scrollDown(force = false) {
  const el = $('conversation');
  if (force || el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
}
function updateTemperature() { $('temperature-value').textContent = Number($('temperature').value).toFixed(2); }
$('temperature').addEventListener('input', updateTemperature);
$('reset-settings').onclick = () => { for (const [id, value] of Object.entries(defaults)) $(id).value = value; updateTemperature(); };
function setRunning(value) {
  running = value; $('send').disabled = value; $('stop').hidden = !value;
  $('clear').disabled = value; $('prompt').disabled = value;
  $('instruction').disabled = value; $('clear-instruction').disabled = value;
  $('model-select').disabled = value || !selectedModel;
  for (const button of $('prompt-presets').children) button.disabled = value;
  $('compose-status').textContent = value ? 'Generating…' : '';
}
async function refresh() {
  if (!selectedModel) return;
  const requestedModel = selectedModel;
  try {
    const result = await fetch(`/api/status?model_id=${encodeURIComponent(requestedModel)}`, { cache: 'no-store' });
    if (!result.ok) throw new Error('Cannot connect to the local server');
    const state = await result.json();
    if (requestedModel !== selectedModel) return;
    if (state.available) {
      const limit = state.max_output_tokens || 192;
      $('max-tokens').max = limit;
      $('max-tokens').title = `1–${limit.toLocaleString()} output tokens; prompt and output share the ${state.context_limit.toLocaleString()}-token context.`;
      if (Number($('max-tokens').value) > limit) $('max-tokens').value = limit;
    }
    $('status-dot').classList.toggle('offline', !state.available);
    $('checkpoint-step').textContent = state.available ? `${state.model_type === 'sft' ? `SFT ${state.sft_stage || 1} · ` : ''}Best · ${state.step.toLocaleString()}` : 'Unavailable';
    $('checkpoint-step').title = state.available ? `${state.tokens.toLocaleString()} tokens · ${state.validation_loss.toFixed(4)} validation loss` : state.message;
    $('model-info').textContent = state.available ? `${state.context_limit.toLocaleString()}-token context · ${state.tokenizer} · Run: ${state.run_name}. Base pretrained model, not instruction-tuned. New completions use this run's latest best.` : state.message;
  } catch (error) { $('checkpoint-step').textContent = 'Offline'; $('checkpoint-step').title = 'Start the playground server on Mac-1.'; $('status-dot').classList.add('offline'); }
}
$('refresh').onclick = refresh;
setInterval(refresh, 15000);
async function loadModels() {
  try {
    const result = await fetch('/api/models', { cache: 'no-store' });
    if (!result.ok) throw new Error('Cannot load models. Refresh after restarting the playground.');
    const catalog = await result.json();
    $('model-select').replaceChildren();
    for (const model of catalog.models) {
      const option = document.createElement('option');
      option.value = model.model_id; option.textContent = model.model_label;
      $('model-select').append(option);
    }
    selectedModel = catalog.default_model;
    $('model-select').value = selectedModel;
    $('model-select').disabled = false;
    await refresh();
  } catch (error) { notice(error.message); $('checkpoint-step').textContent = 'Offline'; }
}
$('model-select').onchange = () => {
  if (running) return;
  selectedModel = $('model-select').value;
  history = []; $('messages').replaceChildren();
  notice('Model switched. Previous conversation context cleared.');
  refresh();
};
loadModels();
function addMessage(role, text) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const head = document.createElement('div'); head.className = 'message-head';
  const label = document.createElement('span'); label.textContent = role === 'user' ? 'You' : 'OpenSML';
  const checkpoint = document.createElement('span'); checkpoint.className = 'checkpoint-tag';
  const copy = document.createElement('button'); copy.className = 'copy'; copy.textContent = 'Copy'; copy.type = 'button';
  const body = document.createElement('div'); body.className = 'message-body'; body.textContent = text;
  copy.onclick = async () => { try { await navigator.clipboard.writeText(body.textContent); copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1200); } catch { notice('Copy is unavailable in this browser. You can select the response text.'); } };
  head.append(label, checkpoint, copy); article.append(head, body);
  const metrics = document.createElement('div'); metrics.className = 'metrics'; article.append(metrics);
  $('messages').append(article); scrollDown(true);
  return { article, body, checkpoint, metrics };
}
function showMetrics(view, event, final = false) {
  view.metrics.replaceChildren();
  const rate = document.createElement('span'); rate.className = 'rate';
  rate.textContent = `${Number(event.tokens_per_second || 0).toFixed(1)} tok/s`;
  const count = document.createElement('span'); count.textContent = `${event.generated_tokens} tokens`;
  const time = document.createElement('span'); time.textContent = `${Number(event.seconds || 0).toFixed(2)}s`;
  view.metrics.append(rate, count, time);
  if (final) { const stop = document.createElement('span'); stop.textContent = event.stop_reason === 'eos' ? 'End of text' : event.stop_reason === 'length' ? 'Token limit' : 'Stopped'; view.metrics.append(stop); }
}
async function generate(event) {
  event.preventDefault();
  if (running) return;
  if (!selectedModel) { notice('Wait for the model list to load.'); return; }
  const prompt = $('prompt').value;
  if (!prompt.trim()) return;
  for (const id of ['max-tokens', 'top-k', 'repetition-penalty', 'seed']) if (!$(id).reportValidity()) return;
  notice();
  const options = { model_id: selectedModel, instruction: $('instruction').value, messages: [...($('use-context').checked ? history : []), { role: 'user', content: prompt }], max_tokens: Number($('max-tokens').value), temperature: Number($('temperature').value), top_k: Number($('top-k').value), repetition_penalty: Number($('repetition-penalty').value), seed: Number($('seed').value) };
  addMessage('user', prompt);
  const view = addMessage('assistant', 'Connecting…'); view.body.classList.add('waiting');
  $('prompt').value = ''; setRunning(true); controller = new AbortController();
  let text = '', started = false, failed = false, completed = false;
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options), signal: controller.signal });
    if (!response.ok) { const error = await response.json(); throw new Error(typeof error.detail === 'string' ? error.detail : 'Check the generation settings and conversation length.'); }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    function consume(frame) {
      const data = frame.split('\n').filter(line => line.startsWith('data: ')).map(line => line.slice(6)).join('\n');
      if (!data) return;
      const item = JSON.parse(data);
      if (item.type === 'status') view.body.textContent = 'Loading…';
      if (item.type === 'start') {
        started = true; view.checkpoint.textContent = `${item.model_id.toUpperCase()} · ${item.step.toLocaleString()}`;
        view.checkpoint.title = item.instruction_tokens ? 'Best checkpoint · instructions included' : 'Best checkpoint';
        view.body.textContent = ''; view.body.classList.remove('waiting'); view.body.classList.add('streaming');
        const notes = [];
        if (item.trimmed_tokens) notes.push(`${item.trimmed_tokens} older tokens trimmed to fit context.`);
        if (item.max_tokens < options.max_tokens) notes.push(`Output capped at ${item.max_tokens.toLocaleString()} tokens after allowing for the prompt and model limits.`);
        notice(notes.join(' '));
      }
      if (item.type === 'text' || item.type === 'done') {
        if (item.type === 'done') completed = true;
        text = item.text; view.body.textContent = text; showMetrics(view, item, item.type === 'done');
        if (item.type === 'done' && !text) view.body.textContent = 'End of text.';
        scrollDown();
      }
      if (item.type === 'error') throw new Error(item.message);
    }
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let split; while ((split = buffer.indexOf('\n\n')) >= 0) { consume(buffer.slice(0, split)); buffer = buffer.slice(split + 2); }
      if (done) break;
    }
    if (!completed) throw new Error('The response stream ended early. Please try again.');
  } catch (error) {
    if (error.name === 'AbortError') { if (!text) view.body.textContent = 'Stopped.'; const label = document.createElement('span'); label.textContent = 'Stopped'; view.metrics.append(label); notice('Generation stopped.'); }
    else { failed = true; notice(error.message); if (!text) view.body.textContent = 'Could not generate a response.'; }
  } finally {
    view.body.classList.remove('waiting', 'streaming');
    if (started && !failed && text) { history.push({ role: 'user', content: prompt }, { role: 'assistant', content: text }); }
    else if (!text) $('prompt').value = prompt;
    controller = null; setRunning(false); $('prompt').focus(); refresh();
  }
}
$('composer').addEventListener('submit', generate);
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } });
$('stop').onclick = () => controller?.abort();
$('clear-instruction').onclick = () => { $('instruction').value = ''; $('instruction').focus(); };
$('clear').onclick = () => { if (running) return; history = []; $('messages').replaceChildren(); $('prompt').value = ''; notice(); $('prompt').focus(); };
window.addEventListener('pagehide', () => controller?.abort());
