import {numeric, sensorNodes, gpuSignal, memoryLevel, displayTransform, liquidScene, liquidFillAt, retargetLiquidFill} from './rack-model.mjs';

const grid = document.getElementById('matrix-grid');
const rackPanel = new URLSearchParams(window.location.search).get('panel') === '1424x280';
grid.dataset.rackPanel = String(rackPanel);
const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
const displays = [], targets = Array.from({length:5}, () => ({}));
const liquidFills = Array(5).fill(null);
const colors = [[42,184,76], [36,160,91], [70,176,60], [50,166,116], [54,180,134]];
const frameMs = 1000/24;
let data = null, received = 0, online = false, requesting = false, timer;
let frameId = 0, lastFrame = -Infinity, motionTime = 0, lastTick = null;

// Precomputed color ramps avoid per-face color math while painting.
const palettes = colors.map(rgb => Array.from({length:64}, (_, i) => {
  // Subdued green highlights avoid neon glare while retaining the wave detail.
  const level = i/63, highlight = .06*Math.max(0,(level-.72)/.28);
  return 'rgb('+rgb.map(c => Math.round((c+(255-c)*highlight)*Math.min(1,level*1.12))).join(',')+')';
}));

for (let rank=0; rank<5; rank++) {
  const panel = document.createElement('div');
  panel.className = 'matrix-node';
  panel.setAttribute('role', 'img');
  const canvas = document.createElement('canvas');
  canvas.className = 'core-display';
  canvas.setAttribute('aria-hidden', 'true');
  panel.append(canvas); grid.append(panel);
  displays.push({panel, canvas, context:canvas.getContext('2d', {alpha:false})});
}

function resize() {
  for (const {canvas} of displays) {
    const rect = canvas.getBoundingClientRect();
    // Limit backing pixels on Retina displays rather than compete with training.
    const ratio = Math.min(window.devicePixelRatio || 1, 1.25);
    canvas.width = Math.max(1, Math.round(rect.width*ratio));
    canvas.height = Math.max(1, Math.round(rect.height*ratio));
  }
  paint();
}

function paint() {
  for (let rank=0; rank<5; rank++) {
    const {canvas, context:ctx} = displays[rank], hw = targets[rank];
    if (!ctx) continue;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle = '#000'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.setTransform(...displayTransform(canvas.width,canvas.height,rackPanel));
    const gpu = gpuSignal(hw.gpu_load);
    const scene = liquidScene(preference.matches ? 0 : motionTime, rank, liquidFillAt(liquidFills[rank],motionTime));
    const palette = palettes[rank];
    for (const face of [...scene.back,...scene.water,...scene.front]) {
      const cage=face.kind==='cage-edge';
      ctx.beginPath();
      face.points.forEach((p,i) => i ? ctx.lineTo(p.x,p.y) : ctx.moveTo(p.x,p.y));
      if (face.close) ctx.closePath();
      if (!cage && face.kind!=='waterline') {
        ctx.fillStyle=palette[Math.round(face.light*63)]; ctx.fill();
      }
      ctx.strokeStyle=!gpu.known ? '#ffb84a' : cage ? palette[35]
        : face.kind==='waterline' ? palette[62] : palette[Math.round(face.light*48)];
      ctx.lineWidth=face.kind==='waterline' ? 1.9 : cage ? 1.25 : .55;
      ctx.lineJoin='round'; ctx.stroke();
    }
    if (gpu.known) {
      // A stationary 0-100% ruler remains readable while the 3D core rotates.
      for (let i=0; i<20; i++) {
        const y=246-i*12, fill=Math.max(0,Math.min(1,gpu.level*20-i));
        const width=(i+1)%5 === 0 ? 12 : 8;
        ctx.fillStyle=palette[10]; ctx.fillRect(342-width,y,width,9);
        if (fill > 0) {
          ctx.fillStyle=palette[54]; ctx.fillRect(342-width,y+9*(1-fill),width,9*fill);
        }
      }
    }
    const ram = memoryLevel(hw.ram_usage, hw.ram_total);
    if (ram.known) {
      // The horizontal strip is physical memory, separate from the GPU ruler.
      for (let i=0; i<24; i++) {
        const fill=Math.max(0,Math.min(1,ram.percent/100*24-i));
        ctx.fillStyle=palette[10]; ctx.fillRect(22+i*12,271,9,4);
        if (fill > 0) {
          ctx.fillStyle=palette[42]; ctx.fillRect(22+i*12,271,9*fill,4);
        }
      }
    }
  }
}

function wantsMotion() {
  return !document.hidden && !preference.matches && targets.some((hw,rank) => {
    const gpu=gpuSignal(hw.gpu_load);
    return gpu.known && (gpu.level>0 || liquidFillAt(liquidFills[rank],motionTime)!==hw.gpu_load);
  });
}

function frame(timestamp) {
  frameId = 0;
  if (!wantsMotion()) { lastTick = null; return; }
  if (lastTick !== null) motionTime += Math.min((timestamp-lastTick)/1000,.1);
  lastTick = timestamp;
  const elapsed = timestamp-lastFrame;
  const continuing=wantsMotion();
  // Paint the final empty tank before stopping the loop after a fall to zero.
  if (elapsed >= frameMs || !continuing) {
    lastFrame = numeric(lastFrame) ? timestamp-elapsed%frameMs : timestamp;
    paint();
  }
  if (continuing) frameId = requestAnimationFrame(frame);
  else lastTick = null;
}

function syncMotion() {
  if (!wantsMotion()) {
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0; lastTick = null;
  } else if (!frameId) frameId = requestAnimationFrame(frame);
}

function render() {
  const nodes = sensorNodes(data, Date.now()/1000, online && Date.now()-received < 15000);
  for (const {rank, hardware} of nodes) {
    targets[rank] = hardware;
    liquidFills[rank] = retargetLiquidFill(liquidFills[rank],hardware.gpu_load,motionTime,
      document.hidden || preference.matches);
    const gpu = gpuSignal(hardware.gpu_load), ram = memoryLevel(hardware.ram_usage,hardware.ram_total);
    const panel = displays[rank].panel;
    panel.dataset.gpuActive = gpu.known && gpu.level > 0;
    panel.dataset.gpuState = gpu.known ? (gpu.level > 0 ? 'active' : 'idle') : 'unavailable';
    const load = gpu.known ? Math.round(hardware.gpu_load)+' percent' : 'unavailable';
    const explanation = gpu.known ? ' shown immediately by the right meter; liquid level eases to this reading over 1.8 seconds' : '; static amber outline means missing readings, not zero load or a powered-off Mac';
    panel.setAttribute('aria-label', 'mac-'+(rank+1)+', GPU load '+load+explanation+', memory '+(ram.known ? Math.round(ram.percent)+' percent' : 'unavailable')+' shown below. Waves, shimmer and rotation are decorative, not individual GPU cores or electricity consumption.');
  }
  if (!document.hidden) paint();
  syncMotion();
}

async function poll() {
  clearTimeout(timer);
  if (document.hidden || requesting) return;
  requesting = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(),6500);
  try {
    const response = await fetch('/api/status', {cache:'no-store',signal:controller.signal});
    if (!response.ok) throw Error('Telemetry unavailable');
    const next = await response.json();
    if (!Array.isArray(next?.rack?.health?.nodes)) throw Error('Invalid sensor feed');
    data = next; online = true; received = Date.now();
  } catch { online = false; }
  finally {
    clearTimeout(timeout); requesting = false; render();
    if (!document.hidden) timer = setTimeout(poll,5000);
  }
}

preference.addEventListener('change',render);
document.addEventListener('visibilitychange', () => {
  clearTimeout(timer); render();
  if (!document.hidden) poll();
});
window.addEventListener('resize',resize);
setInterval(() => { if (!document.hidden) render(); },5000);
resize(); render(); poll();
