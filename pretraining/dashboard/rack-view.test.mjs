import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {setImmediate} from 'node:timers/promises';
import * as model from './rack-model.mjs';

const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');

test('the page contains only the instrument container, no alternate UI', () => {
  const html = read('rack.html');
  assert.equal((html.match(/<main\b/g) || []).length, 1);
  assert.doesNotMatch(html, /<(button|nav|header|footer|section|a)\b/);
  assert.doesNotMatch(html, /inspector|matrix-menu|toast|training|fullscreen/);
  const css = read('rack.css');
  assert.doesNotMatch(css, /background-image|radial-gradient|repeating-linear-gradient/);
  assert.match(css, /background:#000/);
  assert.match(css, /width:100vw; height:100dvh/);
  assert.doesNotMatch(css, /width:min\(|aspect-ratio/);
});

for (const calibrated of [false,true]) test(`five passive canvases stay bounded and pause (panel calibration=${calibrated})`, async () => {
  const created = [], listeners = new Map(), frames = new Map(), transforms = [], shown=[];
  let nextFrame=1, fills=0, strokes=0, preferenceListener, interval;
  const preference = {matches:false, addEventListener:(_, fn) => { preferenceListener=fn; }};
  class Node {
    constructor(tag) {
      this.tag = tag; this.children = []; this.attributes = {}; this.dataset = {};
      this.classList = {toggle:() => {}};
      created.push(this);
    }
    append(...nodes) { this.children.push(...nodes); }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    getBoundingClientRect() { return {width:356,height:calibrated ? 600 : 280}; }
    getContext(kind) {
      assert.equal(kind,'2d');
      this.paints = [];
      const paints = this.paints;
      return {setTransform(...args){ transforms.push(args); },
        fillRect(...args){ paints.push({kind:'rect',style:this.fillStyle,args}); },
        beginPath(){}, lineTo(){}, moveTo(){}, closePath(){},
        fill(){ fills++; paints.push({kind:'fill',style:this.fillStyle}); },
        stroke(){ strokes++; paints.push({kind:'stroke',style:this.strokeStyle}); }};
    }
  }
  const grid = new Node('main'), body = new Node('body');
  const document = {
    body, hidden:false,
    getElementById:id => { assert.equal(id, 'matrix-grid'); return grid; },
    createElement:tag => new Node(tag),
    addEventListener:(event, fn) => listeners.set(event, fn),
  };
  const health = [0,1,2,3,4].map(rank => ({rank, reachable:true, checked_at:Date.now()/1000,
    hardware:{gpu_load:rank === 0 ? 50 : 0, pcpu_load:25, ram_usage:24, ram_total:96}}));
  health[2].reachable = null;
  let requests = 0;
  runInNewContext(read('rack.mjs').replace(/^import[^\n]+\n/, ''), {
    ...model, document, AbortController, URLSearchParams,
    liquidScene:(time,rank,load) => { shown[rank]=load; return model.liquidScene(time,rank,load); },
    window:{matchMedia:() => preference, devicePixelRatio:3,
      location:{search:calibrated ? '?panel=1424x280' : ''},
      addEventListener:(event) => assert.equal(event,'resize')},
    setTimeout:() => 1, clearTimeout:() => {}, setInterval:fn => { interval=fn; return 1; },
    requestAnimationFrame:fn => { const id=nextFrame++; frames.set(id,fn); return id; },
    cancelAnimationFrame:id => frames.delete(id),
    fetch:async url => {
      assert.equal(url, '/api/status'); requests++;
      return {ok:true, json:async () => ({rack:{health:{nodes:health}}})};
    },
  });
  await setImmediate();
  assert.equal(grid.children.length, 5);
  assert.ok(grid.children.every(n => n.tag === 'div' && n.attributes.role === 'img'));
  assert.deepEqual([...listeners.keys()], ['visibilitychange']);
  assert.ok(!created.some(n => ['button','a','nav','input'].includes(n.tag)));
  assert.match(grid.children[0].attributes['aria-label'], /GPU load 50 percent/);
  const canvases = created.filter(n => n.tag === 'canvas');
  assert.equal(canvases.length,5);
  assert.ok(canvases.every(n => n.width === 445 && n.height === (calibrated ? 750 : 350)));
  assert.equal(grid.dataset.rackPanel,String(calibrated));
  assert.deepEqual(transforms.at(-1),[1.25,0,0,calibrated ? 750/350 : 1.25,0,calibrated ? 75 : 0]);
  assert.ok(strokes>fills);
  assert.equal(grid.children[0].dataset.gpuActive, true);
  assert.equal(grid.children[1].dataset.gpuActive, false);
  assert.equal(grid.children[1].dataset.gpuState, 'idle');
  assert.equal(grid.children[2].dataset.gpuState, 'unavailable');
  assert.match(grid.children[2].attributes['aria-label'], /mac-3, GPU load unavailable; static amber outline/);
  assert.ok(canvases[2].paints.some(p => p.kind === 'stroke' && p.style === '#ffb84a'));
  // No GPU or RAM meter is drawn for the unreachable host, only the black clear.
  assert.ok(canvases[2].paints.filter(p => p.kind === 'rect').every(p => p.style === '#000'));
  const pump = timestamp => {
    const [id,fn] = frames.entries().next().value;
    frames.delete(id); fn(timestamp);
  };
  assert.equal(frames.size,1);
  pump(0); const afterFirst=fills;
  pump(16); assert.equal(fills,afterFirst);
  pump(42); assert.ok(fills > afterFirst);
  let clock=42;
  const advance=milliseconds => {
    for (let t=0; t<milliseconds; t+=50) {
      clock+=50;
      if (frames.size) pump(clock);
    }
  };
  health[0].hardware.gpu_load=100;
  canvases[0].paints.length=0;
  interval();
  assert.equal(shown[0],50);
  // The meter updates immediately even while the liquid is halfway up the tank.
  assert.equal(canvases[0].paints.filter(p=>p.kind==='rect' && p.args[0]>=330).length,40);
  advance(600);
  assert.ok(shown[0]>50 && shown[0]<100);
  health[0].hardware.gpu_load=10;
  const beforeRetarget=shown[0];
  interval();
  assert.equal(shown[0],beforeRetarget);
  advance(300);
  assert.ok(shown[0]<beforeRetarget && shown[0]>10);
  preference.matches=true; preferenceListener();
  assert.equal(frames.size,0);
  assert.equal(shown[0],10);
  preference.matches=false; preferenceListener();
  assert.equal(frames.size,1);
  document.hidden = true;
  listeners.get('visibilitychange')();
  assert.equal(frames.size,0);
  assert.equal(requests, 1);
  document.hidden = false;
  for (const host of health) host.hardware.gpu_load=0;
  interval();
  assert.equal(frames.size,1);
  assert.equal(shown[0],10);
  advance(600);
  assert.ok(shown[0]>0 && shown[0]<10);
  advance(1500);
  assert.equal(shown[0],0);
  assert.equal(frames.size,0);
  assert.equal(grid.children[2].dataset.gpuState, 'unavailable');
  assert.equal(grid.children[2].dataset.gpuActive, false);
  for (const canvas of canvases) canvas.paints.length=0;
  health[2].reachable=true;
  health[2].hardware.gpu_load=25;
  interval();
  assert.equal(grid.children[2].dataset.gpuState, 'active');
  assert.match(grid.children[2].attributes['aria-label'], /GPU load 25 percent/);
  assert.ok(canvases[2].paints.some(p => p.kind === 'stroke'));
  assert.ok(canvases[2].paints.every(p => p.style !== '#ffb84a'));
  assert.equal(frames.size,1);
  // All ranks stay Matrix green and subdued, including highlights and waterlines.
  const outlineColors=['rgb(26,114,47)','rgb(22,100,57)','rgb(44,110,37)','rgb(31,103,72)','rgb(34,112,83)'];
  assert.equal(new Set(outlineColors).size,5);
  for (const canvas of canvases) canvas.paints.length=0;
  for (const host of health) { host.reachable=true; host.hardware.gpu_load=100; }
  preference.matches=true; preferenceListener();
  for (const [rank,canvas] of canvases.entries()) {
    assert.ok(canvas.paints.some(p=>p.kind==='stroke' && p.style===outlineColors[rank]));
    const water=canvas.paints.filter(p=>p.kind==='fill');
    assert.ok(water.length>0);
    for (const paint of canvas.paints.filter(p=>p.kind==='fill' || p.kind==='stroke')) {
      const channels=paint.style.match(/\d+/g).map(Number);
      const brightest=Math.max(...channels);
      assert.ok(brightest<=193, 'Green reflections must stay below neon brightness');
      assert.ok((brightest-Math.min(...channels))/brightest>.35, 'Highlights must not wash out to white');
      assert.ok(channels[1]>channels[0] && channels[1]>channels[2], 'Every host uses a green tint');
    }
  }
  for (const host of health) host.reachable=null;
  interval();
  assert.ok(grid.children.every(n => n.dataset.gpuState === 'unavailable'));
  assert.ok(shown.every(value=>value===null));
  assert.equal(frames.size,0);
});
