import test from 'node:test';
import assert from 'node:assert/strict';
import {sensorNodes, gpuSignal, liquidSurface, surfaceMean, liquidScene, memoryLevel, displayTransform, liquidFillAt, retargetLiquidFill, liquidTransitionSeconds} from './rack-model.mjs';

const sample = () => ({rack:{health:{nodes:[
  {rank:0, reachable:true, checked_at:100, hardware:{gpu_load:65}},
]}}});

test('dedicated panel undoes non-native display stretching without changing geometry', () => {
  for (const [width,height] of [[1424,280],[1424,600],[1920,1080],[712,300]]) {
    const [sx,, ,sy,tx,ty]=displayTransform(width/5,height,true);
    assert.equal(tx,0); assert.equal(ty,height*.1);
    // Five equally scaled instruments fit across the calibrated panel.
    assert.ok(Math.abs(sx*1780/width-1) < 1e-12);
    assert.ok(Math.abs(sy*350/height-1) < 1e-12);
  }
  const [sx,,,sy]=displayTransform(480,1080);
  assert.equal(sx,sy);
});

test('five ordered instruments use live host readings, not training logs', () => {
  const nodes = sensorNodes(sample(), 110);
  assert.deepEqual(nodes.map(n => n.rank), [0,1,2,3,4]);
  assert.equal(nodes[0].hardware.gpu_load, 65);
  assert.deepEqual(nodes[1].hardware, {});
});
test('stale or disconnected hardware never appears live', () => {
  assert.deepEqual(sensorNodes(sample(), 135)[0].hardware, {});
  assert.deepEqual(sensorNodes(sample(), 110, false)[0].hardware, {});
  const data = sample(); data.rack.health.nodes[0].reachable = false;
  assert.deepEqual(sensorNodes(data, 110)[0].hardware, {});
});
test('missing and malformed health yields five unavailable instruments', () => {
  for (const data of [null, {}, {rack:{health:{nodes:'invalid'}}}]) {
    const nodes = sensorNodes(data);
    assert.equal(nodes.length, 5);
    assert.ok(nodes.every(n => Object.keys(n.hardware).length === 0));
  }
});
test('idle GPU is a measured zero, not unavailable or invented activity', () => {
  const data = sample(); data.rack.health.nodes[0].hardware.gpu_load = 0;
  assert.equal(sensorNodes(data, 110)[0].hardware.gpu_load, 0);
  assert.deepEqual(gpuSignal(0), {known:true,level:0});
});
test('GPU intensity is bounded and proportional, independently of decorative motion', () => {
  for (const value of [null, NaN, -1, 101]) assert.deepEqual(gpuSignal(value), {known:false,level:0});
  assert.deepEqual(gpuSignal(100), {known:true,level:1});
  assert.deepEqual(gpuSignal(20), {known:true,level:.2});
});
test('liquid fill eases up and down in finite time without overshoot', () => {
  for (const [from,to] of [[0,100],[100,0],[15,85],[90,20]]) {
    const previous=retargetLiquidFill(null,from,0);
    const tween=retargetLiquidFill(previous,to,1);
    assert.equal(liquidFillAt(tween,1),from);
    assert.ok(Math.abs(liquidFillAt(tween,1+liquidTransitionSeconds/2)-(from+to)/2)<1e-12);
    let last=from;
    for (let t=0; t<=2; t+=.01) {
      const value=liquidFillAt(tween,1+t);
      assert.ok(value>=Math.min(from,to) && value<=Math.max(from,to));
      assert.ok(to>from ? value>=last : value<=last);
      last=value;
    }
    assert.equal(liquidFillAt(tween,3),to);
    assert.equal(liquidFillAt(tween,20),to);
  }
});
test('new readings retarget from the visible level; repeated renders do not restart easing', () => {
  const initial=retargetLiquidFill(null,20,0);
  const rising=retargetLiquidFill(initial,90,1);
  assert.equal(retargetLiquidFill(rising,90,1.4),rising);
  const falling=retargetLiquidFill(rising,10,1.7);
  assert.equal(liquidFillAt(falling,1.7),liquidFillAt(rising,1.7));
  assert.equal(liquidFillAt(falling,5),10);
});
test('missing data clears fill immediately; recovery and reduced motion use the real reading', () => {
  const initial=retargetLiquidFill(null,75,0);
  for (const load of [null,undefined,NaN,-1,101]) {
    assert.equal(retargetLiquidFill(initial,load,.1),null);
  }
  assert.equal(liquidFillAt(null,1),null);
  assert.equal(liquidFillAt(retargetLiquidFill(null,40,2),2),40);
  assert.equal(liquidFillAt(retargetLiquidFill(initial,5,2,true),2),5);
});
test('liquid geometry stays bounded, depth sorted and deterministic', () => {
  for (let rank=0; rank<5; rank++) for (let t=0; t<180; t+=3.7) {
    const scene=liquidScene(t,rank,50);
    const faces=[...scene.back,...scene.water,...scene.front];
    assert.ok(faces.length > 300 && faces.length <= 400);
    assert.ok(faces.every(f => f.light >= 0 && f.light <= 1));
    assert.ok(faces.every(f => f.points.length >= 2 && f.points.length <= 13 && f.points.every(p =>
      Number.isFinite(p.z) && p.x > 8 && p.x < 320 && p.y > 4 && p.y < 267)));
    assert.ok(scene.water.every((f,i) => !i || scene.water[i-1].depth <= f.depth));
  }
  assert.deepEqual(liquidScene(5,1,100), liquidScene(5,1,100));
});
test('tank keeps only its six face outlines with no internal grid paths', () => {
  for (const load of [null,0,50,100]) for (let rank=0; rank<5; rank++) {
    const scene=liquidScene(5,rank,load);
    const outlines=[...scene.back,...scene.front].filter(f=>f.kind!=='waterline');
    assert.equal(outlines.length,6);
    assert.ok(outlines.every(f=>f.kind==='cage-edge' && f.close && f.points.length===4));
  }
});
test('waves preserve actual liquid volume at every load and never leave the cube', () => {
  for (const rank of [0,1,2,3,4]) for (const t of [0,2,30,250]) for (let load=0; load<=100; load++) {
    const surface=liquidSurface(t,rank,load);
    assert.equal(surface.level,load/100);
    assert.ok(Math.abs(surfaceMean(surface.heights)-load/100) < 1e-12);
    assert.ok(surface.heights.every(h=>h>=0 && h<=1));
  }
  for (const load of [.00001,99.99999]) assert.ok(liquidSurface(4,0,load).heights.every(h=>h>=0 && h<=1));
});
test('perspective tank has converging edges, not orthographic parallelograms', () => {
  const faces=liquidScene(0,0,100).front.filter(f=>f.kind==='cage-edge');
  const area = f => Math.abs(f.points.reduce((sum,p,i) => {
    const next=f.points[(i+1)%4]; return sum+p.x*next.y-next.x*p.y;
  },0))/2;
  assert.ok(faces.every(f => area(f) > 0));
  assert.ok(faces.some(f => {
    const [a,b,c,d]=f.points;
    return Math.abs((b.x-a.x)*(d.y-c.y)-(b.y-a.y)*(d.x-c.x)) > 1;
  }));
});
test('constant full load still animates without manufacturing utilization', () => {
  assert.notDeepEqual(liquidScene(0,0,100), liquidScene(2,0,100));
  assert.ok(liquidSurface(3,0,100).heights.every(h=>h===1));
  assert.deepEqual(gpuSignal(100), {known:true,level:1});
});
test('partial loads have moving wave peaks as well as troughs around the true level', () => {
  const a=liquidSurface(0,0,50), b=liquidSurface(.7,0,50);
  assert.notDeepEqual(a.heights,b.heights);
  assert.ok(Math.max(...a.heights)>.55 && Math.min(...a.heights)<.45);
  assert.ok(liquidScene(3,0,50).front.some(f=>f.kind==='waterline'));
});
test('unknown and zero GPU retain static empty cages without liquid or invented activity', () => {
  for (const load of [null,undefined,NaN,-1,101]) {
    assert.equal(liquidSurface(3,0,load),null);
    const scene=liquidScene(3,0,load);
    assert.equal(scene.known,false);
    assert.equal(scene.water.length,0);
    assert.equal(scene.front.length+scene.back.length,6);
    assert.deepEqual(scene,liquidScene(30,0,load));
  }
  assert.deepEqual(liquidScene(5,0,0),liquidScene(30,0,0));
  assert.equal(liquidScene(5,0,0).water.length,0);
  assert.equal(liquidScene(5,0,0).known,true);
});
test('each rank has a distinct decorative phase', () => {
  const cores = [0,1,2,3,4].map(rank => JSON.stringify(liquidScene(3,rank,100)));
  assert.equal(new Set(cores).size, 5);
});
test('RAM meter uses actual system memory; missing never becomes a reading', () => {
  assert.deepEqual(memoryLevel(24,96), {known:true,percent:25});
  assert.deepEqual(memoryLevel(0,96), {known:true,percent:0});
  assert.deepEqual(memoryLevel(96,96), {known:true,percent:100});
  for (const [used,total] of [[null,96],[10,0],[97,96],[-1,96],[Infinity,96]]) {
    assert.deepEqual(memoryLevel(used,total), {known:false,percent:0});
  }
});
