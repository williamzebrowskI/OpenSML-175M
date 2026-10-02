export const numeric = value => typeof value === 'number' && Number.isFinite(value);

export function displayTransform(width, height, rackPanel = false) {
  const scale = Math.min(width/356,height/280);
  // The dedicated full-display window pre-stretches the picture to undo the
  // controller's 1424x600 (or 1080p) -> 1424x280 physical-panel conversion.
  if (rackPanel) return [width/356,0,0,height/350,0,height*.1];
  return [scale,0,0,scale,(width-356*scale)/2,(height-280*scale)/2];
}

export function sensorNodes(data, now = Date.now() / 1000, online = true) {
  const hosts = data?.rack?.health?.nodes;
  return Array.from({length:5}, (_, rank) => {
    const host = Array.isArray(hosts) ? hosts.find(n => n?.rank === rank) : null;
    const age = numeric(host?.checked_at) ? Math.max(0, now-host.checked_at) : null;
    const fresh = online && age !== null && age < 35 && host.reachable === true;
    return {rank, hardware:fresh && host.hardware && typeof host.hardware === 'object' ? host.hardware : {}};
  });
}

export function memoryLevel(used, total) {
  const known = numeric(used) && numeric(total) && total > 0 && used >= 0 && used <= total;
  return {known, percent:known ? used / total * 100 : 0};
}

export function gpuSignal(load) {
  const known = numeric(load) && load >= 0 && load <= 100;
  return {known, level:known ? load/100 : 0};
}

export const liquidTransitionSeconds = 1.8;

export function liquidFillAt(transition,time) {
  if (!transition) return null;
  const p=Math.max(0,Math.min(1,(time-transition.started)/liquidTransitionSeconds));
  const eased=p*p*(3-2*p);
  return p===1 ? transition.to : transition.from+(transition.to-transition.from)*eased;
}

export function retargetLiquidFill(previous,load,time,snap=false) {
  if (!gpuSignal(load).known) return null;
  // First/recovered readings are real values, not an invented ramp from zero.
  if (snap || !previous) return {from:load,to:load,started:time};
  if (previous.to===load) return previous;
  return {from:liquidFillAt(previous,time),to:load,started:time};
}

const resolution = 12;
const clamp = x => Math.max(0,Math.min(1,x));

// Use the same triangle areas as the rendered surface, including edge vertices.
export function surfaceMean(heights) {
  const n=resolution, stride=n+1;
  let sum=0;
  for (let z=0; z<n; z++) for (let x=0; x<n; x++) {
    const a=z*stride+x, b=a+1, c=a+stride, d=c+1;
    sum+=2*heights[a]+heights[b]+heights[c]+2*heights[d];
  }
  return sum/(6*n*n);
}

export function liquidSurface(time,rank,load) {
  const gpu=gpuSignal(load);
  if (!gpu.known) return null;
  const t=numeric(time) && gpu.level>0 ? time : 0;
  const phase=rank*.91, heights=[];
  for (let z=0; z<=resolution; z++) for (let x=0; x<=resolution; x++) {
    const u=x/resolution, v=z/resolution;
    heights.push(Math.sin(u*6.2+v*2.1-t*2.0+phase)
      +.46*Math.sin(v*8.0-u*2.6+t*1.3+phase*2)
      +.18*Math.cos((u+v)*13-t*3.2));
  }
  const mean=surfaceMean(heights);
  const span=Math.max(...heights.map(h=>Math.abs(h-mean)),1e-9);
  // Waves redistribute volume rather than making measured utilization oscillate.
  const amplitude=Math.min(.085,gpu.level*.65,(1-gpu.level)*.65);
  return {resolution,level:gpu.level,heights:heights.map(h=>gpu.level+(h-mean)/span*amplitude)};
}

// Low-poly liquid in an outline-only tank. No fluid simulation or WebGL.
export function liquidScene(time,rank,load) {
  const gpu=gpuSignal(load), t=gpu.known && gpu.level>0 && numeric(time) ? time : 0;
  const surface=liquidSurface(t,rank,load), phase=rank*.91;
  const yaw=.62+.14*Math.sin(t*.18+phase), pitch=.37;
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const back=[], front=[], water=[];
  const half=2.5, camera=15, scale=30.7;
  const rotate=([x,y,z])=>{
    const rx=x*cy+z*sy, rz=z*cy-x*sy;
    return [rx,y*cp-rz*sp,y*sp+rz*cp];
  };
  const project=p=>{
    const [x,y,z]=rotate(p), perspective=camera/(camera-z);
    return {x:166+x*scale*perspective,y:127-y*scale*perspective,z};
  };
  function polygon(points,kind,light,lit=true,close=true) {
    const projected=points.map(project);
    return {points:projected,kind,light:clamp(light),lit,close,
      depth:projected.reduce((sum,p)=>sum+p.z,0)/projected.length};
  }
  for (const axis of [0,1,2]) for (const sign of [-1,1]) {
    const normal=[0,0,0]; normal[axis]=sign;
    const [nx,ny,nz]=rotate(normal), center=rotate(normal.map(v=>v*half));
    const visible=nx*-center[0]+ny*-center[1]+nz*(camera-center[2])>0;
    const group=visible ? front : back;
    const dims=[0,1,2].filter(i=>i!==axis);
    const point=(u,v)=>{
      const p=[0,0,0]; p[axis]=sign*half; p[dims[0]]=u; p[dims[1]]=v; return p;
    };
    group.push(polygon([point(-half,-half),point(half,-half),point(half,half),point(-half,half)],'cage-edge',0,false));
  }
  if (surface && surface.level>0) {
    const n=surface.resolution, stride=n+1;
    const point=(x,z)=>[-half+x/n*5,-half+surface.heights[z*stride+x]*5,-half+z/n*5];
    for (let z=0; z<n; z++) for (let x=0; x<n; x++) {
      const a=point(x,z), b=point(x+1,z), c=point(x,z+1), d=point(x+1,z+1);
      const crest=Math.max(0,Math.sin(x*.46+z*.33-t*2.0+phase));
      const glint=Math.pow(Math.max(0,Math.sin(x*.72-z*.51+t*1.6+phase)),10);
      const light=.58+crest*.21+glint*.21;
      water.push(polygon([a,b,d],'surface',light));
      water.push(polygon([a,d,c],'surface',light-.035));
    }
    // Visible tank walls close the volume against the exact same wave boundary.
    for (const axis of [0,2]) for (const sign of [-1,1]) {
      const normal=[0,0,0]; normal[axis]=sign;
      const [nx,ny,nz]=rotate(normal), center=rotate(normal.map(v=>v*half));
      if (nx*-center[0]+ny*-center[1]+nz*(camera-center[2])<=0) continue;
      const boundary=[];
      for (let i=0; i<=n; i++) boundary.push(axis===0 ? point(sign<0 ? 0 : n,i) : point(i,sign<0 ? 0 : n));
      for (let i=0; i<n; i++) {
        const a=boundary[i], b=boundary[i+1];
        const caustic=Math.pow(Math.max(0,Math.sin(i*.63+t*1.9+phase)),6);
        water.push(polygon([[a[0],-half,a[2]],[b[0],-half,b[2]],b,a],'liquid',.26+.10*(axis===0)+.10*caustic));
      }
      front.push(polygon(boundary,'waterline',.94,true,false));
    }
  }
  water.sort((a,b)=>a.depth-b.depth);
  return {known:gpu.known,level:gpu.level,back,water,front};
}
