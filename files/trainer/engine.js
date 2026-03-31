// engine.js — WebGPU training engine, fully self-contained ES module

export async function createEngine(opts = {}) {
  // Delegate to cached factory to avoid re-declaration issues
  return _createEngineImpl(opts);
}

export async function _createEngineImpl({ onStep, config: cfg = {}, weights: savedWeights = null } = {}) {
  // Coerce string numbers from agent output
  const num = v => typeof v === 'string' ? parseFloat(v) : v;
  
  const V     = 72;
  const D     = num(cfg.D)       ?? 64;
  const H     = num(cfg.H)       ?? 4;
  const L     = num(cfg.L)       ?? 2;
  const T     = 64;
  const HD    = D / H;
  const FF    = D * (num(cfg.FF_mult) ?? 4);
  const BATCH = num(cfg.batch)   ?? 4;
  const BT    = BATCH * T;
  const LR    = num(cfg.lr)      ?? 3e-4;
  const BETA1 = num(cfg.beta1)   ?? 0.9;
  const BETA2 = num(cfg.beta2)   ?? 0.95;
  const EPS   = 1e-8;
  const WD    = num(cfg.wd)      ?? 0.1;
  const ACT   = cfg.activation   ?? 'relu';
}

// ── GPU helpers ───────────────────────────────────────────────────────────────
function makeHelpers(device) {
  function upload(data) {
    const arr = new Float32Array(data);
    const buf = device.createBuffer({ size: Math.max(arr.byteLength,16),
      usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true });
    new Float32Array(buf.getMappedRange()).set(arr); buf.unmap(); return buf;
  }
  function uploadInt(data) {
    const arr = new Int32Array(data);
    const buf = device.createBuffer({ size: Math.max(arr.byteLength,16),
      usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true });
    new Int32Array(buf.getMappedRange()).set(arr); buf.unmap(); return buf;
  }
  function empty(n) { return device.createBuffer({ size: Math.max(n*4,16),
    usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST }); }
  function zeros(n) {
    const buf = device.createBuffer({ size: Math.max(n*4,16),
      usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,
      mappedAtCreation: true });
    new Float32Array(buf.getMappedRange()).fill(0); buf.unmap(); return buf;
  }
  function uni(arr) {
    const buf = device.createBuffer({ size: Math.max(arr.byteLength,16),
      usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf,0,arr); return buf;
  }
  async function read(buf, n) {
    const s = device.createBuffer({ size: n*4,
      usage: GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(buf,0,s,0,n*4);
    device.queue.submit([e.finish()]);
    await s.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(s.getMappedRange().slice(0));
    s.unmap(); s.destroy(); return r;
  }
  function pipe(code) { return device.createComputePipeline({ layout:'auto',
    compute:{ module:device.createShaderModule({code}), entryPoint:'main' }}); }
  function mkbg(p, bufs) { return device.createBindGroup({ layout:p.getBindGroupLayout(0),
    entries:bufs.map((buffer,binding)=>({binding,resource:{buffer}}))}); }
  function run(enc, p, bg, x, y=1) {
    const pass=enc.beginComputePass();
    pass.setPipeline(p); pass.setBindGroup(0,bg);
    pass.dispatchWorkgroups(x,y); pass.end();
  }
  return { upload, uploadInt, empty, zeros, uni, read, pipe, mkbg, run };
}

// ── Pipelines ─────────────────────────────────────────────────────────────────
function makePipelines(device, pipe) {
  return {
    EMBED: pipe(`
      @group(0) @binding(0) var<storage,read> tokens: array<i32>;
      @group(0) @binding(1) var<storage,read> tokE: array<f32>;
      @group(0) @binding(2) var<storage,read> posE: array<f32>;
      @group(0) @binding(3) var<storage,read_write> x: array<f32>;
      @group(0) @binding(4) var<uniform> u: vec4<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let bt=g.x; let d=g.y;
        if(bt>=u.x||d>=u.z){return;}
        x[bt*u.z+d]=tokE[u32(tokens[bt])*u.z+d]+posE[(bt%u.y)*u.z+d];
      }
    `),
    LN: pipe(`
      struct Dim{N:u32,C:u32}
      @group(0) @binding(0) var<storage,read> x: array<f32>;
      @group(0) @binding(1) var<storage,read> w: array<f32>;
      @group(0) @binding(2) var<storage,read> b: array<f32>;
      @group(0) @binding(3) var<storage,read_write> y: array<f32>;
      @group(0) @binding(4) var<storage,read_write> mean: array<f32>;
      @group(0) @binding(5) var<storage,read_write> rstd: array<f32>;
      @group(0) @binding(6) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row=g.x; if(row>=d.N){return;}
        let off=row*d.C;
        var m=0.0; for(var i=0u;i<d.C;i++){m+=x[off+i];} m/=f32(d.C);
        var v=0.0; for(var i=0u;i<d.C;i++){let dd=x[off+i]-m;v+=dd*dd;} v/=f32(d.C);
        let rs=1.0/sqrt(v+1e-5); mean[row]=m; rstd[row]=rs;
        for(var i=0u;i<d.C;i++){y[off+i]=(x[off+i]-m)*rs*w[i]+b[i];}
      }
    `),
    LN_BWD_DX: pipe(`
      struct Dim{N:u32,C:u32}
      @group(0) @binding(0) var<storage,read> x: array<f32>;
      @group(0) @binding(1) var<storage,read> w: array<f32>;
      @group(0) @binding(2) var<storage,read> dout: array<f32>;
      @group(0) @binding(3) var<storage,read> mean: array<f32>;
      @group(0) @binding(4) var<storage,read> rstd: array<f32>;
      @group(0) @binding(5) var<storage,read_write> dx: array<f32>;
      @group(0) @binding(6) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row=g.x; if(row>=d.N){return;}
        let off=row*d.C; let m=mean[row]; let rs=rstd[row];
        var sd=0.0; var sdx=0.0;
        for(var i=0u;i<d.C;i++){
          let xh=(x[off+i]-m)*rs; let dy=dout[off+i]*w[i];
          sd+=dy; sdx+=dy*xh;
        }
        let sc=rs/f32(d.C);
        for(var i=0u;i<d.C;i++){
          let xh=(x[off+i]-m)*rs;
          dx[off+i]+=sc*(f32(d.C)*dout[off+i]*w[i]-sd-xh*sdx);
        }
      }
    `),
    LN_BWD_DWB: pipe(`
      struct Dim{N:u32,C:u32}
      @group(0) @binding(0) var<storage,read> x: array<f32>;
      @group(0) @binding(1) var<storage,read> dout: array<f32>;
      @group(0) @binding(2) var<storage,read> mean: array<f32>;
      @group(0) @binding(3) var<storage,read> rstd: array<f32>;
      @group(0) @binding(4) var<storage,read_write> dw: array<f32>;
      @group(0) @binding(5) var<storage,read_write> db: array<f32>;
      @group(0) @binding(6) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let col=g.x; if(col>=d.C){return;}
        var sdw=0.0; var sdb=0.0;
        for(var row=0u;row<d.N;row++){
          let off=row*d.C; let xh=(x[off+col]-mean[row])*rstd[row];
          sdw+=dout[off+col]*xh; sdb+=dout[off+col];
        }
        dw[col]+=sdw; db[col]+=sdb;
      }
    `),
    MM: pipe(`
      struct Dim{M:u32,K:u32,N:u32}
      @group(0) @binding(0) var<storage,read> A: array<f32>;
      @group(0) @binding(1) var<storage,read> B: array<f32>;
      @group(0) @binding(2) var<storage,read> bias: array<f32>;
      @group(0) @binding(3) var<storage,read_write> C: array<f32>;
      @group(0) @binding(4) var<uniform> d: Dim;
      @compute @workgroup_size(16,16)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row=g.y; let col=g.x; if(row>=d.M||col>=d.N){return;}
        var s=0.0; for(var k=0u;k<d.K;k++){s+=A[row*d.K+k]*B[col*d.K+k];}
        C[row*d.N+col]=s+bias[col];
      }
    `),
    MM_NB: pipe(`
      struct Dim{M:u32,K:u32,N:u32}
      @group(0) @binding(0) var<storage,read> A: array<f32>;
      @group(0) @binding(1) var<storage,read> B: array<f32>;
      @group(0) @binding(2) var<storage,read_write> C: array<f32>;
      @group(0) @binding(3) var<uniform> d: Dim;
      @compute @workgroup_size(16,16)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row=g.y; let col=g.x; if(row>=d.M||col>=d.N){return;}
        var s=0.0; for(var k=0u;k<d.K;k++){s+=A[row*d.K+k]*B[col*d.K+k];}
        C[row*d.N+col]=s;
      }
    `),
    MM_BWD_DA: pipe(`
      struct Dim{M:u32,K:u32,N:u32}
      @group(0) @binding(0) var<storage,read> dC: array<f32>;
      @group(0) @binding(1) var<storage,read> B: array<f32>;
      @group(0) @binding(2) var<storage,read_write> dA: array<f32>;
      @group(0) @binding(3) var<uniform> d: Dim;
      @compute @workgroup_size(16,16)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let row=g.y; let col=g.x; if(row>=d.M||col>=d.K){return;}
        var s=0.0; for(var n=0u;n<d.N;n++){s+=dC[row*d.N+n]*B[n*d.K+col];}
        dA[row*d.K+col]+=s;
      }
    `),
    MM_BWD_DB: pipe(`
      struct Dim{M:u32,K:u32,N:u32}
      @group(0) @binding(0) var<storage,read> dC: array<f32>;
      @group(0) @binding(1) var<storage,read> A: array<f32>;
      @group(0) @binding(2) var<storage,read_write> dB: array<f32>;
      @group(0) @binding(3) var<uniform> d: Dim;
      @compute @workgroup_size(16,16)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let n=g.y; let k=g.x; if(n>=d.N||k>=d.K){return;}
        var s=0.0; for(var m=0u;m<d.M;m++){s+=dC[m*d.N+n]*A[m*d.K+k];}
        dB[n*d.K+k]+=s;
      }
    `),
    MM_BWD_DBIAS: pipe(`
      struct Dim{M:u32,N:u32}
      @group(0) @binding(0) var<storage,read> dC: array<f32>;
      @group(0) @binding(1) var<storage,read_write> dbias: array<f32>;
      @group(0) @binding(2) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let n=g.x; if(n>=d.N){return;}
        var s=0.0; for(var m=0u;m<d.M;m++){s+=dC[m*d.N+n];}
        dbias[n]+=s;
      }
    `),
    RELU: pipe(ACT === 'gelu' ? `
      @group(0) @binding(0) var<storage,read_write> x: array<f32>;
      @group(0) @binding(1) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;}
        let v=x[g.x];
        x[g.x]=0.5*v*(1.0+tanh(0.7978845608*(v+0.044715*v*v*v)));
      }
    ` : ACT === 'silu' ? `
      @group(0) @binding(0) var<storage,read_write> x: array<f32>;
      @group(0) @binding(1) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;}
        let v=x[g.x];
        x[g.x]=v/(1.0+exp(-v));
      }
    ` : `
      @group(0) @binding(0) var<storage,read_write> x: array<f32>;
      @group(0) @binding(1) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;} x[g.x]=max(0.0,x[g.x]);
      }
    `),
    RELU_BWD: pipe(`
      @group(0) @binding(0) var<storage,read> x: array<f32>;
      @group(0) @binding(1) var<storage,read> dout: array<f32>;
      @group(0) @binding(2) var<storage,read_write> dx: array<f32>;
      @group(0) @binding(3) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;}
        dx[g.x]+=select(0.0,dout[g.x],x[g.x]>0.0);
      }
    `),
    ADD: pipe(`
      @group(0) @binding(0) var<storage,read_write> x: array<f32>;
      @group(0) @binding(1) var<storage,read> y: array<f32>;
      @group(0) @binding(2) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;} x[g.x]+=y[g.x];
      }
    `),
    COPY: pipe(`
      @group(0) @binding(0) var<storage,read> src: array<f32>;
      @group(0) @binding(1) var<storage,read_write> dst: array<f32>;
      @group(0) @binding(2) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        if(g.x>=N){return;} dst[g.x]=src[g.x];
      }
    `),
    ATTN_FWD: pipe(`
      struct Dim{B:u32,H:u32,T:u32,HD:u32}
      @group(0) @binding(0) var<storage,read> qkv: array<f32>;
      @group(0) @binding(1) var<storage,read_write> out: array<f32>;
      @group(0) @binding(2) var<storage,read_write> att: array<f32>;
      @group(0) @binding(3) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let flat=g.x; if(flat>=d.B*d.H*d.T){return;}
        let b=flat/(d.H*d.T); let h=(flat/d.T)%d.H; let tq=flat%d.T;
        let DD=d.H*d.HD; let scale=1.0/sqrt(f32(d.HD));
        var sc: array<f32,${T}>;
        for(var tk=0u;tk<d.T;tk++){
          if(tk>tq){sc[tk]=-1e9;continue;}
          var dot=0.0;
          let qo=(b*d.T+tq)*DD*3u+h*d.HD;
          let ko=(b*d.T+tk)*DD*3u+DD+h*d.HD;
          for(var dd=0u;dd<d.HD;dd++){dot+=qkv[qo+dd]*qkv[ko+dd];}
          sc[tk]=dot*scale;
        }
        var mx=sc[0]; for(var tk=1u;tk<d.T;tk++){mx=max(mx,sc[tk]);}
        var se=0.0; for(var tk=0u;tk<d.T;tk++){sc[tk]=exp(sc[tk]-mx);se+=sc[tk];}
        for(var tk=0u;tk<d.T;tk++){sc[tk]/=se;}
        for(var tk=0u;tk<d.T;tk++){att[flat*d.T+tk]=sc[tk];}
        for(var dd=0u;dd<d.HD;dd++){
          var val=0.0;
          for(var tk=0u;tk<d.T;tk++){
            let vo=(b*d.T+tk)*DD*3u+2u*DD+h*d.HD;
            val+=sc[tk]*qkv[vo+dd];
          }
          out[(b*d.T+tq)*DD+h*d.HD+dd]=val;
        }
      }
    `),
    ATTN_BWD_DQ: pipe(`
      struct Dim{B:u32,H:u32,T:u32,HD:u32}
      @group(0) @binding(0) var<storage,read> qkv: array<f32>;
      @group(0) @binding(1) var<storage,read> att: array<f32>;
      @group(0) @binding(2) var<storage,read> dout: array<f32>;
      @group(0) @binding(3) var<storage,read_write> dqkv: array<f32>;
      @group(0) @binding(4) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let flat=g.x; if(flat>=d.B*d.H*d.T){return;}
        let b=flat/(d.H*d.T); let h=(flat/d.T)%d.H; let tq=flat%d.T;
        let DD=d.H*d.HD; let scale=1.0/sqrt(f32(d.HD));
        let aoff=flat*d.T;
        var datt: array<f32,${T}>;
        for(var tk=0u;tk<=tq;tk++){
          var s=0.0;
          let vo=(b*d.T+tk)*DD*3u+2u*DD+h*d.HD;
          let doo=(b*d.T+tq)*DD+h*d.HD;
          for(var dd=0u;dd<d.HD;dd++){s+=dout[doo+dd]*qkv[vo+dd];}
          datt[tk]=s;
        }
        var dot2=0.0;
        for(var tk=0u;tk<=tq;tk++){dot2+=att[aoff+tk]*datt[tk];}
        let qo=(b*d.T+tq)*DD*3u+h*d.HD;
        for(var tk=0u;tk<=tq;tk++){
          let dsc=att[aoff+tk]*(datt[tk]-dot2)*scale;
          let ko=(b*d.T+tk)*DD*3u+DD+h*d.HD;
          for(var dd=0u;dd<d.HD;dd++){dqkv[qo+dd]+=dsc*qkv[ko+dd];}
        }
      }
    `),
    ATTN_BWD_DKV: pipe(`
      struct Dim{B:u32,H:u32,T:u32,HD:u32}
      @group(0) @binding(0) var<storage,read> qkv: array<f32>;
      @group(0) @binding(1) var<storage,read> att: array<f32>;
      @group(0) @binding(2) var<storage,read> dout: array<f32>;
      @group(0) @binding(3) var<storage,read_write> dqkv: array<f32>;
      @group(0) @binding(4) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let flat=g.x; if(flat>=d.B*d.H*d.T){return;}
        let b=flat/(d.H*d.T); let h=(flat/d.T)%d.H; let tk=flat%d.T;
        let DD=d.H*d.HD; let scale=1.0/sqrt(f32(d.HD));
        let vo=(b*d.T+tk)*DD*3u+2u*DD+h*d.HD;
        let ko=(b*d.T+tk)*DD*3u+DD+h*d.HD;
        for(var tq=tk;tq<d.T;tq++){
          let flat_q=b*d.H*d.T+h*d.T+tq;
          let att_val=att[flat_q*d.T+tk];
          let doo=(b*d.T+tq)*DD+h*d.HD;
          for(var dd=0u;dd<d.HD;dd++){dqkv[vo+dd]+=att_val*dout[doo+dd];}
          var datt_tk=0.0;
          for(var dd=0u;dd<d.HD;dd++){datt_tk+=dout[doo+dd]*qkv[vo+dd];}
          var dot_tq=0.0;
          for(var tk2=0u;tk2<=tq;tk2++){
            let v2=(b*d.T+tk2)*DD*3u+2u*DD+h*d.HD;
            var datt2=0.0;
            for(var dd=0u;dd<d.HD;dd++){datt2+=dout[doo+dd]*qkv[v2+dd];}
            dot_tq+=att[flat_q*d.T+tk2]*datt2;
          }
          let dsc=att_val*(datt_tk-dot_tq)*scale;
          let qo=(b*d.T+tq)*DD*3u+h*d.HD;
          for(var dd=0u;dd<d.HD;dd++){dqkv[ko+dd]+=dsc*qkv[qo+dd];}
        }
      }
    `),
    CE: pipe(`
      struct Dim{BT:u32,V:u32}
      @group(0) @binding(0) var<storage,read> logits: array<f32>;
      @group(0) @binding(1) var<storage,read> targets: array<i32>;
      @group(0) @binding(2) var<storage,read_write> losses: array<f32>;
      @group(0) @binding(3) var<storage,read_write> dlogits: array<f32>;
      @group(0) @binding(4) var<uniform> d: Dim;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let tok=g.x; if(tok>=d.BT){return;}
        let off=tok*d.V;
        var mx=logits[off]; for(var v=1u;v<d.V;v++){mx=max(mx,logits[off+v]);}
        var se=0.0; for(var v=0u;v<d.V;v++){se+=exp(logits[off+v]-mx);}
        let tgt=targets[tok];
        losses[tok]=-(logits[off+u32(tgt)]-mx)+log(se);
        let sc=1.0/f32(d.BT);
        for(var v=0u;v<d.V;v++){
          let prob=exp(logits[off+v]-mx)/se;
          dlogits[off+v]=sc*(prob-select(0.0,1.0,i32(v)==tgt));
        }
      }
    `),
    EMBED_BWD: pipe(`
      @group(0) @binding(0) var<storage,read> dx: array<f32>;
      @group(0) @binding(1) var<storage,read> tokens: array<i32>;
      @group(0) @binding(2) var<storage,read_write> dTokEmb: array<f32>;
      @group(0) @binding(3) var<storage,read_write> dPosEmb: array<f32>;
      @group(0) @binding(4) var<uniform> u: vec4<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        let d=g.x; if(d>=u.z){return;}
        for(var t=0u;t<u.y;t++){
          var sum=0.0;
          for(var b=0u;b<u.x/u.y;b++){sum+=dx[(b*u.y+t)*u.z+d];}
          dPosEmb[t*u.z+d]+=sum;
        }
        for(var v=0u;v<u.w;v++){
          var sum=0.0;
          for(var bt=0u;bt<u.x;bt++){
            if(u32(tokens[bt])==v){sum+=dx[bt*u.z+d];}
          }
          dTokEmb[v*u.z+d]+=sum;
        }
      }
    `),
    ADAMW: pipe(`
      struct H{lr:f32,b1:f32,b2:f32,eps:f32,wd:f32,step:f32,_p0:f32,_p1:f32}
      @group(0) @binding(0) var<storage,read_write> p: array<f32>;
      @group(0) @binding(1) var<storage,read> g: array<f32>;
      @group(0) @binding(2) var<storage,read_write> m: array<f32>;
      @group(0) @binding(3) var<storage,read_write> v: array<f32>;
      @group(0) @binding(4) var<uniform> h: H;
      @group(0) @binding(5) var<uniform> N: u32;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i=gid.x; if(i>=N){return;}
        let gv=g[i];
        let mn=h.b1*m[i]+(1.0-h.b1)*gv;
        let vn=h.b2*v[i]+(1.0-h.b2)*gv*gv;
        m[i]=mn; v[i]=vn;
        let mh=mn/(1.0-pow(h.b1,h.step));
        let vh=vn/(1.0-pow(h.b2,h.step));
        p[i]=p[i]*(1.0-h.lr*h.wd)-h.lr*mh/(sqrt(vh)+h.eps);
      }
    `),
  };
}

// ── Weight init ───────────────────────────────────────────────────────────────
function randn(n, std=0.02) {
  const a = new Float32Array(n);
  for (let i=0; i<n; i+=2) {
    const u1=Math.random(), u2=Math.random();
    const r=std*Math.sqrt(-2*Math.log(u1+1e-10));
    a[i]=r*Math.cos(2*Math.PI*u2);
    if(i+1<n) a[i+1]=r*Math.sin(2*Math.PI*u2);
  }
  return a;
}

// ── Main export ───────────────────────────────────────────────────────────────
export async function createEngine({ onStep } = {}) {
  // Init WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU not available');
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });

  // Load data
  const chunks = await fetch('/data/train_chunks.json').then(r => r.json());

  // Load tokenizer for sampling
  const tokenizer = await fetch('/data/tokenizer.json').then(r => r.json());
  const vocab = tokenizer.vocab;
  const charToIdx = tokenizer.char_to_idx;

  // Build helpers and pipelines
  const h = makeHelpers(device);
  const P = makePipelines(device, h.pipe);

  // Model params
  function makeParam(data) {
    return { w: h.upload(data), m: h.zeros(data.length), v: h.zeros(data.length), n: data.length };
  }
  const ones  = n => new Float32Array(n).fill(1);
  const oz    = n => new Float32Array(n).fill(0);

  const params = {
    tokEmb: makeParam(randn(V*D)),
    posEmb: makeParam(randn(T*D)),
    lnFW:   makeParam(ones(D)),
    lnFB:   makeParam(oz(D)),
    headW:  makeParam(randn(V*D)),
  };
  const blks = Array.from({length:L}, () => ({
    ln1W: makeParam(ones(D)),     ln1B: makeParam(oz(D)),
    qkvW: makeParam(randn(3*D*D)),qkvB: makeParam(oz(3*D)),
    outW: makeParam(randn(D*D)),  outB: makeParam(oz(D)),
    ln2W: makeParam(ones(D)),     ln2B: makeParam(oz(D)),
    ff1W: makeParam(randn(FF*D)), ff1B: makeParam(oz(FF)),
    ff2W: makeParam(randn(D*FF)), ff2B: makeParam(oz(D)),
  }));

  // Fixed uniforms
  const U = {
    embed:      h.uni(new Uint32Array([BT,T,D,0])),
    BTD:        h.uni(new Uint32Array([BT*D])),
    BTFF:       h.uni(new Uint32Array([BT*FF])),
    lnBTD:      h.uni(new Uint32Array([BT,D])),
    attn:       h.uni(new Uint32Array([BATCH,H,T,HD])),
    CE:         h.uni(new Uint32Array([BT,V])),
    embedBwd:   h.uni(new Uint32Array([BT,T,D,V])),
    mmQKV:      h.uni(new Uint32Array([BT,D,3*D])),
    mmOutP:     h.uni(new Uint32Array([BT,D,D])),
    mmLin1:     h.uni(new Uint32Array([BT,D,FF])),
    mmLin2:     h.uni(new Uint32Array([BT,FF,D])),
    mmHead:     h.uni(new Uint32Array([BT,D,V])),
    mmBwdQKV_DA:  h.uni(new Uint32Array([BT,D,3*D])),
    mmBwdQKV_DB:  h.uni(new Uint32Array([BT,D,3*D])),
    mmBwdOutP_DA: h.uni(new Uint32Array([BT,D,D])),
    mmBwdOutP_DB: h.uni(new Uint32Array([BT,D,D])),
    mmBwdLin1_DA: h.uni(new Uint32Array([BT,D,FF])),
    mmBwdLin1_DB: h.uni(new Uint32Array([BT,D,FF])),
    mmBwdLin2_DA: h.uni(new Uint32Array([BT,FF,D])),
    mmBwdLin2_DB: h.uni(new Uint32Array([BT,FF,D])),
    mmBwdHead_DA: h.uni(new Uint32Array([BT,D,V])),
    mmBwdHead_DB: h.uni(new Uint32Array([BT,D,V])),
  };

  // Activation buffers (reused each step)
  const act = {
    x:       h.empty(BT*D),
    lnFOut:  h.empty(BT*D), lnFMean: h.empty(BT), lnFRstd: h.empty(BT),
    logits:  h.empty(BT*V),
    losses:  h.empty(BT),
  };
  const saved = Array.from({length:L}, () => ({
    xIn:     h.empty(BT*D),
    ln1Out:  h.empty(BT*D), ln1Mean: h.empty(BT), ln1Rstd: h.empty(BT),
    qkv:     h.empty(BT*3*D),
    attnOut: h.empty(BT*D), att: h.empty(BATCH*H*T*T),
    outProj: h.empty(BT*D),
    xAfterAttn: h.empty(BT*D),
    ln2Out:  h.empty(BT*D), ln2Mean: h.empty(BT), ln2Rstd: h.empty(BT),
    ff1Out:  h.empty(BT*FF),
    ff2Out:  h.empty(BT*D),
  }));

  // Data sampling
  function sampleBatch() {
    const tokens  = new Int32Array(BT);
    const targets = new Int32Array(BT);
    for (let b=0; b<BATCH; b++) {
      const chunk = chunks[Math.floor(Math.random()*chunks.length)];
      for (let t=0; t<T; t++) {
        tokens [b*T+t] = chunk[t];
        targets[b*T+t] = chunk[t+1];
      }
    }
    return { tokens, targets };
  }

  // AdamW step
  function adamwStep(enc, param, grad, step) {
    const hh = h.uni(new Float32Array([LR,BETA1,BETA2,EPS,WD,step,0,0]));
    const n  = h.uni(new Uint32Array([param.n]));
    h.run(enc, P.ADAMW, h.mkbg(P.ADAMW,[param.w,grad,param.m,param.v,hh,n]),
      Math.ceil(param.n/64));
  }

  // Single training step
  async function step(stepCount) {
    const { tokens, targets } = sampleBatch();
    const bufTokens  = h.uploadInt(tokens);
    const bufTargets = h.uploadInt(targets);
    const bufDLogits = h.zeros(BT*V);

    // Grad buffers
    const dX      = h.zeros(BT*D);
    const dLnFOut = h.zeros(BT*D);
    const dHeadW  = h.zeros(V*D);
    const dLnFW   = h.zeros(D);
    const dLnFB   = h.zeros(D);
    const dTokEmb = h.zeros(V*D);
    const dPosEmb = h.zeros(T*D);
    const dBlks = Array.from({length:L}, () => ({
      ln1W: h.zeros(D),    ln1B: h.zeros(D),
      qkvW: h.zeros(3*D*D),qkvB: h.zeros(3*D),
      outW: h.zeros(D*D),  outB: h.zeros(D),
      ln2W: h.zeros(D),    ln2B: h.zeros(D),
      ff1W: h.zeros(FF*D), ff1B: h.zeros(FF),
      ff2W: h.zeros(D*FF), ff2B: h.zeros(D),
    }));

    const enc = device.createCommandEncoder();
    const { run, mkbg, zeros, uni } = h;

    // Forward
    run(enc,P.EMBED,mkbg(P.EMBED,[bufTokens,params.tokEmb.w,params.posEmb.w,act.x,U.embed]),
      Math.ceil(BT/64),D);
    for (let l=0;l<L;l++) {
      const bl=blks[l], sv=saved[l];
      run(enc,P.COPY,mkbg(P.COPY,[act.x,sv.xIn,U.BTD]),Math.ceil(BT*D/64));
      run(enc,P.LN,mkbg(P.LN,[act.x,bl.ln1W.w,bl.ln1B.w,sv.ln1Out,sv.ln1Mean,sv.ln1Rstd,U.lnBTD]),Math.ceil(BT/64));
      run(enc,P.MM,mkbg(P.MM,[sv.ln1Out,bl.qkvW.w,bl.qkvB.w,sv.qkv,U.mmQKV]),Math.ceil(3*D/16),Math.ceil(BT/16));
      run(enc,P.ATTN_FWD,mkbg(P.ATTN_FWD,[sv.qkv,sv.attnOut,sv.att,U.attn]),Math.ceil(BATCH*H*T/64));
      run(enc,P.MM,mkbg(P.MM,[sv.attnOut,bl.outW.w,bl.outB.w,sv.outProj,U.mmOutP]),Math.ceil(D/16),Math.ceil(BT/16));
      run(enc,P.ADD,mkbg(P.ADD,[act.x,sv.outProj,U.BTD]),Math.ceil(BT*D/64));
      run(enc,P.COPY,mkbg(P.COPY,[act.x,sv.xAfterAttn,U.BTD]),Math.ceil(BT*D/64));
      run(enc,P.LN,mkbg(P.LN,[act.x,bl.ln2W.w,bl.ln2B.w,sv.ln2Out,sv.ln2Mean,sv.ln2Rstd,U.lnBTD]),Math.ceil(BT/64));
      run(enc,P.MM,mkbg(P.MM,[sv.ln2Out,bl.ff1W.w,bl.ff1B.w,sv.ff1Out,U.mmLin1]),Math.ceil(FF/16),Math.ceil(BT/16));
      run(enc,P.RELU,mkbg(P.RELU,[sv.ff1Out,U.BTFF]),Math.ceil(BT*FF/64));
      run(enc,P.MM,mkbg(P.MM,[sv.ff1Out,bl.ff2W.w,bl.ff2B.w,sv.ff2Out,U.mmLin2]),Math.ceil(D/16),Math.ceil(BT/16));
      run(enc,P.ADD,mkbg(P.ADD,[act.x,sv.ff2Out,U.BTD]),Math.ceil(BT*D/64));
    }
    run(enc,P.LN,mkbg(P.LN,[act.x,params.lnFW.w,params.lnFB.w,act.lnFOut,act.lnFMean,act.lnFRstd,U.lnBTD]),Math.ceil(BT/64));
    run(enc,P.MM_NB,mkbg(P.MM_NB,[act.lnFOut,params.headW.w,act.logits,U.mmHead]),Math.ceil(V/16),Math.ceil(BT/16));
    run(enc,P.CE,mkbg(P.CE,[act.logits,bufTargets,act.losses,bufDLogits,U.CE]),Math.ceil(BT/64));

    // Backward
    run(enc,P.MM_BWD_DB,mkbg(P.MM_BWD_DB,[bufDLogits,act.lnFOut,dHeadW,U.mmBwdHead_DB]),Math.ceil(D/16),Math.ceil(V/16));
    run(enc,P.MM_BWD_DA,mkbg(P.MM_BWD_DA,[bufDLogits,params.headW.w,dLnFOut,U.mmBwdHead_DA]),Math.ceil(D/16),Math.ceil(BT/16));
    run(enc,P.LN_BWD_DX,mkbg(P.LN_BWD_DX,[act.x,params.lnFW.w,dLnFOut,act.lnFMean,act.lnFRstd,dX,U.lnBTD]),Math.ceil(BT/64));
    run(enc,P.LN_BWD_DWB,mkbg(P.LN_BWD_DWB,[act.x,dLnFOut,act.lnFMean,act.lnFRstd,dLnFW,dLnFB,U.lnBTD]),Math.ceil(D/64));

    for (let l=L-1;l>=0;l--) {
      const bl=blks[l], sv=saved[l], db=dBlks[l];
      const dFF2Out=zeros(BT*D);
      run(enc,P.COPY,mkbg(P.COPY,[dX,dFF2Out,U.BTD]),Math.ceil(BT*D/64));
      const dFF1Act=zeros(BT*FF);
      run(enc,P.MM_BWD_DA,mkbg(P.MM_BWD_DA,[dFF2Out,bl.ff2W.w,dFF1Act,U.mmBwdLin2_DA]),Math.ceil(FF/16),Math.ceil(BT/16));
      run(enc,P.MM_BWD_DB,mkbg(P.MM_BWD_DB,[dFF2Out,sv.ff1Out,db.ff2W,U.mmBwdLin2_DB]),Math.ceil(FF/16),Math.ceil(D/16));
      run(enc,P.MM_BWD_DBIAS,mkbg(P.MM_BWD_DBIAS,[dFF2Out,db.ff2B,uni(new Uint32Array([BT,D]))]),Math.ceil(D/64));
      const dFF1Pre=zeros(BT*FF);
      run(enc,P.RELU_BWD,mkbg(P.RELU_BWD,[sv.ff1Out,dFF1Act,dFF1Pre,U.BTFF]),Math.ceil(BT*FF/64));
      run(enc,P.MM_BWD_DB,mkbg(P.MM_BWD_DB,[dFF1Pre,sv.ln2Out,db.ff1W,U.mmBwdLin1_DB]),Math.ceil(D/16),Math.ceil(FF/16));
      run(enc,P.MM_BWD_DBIAS,mkbg(P.MM_BWD_DBIAS,[dFF1Pre,db.ff1B,uni(new Uint32Array([BT,FF]))]),Math.ceil(FF/64));
      const dLn2Out=zeros(BT*D);
      run(enc,P.MM_BWD_DA,mkbg(P.MM_BWD_DA,[dFF1Pre,bl.ff1W.w,dLn2Out,U.mmBwdLin1_DA]),Math.ceil(D/16),Math.ceil(BT/16));
      run(enc,P.LN_BWD_DX,mkbg(P.LN_BWD_DX,[sv.xAfterAttn,bl.ln2W.w,dLn2Out,sv.ln2Mean,sv.ln2Rstd,dX,U.lnBTD]),Math.ceil(BT/64));
      run(enc,P.LN_BWD_DWB,mkbg(P.LN_BWD_DWB,[sv.xAfterAttn,dLn2Out,sv.ln2Mean,sv.ln2Rstd,db.ln2W,db.ln2B,U.lnBTD]),Math.ceil(D/64));
      const dOutProj=zeros(BT*D);
      run(enc,P.COPY,mkbg(P.COPY,[dX,dOutProj,U.BTD]),Math.ceil(BT*D/64));
      run(enc,P.MM_BWD_DB,mkbg(P.MM_BWD_DB,[dOutProj,sv.attnOut,db.outW,U.mmBwdOutP_DB]),Math.ceil(D/16),Math.ceil(D/16));
      run(enc,P.MM_BWD_DBIAS,mkbg(P.MM_BWD_DBIAS,[dOutProj,db.outB,uni(new Uint32Array([BT,D]))]),Math.ceil(D/64));
      const dAttnOut=zeros(BT*D);
      run(enc,P.MM_BWD_DA,mkbg(P.MM_BWD_DA,[dOutProj,bl.outW.w,dAttnOut,U.mmBwdOutP_DA]),Math.ceil(D/16),Math.ceil(BT/16));
      const dQKV=zeros(BT*3*D);
      run(enc,P.ATTN_BWD_DQ,mkbg(P.ATTN_BWD_DQ,[sv.qkv,sv.att,dAttnOut,dQKV,U.attn]),Math.ceil(BATCH*H*T/64));
      run(enc,P.ATTN_BWD_DKV,mkbg(P.ATTN_BWD_DKV,[sv.qkv,sv.att,dAttnOut,dQKV,U.attn]),Math.ceil(BATCH*H*T/64));
      run(enc,P.MM_BWD_DB,mkbg(P.MM_BWD_DB,[dQKV,sv.ln1Out,db.qkvW,U.mmBwdQKV_DB]),Math.ceil(D/16),Math.ceil(3*D/16));
      run(enc,P.MM_BWD_DBIAS,mkbg(P.MM_BWD_DBIAS,[dQKV,db.qkvB,uni(new Uint32Array([BT,3*D]))]),Math.ceil(3*D/64));
      const dLn1Out=zeros(BT*D);
      run(enc,P.MM_BWD_DA,mkbg(P.MM_BWD_DA,[dQKV,bl.qkvW.w,dLn1Out,U.mmBwdQKV_DA]),Math.ceil(D/16),Math.ceil(BT/16));
      run(enc,P.LN_BWD_DX,mkbg(P.LN_BWD_DX,[sv.xIn,bl.ln1W.w,dLn1Out,sv.ln1Mean,sv.ln1Rstd,dX,U.lnBTD]),Math.ceil(BT/64));
      run(enc,P.LN_BWD_DWB,mkbg(P.LN_BWD_DWB,[sv.xIn,dLn1Out,sv.ln1Mean,sv.ln1Rstd,db.ln1W,db.ln1B,U.lnBTD]),Math.ceil(D/64));
    }

    run(enc,P.EMBED_BWD,mkbg(P.EMBED_BWD,[dX,bufTokens,dTokEmb,dPosEmb,U.embedBwd]),Math.ceil(D/64));

    // AdamW
    const s = stepCount + 1;
    adamwStep(enc,params.tokEmb,dTokEmb,s);
    adamwStep(enc,params.posEmb,dPosEmb,s);
    adamwStep(enc,params.lnFW,  dLnFW,  s);
    adamwStep(enc,params.lnFB,  dLnFB,  s);
    adamwStep(enc,params.headW, dHeadW, s);
    for (let l=0;l<L;l++) {
      const bl=blks[l], db=dBlks[l];
      adamwStep(enc,bl.ln1W,db.ln1W,s); adamwStep(enc,bl.ln1B,db.ln1B,s);
      adamwStep(enc,bl.qkvW,db.qkvW,s); adamwStep(enc,bl.qkvB,db.qkvB,s);
      adamwStep(enc,bl.outW,db.outW,s); adamwStep(enc,bl.outB,db.outB,s);
      adamwStep(enc,bl.ln2W,db.ln2W,s); adamwStep(enc,bl.ln2B,db.ln2B,s);
      adamwStep(enc,bl.ff1W,db.ff1W,s); adamwStep(enc,bl.ff1B,db.ff1B,s);
      adamwStep(enc,bl.ff2W,db.ff2W,s); adamwStep(enc,bl.ff2B,db.ff2B,s);
    }

    device.queue.submit([enc.finish()]);

    const losses = await h.read(act.losses, BT);
    const loss = losses.reduce((a,b)=>a+b,0)/BT;
    if (onStep) onStep(loss, stepCount);
    return loss;
  }

  // Character-level sampling
  async function sample(prefix='Once', maxLen=200) {
    const ids = [...prefix].map(c => charToIdx[c] ?? 1);

    for (let gen=0; gen<maxLen; gen++) {
      const contextIds = ids.slice(-T);
      const padded = new Int32Array(T);
      contextIds.forEach((id,i) => padded[i] = id);

      const bufTokens = h.uploadInt(padded);
      const bufX2     = h.empty(T*D);
      const lnFOut2   = h.empty(T*D);
      const lnFMean2  = h.empty(T);
      const lnFRstd2  = h.empty(T);
      const logits2   = h.empty(T*V);

      const uE2 = h.uni(new Uint32Array([T,T,D,0]));
      const uL2 = h.uni(new Uint32Array([T,D]));
      const uA2 = h.uni(new Uint32Array([1,H,T,HD]));
      const mH2 = h.uni(new Uint32Array([T,D,V]));
      const sv2 = Array.from({length:L}, () => ({
        xIn:        h.empty(T*D),
        ln1Out:     h.empty(T*D), ln1Mean: h.empty(T), ln1Rstd: h.empty(T),
        qkv:        h.empty(T*3*D),
        attnOut:    h.empty(T*D), att: h.empty(H*T*T),
        outProj:    h.empty(T*D),
        xAfterAttn: h.empty(T*D),
        ln2Out:     h.empty(T*D), ln2Mean: h.empty(T), ln2Rstd: h.empty(T),
        ff1Out:     h.empty(T*FF),
        ff2Out:     h.empty(T*D),
      }));

      const mmQ2  = h.uni(new Uint32Array([T,D,3*D]));
      const mmO2  = h.uni(new Uint32Array([T,D,D]));
      const mmL12 = h.uni(new Uint32Array([T,D,FF]));
      const mmL22 = h.uni(new Uint32Array([T,FF,D]));
      const uFF2  = h.uni(new Uint32Array([T*FF]));
      const uBTD2 = h.uni(new Uint32Array([T*D]));

      const enc2 = device.createCommandEncoder();
      const { run, mkbg } = h;

      run(enc2,P.EMBED,mkbg(P.EMBED,[bufTokens,params.tokEmb.w,params.posEmb.w,bufX2,uE2]),
        Math.ceil(T/64),D);
      for (let l=0;l<L;l++) {
        const bl=blks[l], sv=sv2[l];
        run(enc2,P.COPY,mkbg(P.COPY,[bufX2,sv.xIn,uBTD2]),Math.ceil(T*D/64));
        run(enc2,P.LN,mkbg(P.LN,[bufX2,bl.ln1W.w,bl.ln1B.w,sv.ln1Out,sv.ln1Mean,sv.ln1Rstd,uL2]),Math.ceil(T/64));
        run(enc2,P.MM,mkbg(P.MM,[sv.ln1Out,bl.qkvW.w,bl.qkvB.w,sv.qkv,mmQ2]),Math.ceil(3*D/16),Math.ceil(T/16));
        run(enc2,P.ATTN_FWD,mkbg(P.ATTN_FWD,[sv.qkv,sv.attnOut,sv.att,uA2]),Math.ceil(H*T/64));
        run(enc2,P.MM,mkbg(P.MM,[sv.attnOut,bl.outW.w,bl.outB.w,sv.outProj,mmO2]),Math.ceil(D/16),Math.ceil(T/16));
        run(enc2,P.ADD,mkbg(P.ADD,[bufX2,sv.outProj,uBTD2]),Math.ceil(T*D/64));
        run(enc2,P.COPY,mkbg(P.COPY,[bufX2,sv.xAfterAttn,uBTD2]),Math.ceil(T*D/64));
        run(enc2,P.LN,mkbg(P.LN,[bufX2,bl.ln2W.w,bl.ln2B.w,sv.ln2Out,sv.ln2Mean,sv.ln2Rstd,uL2]),Math.ceil(T/64));
        run(enc2,P.MM,mkbg(P.MM,[sv.ln2Out,bl.ff1W.w,bl.ff1B.w,sv.ff1Out,mmL12]),Math.ceil(FF/16),Math.ceil(T/16));
        run(enc2,P.RELU,mkbg(P.RELU,[sv.ff1Out,uFF2]),Math.ceil(T*FF/64));
        run(enc2,P.MM,mkbg(P.MM,[sv.ff1Out,bl.ff2W.w,bl.ff2B.w,sv.ff2Out,mmL22]),Math.ceil(D/16),Math.ceil(T/16));
        run(enc2,P.ADD,mkbg(P.ADD,[bufX2,sv.ff2Out,uBTD2]),Math.ceil(T*D/64));
      }
      run(enc2,P.LN,mkbg(P.LN,[bufX2,params.lnFW.w,params.lnFB.w,lnFOut2,lnFMean2,lnFRstd2,uL2]),Math.ceil(T/64));
      run(enc2,P.MM_NB,mkbg(P.MM_NB,[lnFOut2,params.headW.w,logits2,mH2]),Math.ceil(V/16),Math.ceil(T/16));
      device.queue.submit([enc2.finish()]);

      // Read logits for last token position
      const lastPos = contextIds.length - 1;
      const allLogits = await h.read(logits2, T*V);
      const logitsLast = allLogits.slice(lastPos*V, (lastPos+1)*V);

      // Temperature sampling (temp=0.8)
      const temp = 0.8;
      const scaled = logitsLast.map(l => l/temp);
      const mx = Math.max(...scaled);
      const exps = scaled.map(l => Math.exp(l-mx));
      const sum = exps.reduce((a,b)=>a+b,0);
      const probs = exps.map(e => e/sum);

      // Sample from distribution
      let r = Math.random(), cumulative = 0, nextId = 0;
      for (let i=0; i<probs.length; i++) {
        cumulative += probs[i];
        if (r <= cumulative) { nextId = i; break; }
      }

      ids.push(nextId);
      if (nextId === 0) break; // pad token = stop
    }

    return ids.map(id => vocab[id] ?? '?').join('');
  }

  // ── Weight export (GPU → CPU arrays for persistence) ─────────────────────
  async function exportWeights() {
    const out = {};
    async function readParam(name, param) {
      out[name] = await h.read(param.w, param.n);
    }
    await readParam('tokEmb', params.tokEmb);
    await readParam('posEmb', params.posEmb);
    await readParam('lnFW',   params.lnFW);
    await readParam('lnFB',   params.lnFB);
    await readParam('headW',  params.headW);
    for (let l = 0; l < L; l++) {
      const bl = blks[l];
      for (const [k, v] of Object.entries(bl)) {
        await readParam(`blk${l}_${k}`, v);
      }
    }
    return out;
  }

  // ── Weight import (CPU arrays → GPU) ─────────────────────────────────────
  function importWeights(saved) {
    function loadParam(name, param) {
      if (!saved[name]) return;
      device.queue.writeBuffer(param.w, 0, saved[name]);
    }
    loadParam('tokEmb', params.tokEmb);
    loadParam('posEmb', params.posEmb);
    loadParam('lnFW',   params.lnFW);
    loadParam('lnFB',   params.lnFB);
    loadParam('headW',  params.headW);
    for (let l = 0; l < L; l++) {
      const bl = blks[l];
      for (const [k, v] of Object.entries(bl)) {
        loadParam(`blk${l}_${k}`, v);
      }
    }
  }

  // Load saved weights if provided
  if (savedWeights) {
    importWeights(savedWeights);
    console.log('Weights restored from saved session');
  }

  return { step, sample, exportWeights, importWeights };

}