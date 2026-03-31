// Public-facing engine API imported by the JupyterLite kernel
// Wraps the WebGPU training loop from training_loop.js logic

export async function createEngine() {
    const V = 72, D = 64, H = 4, L = 2, T = 64;
    const HD = D/H, FF = D*4;
    const BATCH = 4, BT = BATCH*T;
    const LR = 3e-4, BETA1 = 0.9, BETA2 = 0.95, EPS = 1e-8, WD = 0.1;
  
    // Load data
    const chunks = await fetch('/data/train_chunks.json').then(r => r.json());
  
    // Init WebGPU
    const adapter = await navigator.gpu.requestAdapter();
    const device  = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
  
    // Paste the full contents of training_loop.js here EXCEPT the
    // bottom section that runs the loop — just the setup + trainStep function.
    // Then expose:
  
    let stepCount = 0;
  
    return {
      async step() {
        const { tokens, targets } = sampleBatch();
        const loss = await trainStep(tokens, targets, stepCount);
        stepCount++;
        return loss;
      },
      async sample(prefix, maxLen) {
        // Character-level sampling — we'll add this next
        return prefix + '...';
      },
      get stepCount() { return stepCount; }
    };
  }