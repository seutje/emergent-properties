# Emergent Properties — 3D Web-Based Audio Visualizer — Detailed Design Document

## 0) Executive Summary
A static Three.js + TensorFlow.js web app structured as **separate ES modules** (no build step). The HTML page includes **one** `<script type="module" src="./js/main.js"></script>` entry point that imports all other modules. The app renders a large 3D field of equally spaced particles; each responds **slightly differently** to audio in real time. An **MLP** (TensorFlow.js) maps **(particle_state, audio_features) → Δposition, color, flicker_rate, size** per particle.

- **Minimal deps (CDN):** three.js, OrbitControls, TensorFlow.js, lil‑gui (or dat.GUI), optional tiny-colormap.
- **Audio sources:** bundled MP3s + user uploads (drag & drop + file picker).
- **Controls:** GUI exposes every tweakable parameter (min/max/scale/weights‑seed/model rate, etc.).
- **Performance:** batched model inference each frame (or at a decimated rate), Instanced/Points rendering, efficient attribute updates.

---

## 1) Goals & Non‑Goals
**Goals**
1. Static, portable, single-page app (index.html + assets/).
2. Smooth, interactive 3D visualization of ~10k–100k particles (target 30–60 FPS on typical laptops).
3. Real-time audio feature extraction via Web Audio API.
4. Per-particle response variety; deterministic differences (seeded) and optional noise.
5. Full GUI to tune rendering, audio feature scaling, and MLP output ranges.

**Non‑Goals**
- No server, backend, or model training pipeline.
- No persistent user accounts; optional localStorage for presets only.
- No heavy build tools. ES modules loaded via CDN only.

---

## 2) System Architecture
**Coding convention:** **Classical inheritance is preferred** (ES6 classes with `extends`), with clear base classes for shared behavior (e.g., `BaseModule`, `BaseController`).

**High-level flow**
```
Audio Source (MP3/Upload/Mic?) → Web Audio API Graph → AnalyserNode/Feature Extractor
        ↓                                             ↑
     Feature Vector (per frame/decimated)             |
        ↓                                             |
Particle State (static + dynamic)  →  Batch Input → TensorFlow.js MLP → Batch Output
        ↓                                                             ↓
   Per-Particle Attributes (Δpos, color, flicker, size) → GPU via Three.js (instanced/points)
```

**Key runtime modules**
- **AudioManager**: loads/plays MP3, file uploads, creates `AudioContext` + `AnalyserNode`.
- **FeatureExtractor**: computes RMS, spectral centroid, rolloff, band energies (low/mid/high), peak, tempo proxy.
- **ParticleField**: creates geometry + per-instance attributes (position, phase, idHash, baseColor, etc.).
- **MLPModel**: tf.js `Sequential` model, small; supports random seed init & parameter GUI exposure (layers, activations, scaling).
- **Renderer**: Three.js scene/camera/OrbitControls/render loop; maintains `time` uniform; applies batched updates.
- **UIController**: lil‑gui controls; loads/saves presets; binds to modules.

---

## 3) Dependencies (CDN)
- **Three.js** (ESM): `https://unpkg.com/three@0.160.0/build/three.module.js`
- **OrbitControls**: `https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js`
- **TensorFlow.js**: `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js`
- **lil-gui**: `https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.umd.min.js`
- **(Optional) tinycolor2 / colormap** for palettes

**Why these:** battle-tested, tree-shakable (ESM), reliable CDNs, no build step.

---

## 4) File/Folder Structure (Static Hosting)
```
/ (repo root)
  index.html                 # one <script type="module" src="./js/main.js"></script>
  /assets
     /audio                  # bundled MP3s
       track1.mp3
       track2.mp3
     /icons                  # favicon etc.
  /styles
     main.css                # minimal CSS (or inline)
  /js                        # ES module files (no build step)
     main.js                 # entry point — wires modules together
     core/
       BaseModule.js         # abstract base (lifecycle/logging); preferred classical inheritance
     audio/
       AudioManager.js       # loads/plays audio, analyser graph
       FeatureExtractor.js   # derives audio features from AnalyserNode
     render/
       Renderer.js           # scene/camera/loop; OrbitControls
       ParticleField.js      # geometry, attributes, GPU buffers
       Shaders.js            # shader strings & builders
     ml/
       MLPModel.js           # tf.js model creation & batched inference
     ui/
       UIController.js       # lil‑gui controls, presets, bindings
  README.md
  DESIGN.md                  # this document
  PLAN.md                    # phased planning document
```

---

## 5) Data Models
### 5.1 Particle State (per instance)
- `gridIndex` = (ix, iy, iz) → derived position = baseGridPos
- `basePos` = vec3 (normalized to [-1,1] cube)
- `id` = integer (0..N-1)
- `idHash` = pseudo-random in [0,1] seeded by `id` and global seed
- `phase` = random [0, 2π]
- `prevVel` = vec3 (smoothed)
- `distOrigin` = |basePos|
- `age` = seconds since start (optional)

These are packed into GPU attributes and/or included in MLP input.

### 5.2 Audio Feature Vector (per frame)
- `rms` (energy)
- `specCentroid`
- `specRolloff`
- `bandLow`, `bandMid`, `bandHigh` (sum/avg energy in FFT bands)
- `peak` (max magnitude)
- `zeroCrossRate` (approx)
- `tempoProxy` (EMA of beat peaks) — optional, cheap
- Normalization: min/max & z-score toggles, GUI-exposed.

### 5.3 MLP Inputs & Outputs
**Input vector (concat)**
```
[x, y, z, distOrigin, idHash, phaseSin, phaseCos, prevSpeed,
 rms, centroid, rolloff, low, mid, high, peak, zcr, tempo]
```
(~17–20 dims depending on toggles)

**Output vector**
```
ΔposX, ΔposY, ΔposZ, sizeDelta, hueDelta, satDelta, lightDelta,
 flickerRate, flickerDepth
```
(8–10 dims; all ranges GUI-scaled)

**Activations**
- Hidden: `tanh` (cheap, bounded)
- Output: mixed (e.g., `tanh` for deltas; `sigmoid` for rates/depths, later scaled)

---

## 6) Rendering Strategy
### Option A: **THREE.Points + custom ShaderMaterial** (Recommended)
- Geometry: `BufferGeometry` with `position` (base grid pos) and custom attributes:
  - `aPhase`, `aIdHash`, `aBaseColor`, `aSizeBase`, `aFlickerRate`, `aFlickerDepth`, `aDeltaPos`, `aColorDelta`, `aSizeDelta`.
- Uniforms: `uTime`, `uGlobalScale`, `uBloomOn`, etc.
- Vertex shader offsets position by `aDeltaPos` and modulates `gl_PointSize`.
- Fragment shader computes color = base ± delta, flicker with `sin(uTime * rate + phase)`.
- Pros: handles 50k–150k particles, single draw call, per-vertex attributes.

### Option B: **InstancedMesh of small spheres**
- Pros: nicer look with lighting.
- Cons: heavier; fewer instances (~5k–20k). Use if visual priority > count.

**Chosen default:** Option A (Points) for scale; add a toggle to switch to InstancedMesh in advanced settings.

---

## 7) Audio Pipeline (Web Audio API)
1. `AudioContext` created on user gesture (Play button).
2. Source options:
   - **Bundled MP3**: fetched via `fetch` → `decodeAudioData` → `AudioBufferSourceNode`.
   - **User Upload**: `input[type=file]` & drag-drop → `decodeAudioData`.
   - (Optional) Microphone: `getUserMedia` toggle.
3. Graph: `Source → GainNode (master) → AnalyserNode → destination`.
4. Analyser config: `fftSize=2048`, `smoothingTimeConstant=0.8`.
5. Feature extraction executed per animation frame or at a fixed cadence (e.g., 30 Hz) with EMA smoothing.

---

## 8) TensorFlow.js MLP
**Model**
```js
const model = tf.sequential({layers:[
  tf.layers.dense({units: H, inputShape: [IN], activation:'tanh', useBias:true}),
  tf.layers.dense({units: H, activation:'tanh', useBias:true}),
  tf.layers.dense({units: OUT, activation:'tanh', useBias:true}),
]});
```
- H default = 32; IN≈18; OUT≈9 (configurable).
- Weights initialized with seeded PRNG (GUI seed) → `tf.initializers.randomUniform({minval:-0.5,maxval:0.5,seed})`.

**Batching**
- Build tensor of shape `[N, IN]` every k frames (k=1..4 via GUI).
- Single `model.predict(input)` yields `[N, OUT]`.
- Convert to typed arrays once; update GPU attributes with `BufferAttribute.needsUpdate = true`.

**Performance hints**
- Prefer WebGL backend (`tf.setBackend('webgl')` fallback to `wasm`/`cpu`).
- Decimate MLP rate (e.g., run at 30 Hz) and interpolate deltas in shader or JS.
- Clamp N based on hardware (GUI slider with FPS indicator).

---

## 9) GUI / Controls (lil‑gui)
### 9.1 Playback & Audio
- **Source**: dropdown (Bundled 1/2/3, Upload, Mic)
- **Transport**: Play/Pause/Stop, Seek, Volume, Loop
- **Feature Smoothing**: window (ms), EMA α

### 9.2 Camera & Scene
- **Camera**: orbit sensitivity, auto-rotate toggle & speed
- **Scene**: background color/gradient, fog toggle/intensity, bloom postFX toggle (if implemented)

### 9.3 Particles
- **Layout**: gridX, gridY, gridZ; spacing; jitter
- **Count**: N (derived from grid dims)
- **Base Size**: px
- **Base Color**: palette selector / color picker
- **Noise**: per-particle randomization depth

### 9.4 Feature Scaling (per feature)
For each: `rms`, `centroid`, `rolloff`, `low`, `mid`, `high`, `peak`, `zcr`, `tempo`
- **min**, **max**, **scale** (gain), **bias**, **enable** (checkbox)
- Normalization mode: raw / min-max / z-score (global live stats)

### 9.5 MLP
- **Seed** (integer), **Hidden Units** (8–64), **Layers** (1–3), **Activation** (tanh/relu)
- **Backend** (auto/webgl/wasm/cpu), **Rate** (Hz)
- **Input Mask** (toggle particle/audio dims)

### 9.6 Output Mapping (global clamps)
- **ΔPosition**: maxX/Y/Z, global scale
- **Size Delta**: min/max, response curve (linear/exp)
- **Color Δ (H/S/L or RGB)**: min/max per channel
- **Flicker Rate**: min/max (Hz)
- **Flicker Depth**: 0..1
- **Blend**: mix model vs. baseline (0..1)

### 9.7 Advanced
- **Renderer**: Points vs InstancedMesh
- **Attributes Update**: full vs partial (update only changed ranges)
- **MLP Interpolation**: on/off
- **FPS Display**: on/off
- **Reset**: to defaults
- **Presets**: save/load (localStorage JSON)

---

## 10) UI/UX
- Clean top control bar (source + transport); lil‑gui docked right.
- Drag & drop overlay for uploads.
- Visual meters: small spectrum & RMS bar.
- Mobile: fallback to fewer particles, larger UI.
- Accessibility: keyboard focus for Play/Stop; colorblind-safe palette presets.

---

## 11) Shaders (Points Path)
**Vertex**
```
attribute vec3 aDeltaPos;
attribute float aSizeBase;
attribute float aSizeDelta;
attribute float aFlickerRate;
attribute float aPhase;
uniform float uTime;
...
void main() {
  float flicker = 0.5 + 0.5 * sin(uTime * aFlickerRate + aPhase);
  vec3 pos = position + aDeltaPos * flicker; // or mix with baseline
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  gl_PointSize = (aSizeBase + aSizeDelta * flicker) * uPointScale; // scale by perspective in JS
}
```

**Fragment**
```
precision mediump float;
attribute vec3 aBaseColor;
attribute vec3 aColorDelta;
...
void main(){
  vec3 col = aBaseColor + aColorDelta * /* optional flicker */;
  // circular point sprite
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float alpha = smoothstep(1.0, 0.9, 1.0 - dot(uv,uv));
  gl_FragColor = vec4(col, alpha);
}
```

---

## 12) Performance Plan
- Default N = 30k points; cap at 150k; adaptive downshift if FPS < threshold.
- Use `requestAnimationFrame` once; decimate feature & model updates.
- Update BufferAttributes with `usage = DynamicDrawUsage`; avoid re-allocs.
- Avoid per-particle JS objects in hot loops; rely on typed arrays.
- Optionally move some math (interpolation) to shaders.

---

## 13) Persistence & Presets
- Save/load JSON to `localStorage` (GUI button).
- Include a few curated presets (e.g., "Calm Field", "Bass Warp", "Glitter Grid").

---

## 14) Error Handling & Edge Cases
- Autoplay restriction: show Play gate; resume context on gesture.
- Uploaded file errors: invalid codec → toast message.
- Large files: show load spinner, decode progress (best-effort).
- Backend selection fails → fall back to `cpu`.

---

## 15) Security & Privacy
- All in-browser; no upload to server.
- For mic input, prompt permission; provide clear on/off toggle.

---

## 16) Testing Matrix
- Chrome, Firefox, Safari (desktop); recent iOS/Android.
- GPU low-power vs dGPU; window resize; device pixel ratio > 1.
- MP3s with different sample rates; long tracks; seeking.

---

## 17) Implementation Sketch (Modular, Single Include)
### 17.1 `index.html`
```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>3D Audio Visualizer</title>
  <link rel="stylesheet" href="./styles/main.css"/>
  <!-- CDN dependencies -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js"></script>
  <script type="module" src="./js/main.js"></script>
</head>
<body>
  <div id="app"></div>
  <input id="file" type="file" accept="audio/*" hidden/>
</body>
</html>
```

### 17.2 `js/core/BaseModule.js`
```js
export class BaseModule {
  constructor(name){ this.name = name; }
  init(){ /* optional */ }
  dispose(){ /* optional */ }
  log(...a){ console.log(`[${this.name}]`, ...a); }
}
```

### 17.3 `js/render/Renderer.js`
```js
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { BaseModule } from '../core/BaseModule.js';

export class Renderer extends BaseModule {
  constructor(container){ super('Renderer'); this.container = container; }
  init(){
    this.renderer = new THREE.WebGLRenderer({antialias:true});
    this.container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    this.camera.position.set(0,0,6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    window.addEventListener('resize', ()=>this.resize()); this.resize();
  }
  resize(){
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(w,h,false); this.camera.aspect = w/h; this.camera.updateProjectionMatrix();
  }
  render(){ this.controls.update(); this.renderer.render(this.scene, this.camera); }
}
```

### 17.4 `js/ml/MLPModel.js`
```js
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js';
import { BaseModule } from '../core/BaseModule.js';

export class MLPModel extends BaseModule {
  constructor(cfg){ super('MLPModel'); this.cfg = cfg; this.model = null; }
  async init(){
    const IN=this.cfg.inputSize, H=this.cfg.hidden, OUT=this.cfg.outputSize;
    this.model = tf.sequential({layers:[
      tf.layers.dense({units:H, inputShape:[IN], activation:'tanh'}),
      tf.layers.dense({units:H, activation:'tanh'}),
      tf.layers.dense({units:OUT, activation:'tanh'})
    ]});
  }
  predict(batch){ return tf.tidy(()=> this.model.predict(batch)); }
}
```

### 17.5 `js/main.js` (entry)
```js
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.umd.min.js';
import { Renderer } from './render/Renderer.js';
// import other modules (AudioManager, FeatureExtractor, ParticleField, UIController, etc.)

const app = document.getElementById('app');
const renderer = new Renderer(app); renderer.init();
const gui = new GUI();

let last=performance.now();
function loop(t){
  const dt=(t-last)/1000; last=t;
  // TODO: pull features → model → update particle attributes
  renderer.render();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
```

*(Other module files follow the same pattern; see folder structure.)*

---

## 18) Roadmap
- **MVP**: Points renderer, bundled track playback, feature extraction, MLP inference (30 Hz), GUI for output clamps.
- **v1**: Uploads, presets, palette system, InstancedMesh toggle, mic input.
- **v1.1**: PostFX (bloom), tempo proxy refinement, shader color blending modes.

---

## 19) Acceptance Criteria
1. Runs from localhost with a simple node.js HTTP server or from GitHub Pages.
2. Default preset achieves stable 45–60 FPS with 30k particles on mid-tier laptop.
3. Switching tracks or uploads updates features without reload.
4. Every listed parameter exposed in GUI with min/max/scale/bias.
5. Camera orbit/pan/zoom works; scene is navigable.

---

## 20) Risks & Mitigations
- **Performance cliffs** → adaptive particle count, decimated model rate.
- **Autoplay restrictions** → user gate UI.
- **Device variability** → presets tuned per platform (desktop/mobile flags).
- **TensorFlow backend quirks** → backend selector + fallback.

---

## 21) Glossary
- **Δ**: per-frame change (applied additively or scaled by flicker/time).
- **Decimation**: running compute at lower-than-render framerate.
- **EMA**: exponential moving average (feature smoothing).

