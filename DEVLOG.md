# DEVLOG

## 2025-11-07
- Completed Phase 1 renderer pass: lifecycle-aware `BaseModule`, responsive Three.js renderer, and animated placeholder particle cloud.
- Next up: human QA on camera feel/performance plus FPS instrumentation notes.

## 2025-11-07
- Added npm tooling: package manifest with Jest + http-server plus start/test scripts for consistent local serving.

## 2025-11-07
- Completed Phase 2 audio pipeline work: production-ready `AudioManager`, autoplay gate UI, bundled track selector, and drag/drop + file upload flow wired into `main.js`.
- Added dark glassmorphic transport controls + drop overlay styling per DESIGN.md vibe, plus Jest coverage for AudioManager behaviors.

## 2025-11-07
- Fulfilled Phase 3 AI tasks by replacing `FeatureExtractor` with a full spectrum/temporal analysis module (RMS, centroid, rolloff, band energies, zero-crossings, tempo proxy) plus EMA + decimation controls.
- Surfaced feature sampling controls and live readouts via lil-gui and added Jest coverage for extraction math, smoothing, and decimation flow.

## 2025-11-07
- Executed Phase 4 particle work: built a 32k+ point `ParticleField` with custom shader material, seeded grid layout, palette jitter, and deterministic attributes pulled straight from DESIGN.md §5.
- Added attribute handles/delta buffers so downstream MLP code can mutate position, color, size, and flicker, plus lil-gui-friendly reset helpers.
- Introduced `particleUtils` with deterministic generators and Jest coverage to lock in seeding + range guarantees ahead of ML integration.

## 2025-11-07
- Completed Phase 5 implementation tasks: rebuilt `MLPModel` with configurable layers/activations/backends plus Jest coverage, and added an `MLPOrchestrator` that batches particle state + audio features into tf.js tensors and applies inference deltas to the particle attributes without stalling the render loop.
- Wired the orchestrator into `main.js`, decimating inference cadence, exposing lil-gui knobs for backend, hidden layers, rate, blend, and output clamps, and ensured controls stay in sync after rebuilds to keep design intent traceable.

## 2025-11-07
- Added an FPS tracker inside the render loop and surfaced it as a read-only entry in the live feature folder so collaborators can monitor perf alongside the audio metrics.

## 2025-11-07
- Reverted the particle count tweak so the seeded grid stays at 32,768 points, and instead trimmed the default MLP to a single 32-unit hidden layer (config + GUI defaults) which measurably improves WebGL FPS by lowering inference cost without hurting visual variation.
