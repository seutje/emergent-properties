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

## 2025-11-07
- Added the `@tensorflow/tfjs-node` backend for Jest runs plus a Node-aware loader inside `MLPModel` so automated tests use the accelerated native backend without console nags.
- Updated the ML unit tests to target the `tensorflow` backend, keeping coverage for rebuild flows while avoiding the previous runtime warning.

## 2025-11-07
- Executed Phase 6 UI work: introduced a full `UIController` that groups audio, renderer, and model controls, surfaces curated presets (with save/load via localStorage), and wires those bindings directly into `ParticleField`, `FeatureExtractor`, and MLP modules.
- Added a reusable `PresetStore` helper plus Jest coverage to lock down preset persistence logic, and exposed new particle tuning setters + render stats to keep the GUI state reliable.
- Shipped a closable onboarding overlay that points users toward drag-and-drop uploads and the presets panel, with matching glassmorphic styles to stay on-brand.

## 2025-11-07
- Boosted audioreactivity by adding a feature-driven envelope in `MLPOrchestrator` that dynamically scales MLP deltas, flicker, and blend smoothing; exposed the new knobs via lil-gui and wired them into preset capture/apply flows plus PLAN.md.
- Added Jest coverage for the new `deriveReactivity` helper to lock down gain/blend behavior across spikes and quiet passages, and updated CI to keep the added test suite green.

## 2025-11-07
- Delivered the in-browser training workflow: created a TensorFlow.js web worker pipeline (`MLPTrainingWorker`) that synthesizes datasets from user-specified feature↔particle correlations, runs pause/resume/abort-able training loops, and reports achieved correlation scores plus best checkpoints back to the main thread.
- Added `MLPTrainingManager`, correlation targets/utils, weight serialization helpers, and wired the manager into `main.js` so freshly trained weights auto-apply to the live orchestrator while staying exportable/importable (JSON snapshots with metadata).
- Built a dedicated glassmorphic Training Panel UI with dynamic correlation editors, training controls, progress readouts, result summaries, and model import/export inputs; updated PLAN.md and stylesheet to reflect the new workflow.
- Extended Jest coverage via `MLPTrainingUtils.test.js` to lock down dataset generation, correlation math, and evaluation helpers.

## 2025-11-07
- Updated the default preset to the new art-directed baseline: switched the TensorFlow backend to `webgl`, enabled `relu` activations, maxed the model blend, and retuned the orchestration clamps/reactivity envelope to the requested ranges so the GUI loads with the approved look/feel without needing preset swaps.

## 2025-11-07
- Added a particle seed control: `ParticleField` can now regenerate its seeded layout on demand, lil-gui exposes the seed so users can dial in deterministic variations, and reseeding keeps geometry/delta buffers in sync.
- Wired `UIController` to notify `MLPOrchestrator.refreshParticleState`, implemented that refresh hook so the MLP rebuilds its static tensors/flicker baselines after reseeds, and extended `MLPOrchestrator.test.js` to lock down the new behavior.
