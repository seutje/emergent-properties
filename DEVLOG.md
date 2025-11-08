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

## 2025-11-08
- Removed the correlation cap in `TrainingPanel` by defaulting the limit to infinity, so the Training UI now allows users to stack as many feature→parameter correlations as they need before launching a run.

## 2025-11-08
- Added a 1-second delayed auto-advance to `AudioManager` so bundled tracks seamlessly roll into the next song, and backfilled Jest coverage to lock in the behavior (with a shorter delay for the test harness).

## 2025-11-08
- Bootstrapped a `ModelSnapshotLoader` (plus Jest coverage) that fetches and validates `assets/models/default.json`, ensuring we can hydrate the curated snapshot with friendly errors when the asset is unreachable.
- Updated `main.js` to load that snapshot on startup, apply its weights/config to the live MLP, and seed the training manager with the stored correlations/options so the app defaults to the approved baseline without manual imports.

## 2025-11-08
- Switched the feature extractor’s default EMA toggle off so fresh sessions show unsmoothed feature spikes; the lil-gui control still allows enabling smoothing when desired.

## 2025-11-08
- Added a perceived-linear volume slider (defaulting to 70%) to the transport controls so listeners can dial gain without leaving the glassmorphic UI, plus matching styles and readouts to keep it on-brand.
- Reworked `AudioManager` volume handling to treat slider positions as perceived loudness (squared before hitting the gain node), expose the live volume in STATE events, and backfilled Jest coverage for the new curve.

## 2025-11-08
- Set the lil-gui controller to initialize in a closed state (with an opt-out flag) so first impressions stay focused on the scene while still letting power users pop the panel open when needed.

## 2025-11-08
- Expanded the MLP output schema + training target list to cover rotation speed, wobble strength/frequency, color mix, alpha scale, and point scale, and added a snapshot upgrader that pads legacy weights/correlations so older presets load cleanly.
- Wired those new outputs through `MLPOrchestrator`, letting the model drive ParticleField rotation/wobble/uniforms, and surfaced matching clamp controls inside lil-gui + built-in presets for save/load parity.
- Added Jest coverage for the snapshot upgrade path and the new global modulation behavior, and ran the full Jest suite to keep everything green.

## 2025-11-08
- Let the MLP steer camera zoom: added a `cameraZoom` training target with default clamps, renderer-level zoom setters/getters, and wiring through `MLPOrchestrator` so global outputs now include orbit distance.
- Updated lil-gui presets/state so camera zoom clamps save/load alongside other outputs, refreshed the snapshot upgrader’s correlations, and extended the orchestrator/unit tests; full Jest suite remains green.

## 2025-11-09
- Split the feature extractor’s low/mid spectrum bands into `bandSub`, `bandBass`, `bandLowMid`, `bandMid`, and `bandHigh`, updated the GUI labels + reactivity logic to use the richer feature set, and tuned the band defaults so tempo detection still keys off sub/bass energy.
- Augmented the snapshot upgrader to remap legacy `bandLow` correlations, pad kernels when the stored input size lags behind the new 19-dim vector, and refreshed the curated default snapshot metadata so presets reference the new band names.
- Backfilled Jest coverage across `FeatureExtractor`, `MLPOrchestrator`, and `ModelSnapshotUpgrade` to lock down the new behavior and ran those suites via `npm test -- js/audio/FeatureExtractor.test.js js/ml/MLPOrchestrator.test.js js/ml/ModelSnapshotUpgrade.test.js`.

## 2025-11-09
- Added a "Random model" control to the Training Panel that reseeds the TensorFlow model (and refreshes the orchestrator) so humans can audition fresh weight initializations before running a training session; the button disables during runs to avoid mid-training rebuilds.

## 2025-11-09
- Added a Finetune button to the training panel that snapshots the active MLP weights, feeds them into the training manager, and lets the worker resume training from the current model instead of random init—complete with button-state wiring, PLAN updates, and a full Jest run to keep coverage green.

## 2025-11-08
- Added particle positional inputs (x/y/z/dist) to the training correlations picker, plumbed the metadata through the training manager/worker so positional drivers synthesize targets correctly, and grouped the UI select into audio vs particle options.
- Updated the synthetic dataset builder + achievement metrics to normalize base particle features, propagate them through snapshots, and expanded the Jest suite (`npm test -- js/ml/MLPTrainingUtils.test.js`) to cover the new pathways.

## 2025-11-09
- Taught the app to reseed the MLP on startup and every time a new track (bundled or upload) begins by adding a shared `randomizeActiveModel` helper, wiring it through `main.js`, lil-gui, and the training panel so the UI reflects the new seeds, and extending AudioManager events with offset/resume metadata to avoid re-randomizing on pauses; new Jest coverage (`js/ml/randomizeActiveModel.test.js`, `js/audio/AudioManager.test.js`) keeps the workflow locked down.

## 2025-11-09
- Updated `AudioManager` auto-advance logic so when the final bundled track ends it loops back to the first track, keeping playlists continuous, and extended `js/audio/AudioManager.test.js` to lock in the wraparound behavior (`npm test -- js/audio/AudioManager.test.js`).
