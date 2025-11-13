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

## 2025-11-09
- Moved the training sliders/buttons into a new lil-gui “Training” folder so epochs/batch/noise + action buttons (train, finetune, pause, resume, abort, randomize) live alongside the rest of the controls, complete with status/error readouts that stay in sync with the training manager lifecycle.
- Added event wiring so the GUI state reflects worker progress/result messages and disables buttons appropriately, then hid the redundant controls inside `TrainingPanel` (it now defaults to just correlations/export when invoked from UIController).
- Ran `npm test` to keep the Jest suite green after the UI reshuffle.

## 2025-11-09
- Embedded the remaining TrainingPanel (correlation editor, results, import/export) directly inside the lil-gui “Training” folder, so there’s no separate floating overlay and every training workflow now lives in one pane. The embedded variant reuses the existing styling but drops the redundant status bar since lil-gui now shows those stats.
- Added an inline container + CSS overrides to keep the embedded panel flush with the GUI layout, and updated the panel itself to support optional headers/status bars and arbitrary mount targets for future reuse.

## 2025-11-09
- Halved the particle field wobble by introducing a damping scalar inside `ParticleField.update`, so existing wobble settings/ML outputs now produce half the tilt without forcing preset changes.
- Skipped automated tests because the adjustment only affects the visual rotation math; verified by code inspection.

## 2025-11-09
- Added a curated snapshot pool loader that enumerates `assets/models/01-29.json`, exposes a random-pool fetcher, and covered it (plus the new snapshot import path in `randomizeActiveModel`) with dedicated Jest specs.
- Updated startup/track randomization to pull from the curated pool (falling back to seeded rebuilds on failure) so every new track spins up one of the vetted models; reran `npm test -- js/ml/ModelSnapshotLoader.test.js js/ml/randomizeActiveModel.test.js` to keep things green.

## 2025-11-09
- Let the transport remember uploaded MP3s by caching their decoded buffers, surfacing them through a new `getUploadedTracks()` API, and teaching the AudioManager tests to replay uploads via their IDs.
- Rebuilt the transport select UI to split bundled album cuts vs custom uploads into optgroups and hydrate the custom list whenever a file picker or drag/drop upload completes; reran `npm test -- js/audio/AudioManager.test.js` to keep coverage green.

## 2025-11-09
- Renamed each curated model’s `metadata.label` to match its filename (01–29) and taught the Training Panel to display the active model label, defaulting to “custom” whenever in-browser training replaces the weights.
- Wired the randomization/import pathways to update that label ribbon and added styling for the new readout; verified `npm test -- js/ml/randomizeActiveModel.test.js` after the UI changes.

## 2025-11-09
- Rerouted the Web Audio graph so `AnalyserNode` taps the raw source before it passes through the user-controlled volume gain, preventing loudness tweaks from biasing feature extraction, and backfilled coverage via `npm test -- js/audio/AudioManager.test.js`.

## 2025-11-10
- Added a Repeat toggle to the transport row so listeners can lock the current track into a loop, complete with matching styling and state wiring in `main.js`.
- Extended `AudioManager` with a persisted repeat flag that restarts finished tracks before any auto-advance kicks in, and taught the Jest suite to cover the new API/behavior.
- Ran `npm test -- js/audio/AudioManager.test.js` to keep the audio pipeline coverage green after the transport upgrade.

## 2025-11-09
- Added a configurable 1-second pre-roll inside `AudioManager` so fresh tracks pause briefly while curated/loaded models hydrate, updated progress math to avoid negative timelines during the wait, and emit the scheduled delay for downstream consumers.
- Extended `js/audio/AudioManager.test.js` with coverage for the new delay/resume behavior and reran `npm test -- js/audio/AudioManager.test.js` to keep the suite green.

## 2025-11-09
- Introduced a glyph-based text layout builder so the particle field can be remapped from its old cube into the characters of the active track, complete with deterministic seeded jitter and new defaults for text dimensions/depth.
- Added `TextParticleLayout.test.js` to lock in the glyph mask + layout math, rewired `ParticleField` to accept `setTrackLabel`, and taught `main.js` to update the field whenever `TRACK_LOADED` fires.
- Ran `npm test -- TextParticleLayout` to exercise the new unit tests.

## 2025-11-09
- Added a Next Model button to the glassmorphic transport bar that sequentially loads curated snapshots via a new `ModelCycler`, keeping the button in sync with random/autoloaded models so listeners can audition the full set without diving into lil-gui.
- Wired the button through `main.js`, updated the transport styling for the new ghost action, and covered the cycler helper with `js/ml/ModelCycler.test.js`.
- Ran `npm test -- js/ml/ModelCycler.test.js` to keep coverage green.

## 2025-11-09
- Introduced `TrackModelResolver` to map bundled tracks to deterministic curated snapshot URLs and added Jest coverage to lock the mapping rules in place.
- Updated `main.js` so bundled tracks load their mapped models while uploads continue to randomize from the curated pool; ran `npm test -- TrackModelResolver` after wiring the new flow.

## 2025-11-09
- Added a Next Track transport button between Stop and the selector that calls the new `AudioManager.playNextTrack()` API so listeners can hop through bundled + uploaded songs without touching the dropdown.
- Extended `AudioManager` with a deduped track sequence helper plus Jest coverage for the new skip behavior (including uploads/no-track edge cases) and refreshed the transport state wiring to disable the button when only one track is available.
- Ran `npm test` to cover the new behavior across the full suite.

## 2025-11-10
- Restacked the particle glyph layout so each word in the active track title gets its own centered line, keeping seeded jitter + depth but producing legible vertical titles.
- Updated `TextParticleLayout.test.js` with coverage for the multi-line, center-alignment math and reran `npm test -- TextParticleLayout` to keep the suite green.

## 2025-11-11
- Added an `OutputResponseController` layer that treats MLP predictions as targets, then applies per-channel attack/release EMA, slew limits, hysteresis, critically damped springing, and quiet-section gating before mutating renderer globals.
- Rewired `MLPOrchestrator` to register its global outputs with the controller, feed modifiers + dt into the new pass, expanded Jest coverage (`OutputResponseController.test.js`, updated orchestrator tests), and ran `npm test -- OutputResponseController` to validate the controller behaviors.

## 2025-11-11
- Extended the training defaults with explicit temporal smoothness, slew-rate, Jacobian, and noise-consistency knobs so UI/worker settings can clamp overreactions without manual tweaks.
- Rebuilt `MLPTrainingWorker`’s loop around a custom `optimizer.minimize` pass that batches data, injects Gaussian perturbations, applies TV/L2 penalties, and enforces a curriculum-weighted slew limiter plus weight decay before serializing the best snapshot.
- Introduced `MLPTrainingRegularizers` (plus Jest coverage) to keep the curriculum math + config sanitizers deterministic, and ran `npm test -- MLPTrainingRegularizers` to ensure the helpers stay green.

## 2025-11-11
- Rebuilt the audio transport markup into stacked rows (primary actions, slider band, meta row) and restyled the buttons, dropdown, repeat toggle, and status tile so they match the rectangular neon aesthetic from the reference screenshot.
- Overhauled the CSS with shared gradients/border variables, a center-aligned panel, and a horizontal slider track with a square thumb so the transport now mirrors the provided UI while staying responsive.
- Visual-only change, so no Jest suites were run for this pass.

## 2025-11-12
- Added an idle visibility controller that listens for pointer movement/hover state, hides the transport after ~3 seconds of inactivity, and immediately fades it back in as soon as the user wiggles the mouse or parks over the controls.
- Wired the controller into `main.js`, introduced the shared helper under `js/ui/`, and updated the transport CSS with the hidden state/transition styling so the bar stays interactive the moment it reappears.
- Ran `npm test -- js/ui/IdleVisibilityController.test.js` to cover the new helper behavior.

## 2025-11-12
- Implemented a particle budget controller that monitors runtime FPS, targets 60/30 FPS profiles, and dynamically dials the particle field up or down for desktop and mobile hardware.
- Rebuilt the particle field geometry pipeline so counts can be reapplied without recreating the scene, surfaced live particle stats in the GUI, and added Jest coverage for the budget controller plus mobile detection helpers.
- Ran `npm test` to verify the suite with the new performance controls.

## 2025-11-12
- Tightened the audio transport styling by lowering typography scale, trimming padding/gaps, and flattening the cards/buttons so the bar reads as a compact strip.
- Removed the rounded corners from the panel, controls, slider track, and thumb while keeping the gradient/glass treatments for continuity with the rest of the UI.
- Visual-only adjustments; Jest suites were not rerun for this pass.

## 2025-11-13
- Added an inference-aware lookahead scheduler to `MLPOrchestrator` that measures prediction latency, applies an EMA/bias clamp, and fires the next inference early so visuals stay aligned with the audio on slower machines.
- Exposed the calculated lookahead in the orchestrator stats/UI plumbing and covered the new scheduler helpers with Jest tests (including the early-trigger path).
- Ran `npm test` to cover the updated orchestrator suite.

## 2025-11-13
- Reworked the synthetic dataset mixer so each requested correlation blends the desired signal with an uncorrelated component, letting the training loop match the exact strength (and dial it back when necessary) instead of always maximizing.
- Updated the training utility tests to assert that generated datasets and achievement calculations now land near the requested strengths, and ran `npm test` to cover the changes.

## 2025-11-13
- Synced every model snapshot under `assets/models/` so the `metadata.label` now matches the filename for easier identification a
nd tooling alignment.
- Ran `npm test` to ensure the dataset label updates don't impact the code.
