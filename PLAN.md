# Emergent Properties Build Plan

This plan assumes a human collaborator and one or more AI coding agents working in a "vibe-coding" loop: the human curates intent and aesthetics, while the AI handles most implementation with frequent checkpoints.

## Phase 0 — Alignment & Environment
- [x] [Human] Kickoff vibe-sync between human + AI about audio-visual goals, creative references, and scope guardrails.
- [x] [Human] Confirm repo structure, toolchain (static hosting, no build), and access to DESIGN.md & assets.
- [x] [AI] Scaffold `index.html`, `styles/`, and `/js` folders per DESIGN.md tree.
- [x] [Human] Validate environment by launching a static file server and seeing the blank app shell load.
- [x] [AI] Add `package.json` tooling (Jest + `http-server` start script) for local dev/test loops.

**Acceptance criteria:**
- Human feels the creative direction is locked.
- Repository mirrors the directory skeleton in DESIGN.md.
- Static server renders a blank canvas without console errors.

## Phase 1 — Core Rendering Loop
- [x] [AI] Implement `Renderer`, `BaseModule`, and main render loop wiring with placeholder particle cloud.
- [x] [Human] Inspect rendering performance (FPS overlay or browser stats) and confirm camera controls feel smooth.
- [x] [AI] Add responsive resizing + OrbitControls tuning per DESIGN.md.

**Acceptance criteria:**
- Render loop hits 60 FPS on an empty scene.
- Camera orbit/pan/zoom works on desktop + trackpad (human verified).
- Code follows ES module structure with one `main.js` entry point.

## Phase 2 — Audio Pipeline
- [x] [AI] Build `AudioManager` to load bundled MP3s, manage play/pause, and expose analyser data.
- [x] [Human] Drop in test MP3s, confirm playback works, and approve UX for track switching (manual QA).
- [x] [AI] Implement drag & drop + file picker for user uploads with gating for autoplay policies.
- [x] [Human] Verify uploads from local disk play and stop cleanly; log any browser-specific blockers.

**Acceptance criteria:**
- Bundled tracks and user uploads both reach the analyser node without runtime errors.
- Human can start/stop audio after a user gesture in Chrome + Firefox.
- AudioManager emits events/signals other modules can subscribe to.

## Phase 3 — Feature Extraction & Smoothing
- [x] [AI] Implement `FeatureExtractor` computing RMS, spectral centroid, rolloff, band energies, and tempo proxy as per DESIGN.md §6.
- [x] [Human] Listen/observe lil-gui debug readouts to ensure features react plausibly to bass/treble-heavy tracks.
- [x] [AI] Add smoothing/EMA + decimation toggles exposed through lil-gui.

**Acceptance criteria:**
- Feature vectors update at the configured sample rate without blocking the render loop.
- GUI displays live feature values and allows human-tuned smoothing parameters.
- Human signs off that feature trends match the audio intuitively.

## Phase 4 — Particle Field & Attributes
- [x] [AI] Create `ParticleField` with instanced geometry (≥30k particles), seeded variation, and GPU buffers for attributes described in DESIGN.md §5.
- [x] [Human] Confirm baseline color palette + spacing match vibe references; request tweaks if needed.
- [x] [AI] Wire attribute update hooks so external systems (MLP outputs) can mutate position/color/flicker.

**Acceptance criteria:**
- Scene renders ≥30k particles at ≥45 FPS on reference hardware.
- Particle attributes (idHash, baseColor, phase) are deterministic when seed is constant.
- Human approves the default visual composition before ML modulation.

## Phase 5 — MLP Integration
- [x] [AI] Implement `MLPModel` with TensorFlow.js sequential architecture (input→hidden→hidden→output) matching DESIGN.md specs.
- [x] [Human] Review lil-gui-exposed hyperparameters (layer sizes, activation, output clamps) for usability.
- [x] [AI] Batch features + particle state into tf.js tensors, run inference at decimated rate, and feed deltas to `ParticleField`.
- [x] [Human] Stress test varying particle counts + model sizes; record FPS + inference latency for tuning.

**Acceptance criteria:**
- Inference layer updates particle transforms without frame hitches.
- GUI exposes all relevant model parameters with safe ranges.
- Human validates that different audio tracks produce distinct, stable particle responses.

## Phase 6 — UI & Presets
- [x] [AI] Implement `UIController` (lil-gui) grouping render/audio/model settings, plus preset save/load (localStorage).
- [x] [Human] Curate at least three vibe presets (e.g., Chill Bloom, Pulsar Storm, Minimal Drift) and verify they persist across reloads.
- [x] [AI] Add lightweight onboarding text or overlay guiding users to drop tracks and tweak presets.

**Acceptance criteria:**
- All adjustable parameters in DESIGN.md are surfaced in the GUI.
- Presets serialize+hydrate accurately; human can toggle and see immediate effect.
- First-time user instructions appear and dismiss gracefully.

## Phase 7 — Polish & QA
- [ ] [Human] Run through cross-browser smoke tests (Chrome, Firefox, Safari if possible) and log quirks.
- [x] [AI] Add adaptive reactivity envelope plus GUI controls to amplify particle response to audio dynamics.
- [x] [AI] Harden the MLP training loop with temporal/input smoothness regularizers so inference outputs stay stable under jittery audio.
- [ ] [AI] Optimize bottlenecks discovered (particle budget, shader tweaks, throttled inference) and fix logged issues.
- [ ] [Human] Approve final visual storytelling and audio-reactive feel.
- [ ] [AI] Prepare release checklist (README updates, asset credits, deploy instructions).
- [x] [AI] Insert a post-MLP response controller so renderer globals get slew-limited, hysteretic smoothing tied to audio energy.
- [x] [AI] Add in-browser training workflow with user-defined correlations plus model import/export tooling.
- [x] [AI] Load curated default MLP snapshot (assets/models/default.json) on boot to sync baseline model + training settings.
- [x] [AI] Disable EMA smoothing by default so first-time loads reflect raw audio dynamics; GUI still exposes the toggle.
- [x] [AI] Add a perceived-linear volume slider to the transport UI with a 70% default setting.
- [x] [AI] Add a Repeat toggle to the transport controls so listeners can loop the current track when desired.
- [x] [AI] Add a Random Model control to the training panel so users can reseed weights without running training.
- [x] [AI] Add a Finetune training control that continues new runs from the current active model weights.
- [x] [AI] Automatically reseed the active model at startup and whenever a new track begins playback.
- [x] [AI] Delay new track playback by ~1 second so curated/loaded models have time to hydrate before audio starts piping through.
- [x] [AI] Form the default particle layout into the active track title so the field spells the song name.
- [x] [AI] Add a Next Model transport control to cycle through curated MLP snapshots sequentially.
- [x] [AI] Auto-hide the transport bar after a few seconds of pointer inactivity while keeping it visible during hover.

**Acceptance criteria:**
- App meets DESIGN.md §19 acceptance criteria.
- Known issues list is empty or explicitly waived by the human.
- README documents setup, controls, and deployment steps for static hosting.
