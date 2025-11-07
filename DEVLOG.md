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
