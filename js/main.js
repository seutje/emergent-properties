import { AudioManager, AudioManagerEvents } from './audio/AudioManager.js';
import { FeatureExtractor, FEATURE_KEYS } from './audio/FeatureExtractor.js';
import { Renderer } from './render/Renderer.js';
import { MLPModel } from './ml/MLPModel.js';
import { MLPOrchestrator } from './ml/MLPOrchestrator.js';
import { MLPTrainingManager } from './ml/MLPTrainingManager.js';
import {
  loadDefaultModelSnapshot,
  loadModelSnapshot,
  loadRandomSnapshotFromPool,
  MODEL_POOL_SNAPSHOT_URLS,
} from './ml/ModelSnapshotLoader.js';
import { UIController } from './ui/UIController.js';
import { PARTICLE_PARAMETER_COUNT } from './ml/MLPTrainingTargets.js';
import { upgradeModelSnapshot } from './ml/ModelSnapshotUpgrade.js';
import { PARTICLE_POSITIONAL_FEATURES } from './ml/MLPTrainingFeatures.js';
import { randomizeActiveModel } from './ml/randomizeActiveModel.js';
import { ModelCycler } from './ml/ModelCycler.js';
import { createTrackSnapshotMap, getTrackSnapshotUrl } from './ml/TrackModelResolver.js';
import { attachIdleVisibilityController } from './ui/IdleVisibilityController.js';

const BUNDLED_TRACKS = [
  { id: 'track-01', title: 'My Comrade', url: './assets/audio/01 - My Comrade.mp3' },
  { id: 'track-02', title: 'Dead Air', url: './assets/audio/02 - Dead Air.mp3' },
  { id: 'track-03', title: "Who's Counting", url: "./assets/audio/03 - Who's Counting.mp3" },
  { id: 'track-04', title: 'Acceptance Criteria', url: './assets/audio/04 - Acceptance Criteria.mp3' },
  { id: 'track-05', title: 'Digital Coherence', url: './assets/audio/05 - Digital Coherence.mp3' },
  { id: 'track-06', title: 'Quantized Being', url: './assets/audio/06 - Quantized Being.mp3' },
  { id: 'track-07', title: 'Super Position', url: './assets/audio/07 - Super Position.mp3' },
  { id: 'track-08', title: 'Duality', url: './assets/audio/08 - Duality.mp3' },
  { id: 'track-09', title: 'Negative Space', url: './assets/audio/09 - Negative Space.mp3' },
  { id: 'track-10', title: 'Recursive Dreams', url: './assets/audio/10 - Recursive Dreams.mp3' },
];

const TRACK_SNAPSHOT_MAP = createTrackSnapshotMap(BUNDLED_TRACKS, MODEL_POOL_SNAPSHOT_URLS);

const BASE_PARTICLE_DIMS = 8;

const FALLBACK_MLP_CONFIG = {
  inputSize: BASE_PARTICLE_DIMS + FEATURE_KEYS.length,
  outputSize: PARTICLE_PARAMETER_COUNT,
  hiddenLayers: [32],
  backend: 'webgl',
  activation: 'relu',
  outputActivation: 'tanh',
  seed: 1337,
};

const DEFAULT_VOLUME_PERCENT = 70;

const clampPercent = (value) => {
  if (!Number.isFinite(value)) return DEFAULT_VOLUME_PERCENT;
  if (value <= 0) return 0;
  if (value >= 100) return 100;
  return Math.round(value);
};

const percentToVolume = (percent) => clampPercent(percent) / 100;
const volumeToPercent = (value) => {
  if (!Number.isFinite(value)) {
    return DEFAULT_VOLUME_PERCENT;
  }
  const clamped = Math.min(1, Math.max(0, value));
  return Math.round(clamped * 100);
};

bootstrap().catch((error) => {
  console.error('[main] Failed to bootstrap application', error);
});

async function bootstrap() {
  const appRoot = document.getElementById('app');
  const fileInput = document.getElementById('file-input');
  if (!appRoot || !fileInput) {
    throw new Error('Required root elements (#app, #file-input) are missing from the DOM.');
  }

  const defaultSnapshot = await loadInitialModelSnapshot();

  const renderer = new Renderer(appRoot);
  renderer.init();
  const particleField = renderer.getParticleField();
  const applyTrackLayout = (trackMeta) => {
    if (trackMeta) {
      particleField?.setTrackLabel(trackMeta.title || trackMeta.id || '');
    }
  };

  const audioManager = new AudioManager({
    tracks: BUNDLED_TRACKS,
    fileInput,
    dropTarget: document.body,
    initialVolume: percentToVolume(DEFAULT_VOLUME_PERCENT),
  });
  audioManager.init();

  const featureExtractor = new FeatureExtractor(audioManager.getAnalyser());
  featureExtractor.init();

  const mlpModel = new MLPModel(defaultSnapshot?.config || FALLBACK_MLP_CONFIG);
  await mlpModel.init();

  const mlpController = new MLPOrchestrator({
    model: mlpModel,
    particleField,
    featureExtractor,
    renderer,
    options: {
      rateHz: 24,
      blend: 1,
    },
  });
  await mlpController.init();

  const trainingManager = new MLPTrainingManager({
    model: mlpModel,
    featureKeys: FEATURE_KEYS,
    audioDims: FEATURE_KEYS.length,
    baseDims: mlpController.baseDims,
    getBaseSamples: (count = 512) => mlpController.getBaseSamples(count),
    positionalFeatures: PARTICLE_POSITIONAL_FEATURES,
  });
  configureTrainingManagerFromSnapshot(trainingManager, defaultSnapshot);
  trainingManager.init();
  trainingManager.on('result', async (payload) => {
    if (payload?.weights) {
      try {
        await mlpModel.applyWeights(payload.weights);
        await mlpController.runOnce();
      } catch (error) {
        console.error('[main] Failed to apply trained weights', error);
      }
    }
  });

  const liveStats = { fps: 0, inferenceMs: 0 };
  const uiController = new UIController({
    renderer,
    particleField,
    audioManager,
    featureExtractor,
    mlpModel,
    mlpController,
    stats: liveStats,
    trainingManager,
    mlpTrainingOptions: trainingManager.getTrainingOptions(),
  });
  uiController.init();

  const modelCycler = new ModelCycler(MODEL_POOL_SNAPSHOT_URLS);
  const AUTO_RANDOMIZE_REASONS = new Set(['startup', 'track', 'upload']);

  const applyRandomization = async (payload = {}) => {
    const result = await randomizeActiveModel({
      mlpModel,
      mlpController,
      trainingManager,
      uiController,
      ...payload,
    });
    if (payload.snapshotUrl) {
      modelCycler.syncTo(payload.snapshotUrl);
    }
    return result;
  };

  const randomizeModel = async (details = {}) => {
    const reason = details?.reason || 'manual';
    if (AUTO_RANDOMIZE_REASONS.has(reason)) {
      try {
        const { snapshot, url } = await loadRandomSnapshotFromPool();
        return await applyRandomization({
          ...details,
          snapshot,
          snapshotUrl: url,
        });
      } catch (error) {
        console.warn('[main] Failed to load curated model snapshot, falling back to random seed.', error);
      }
    }

    try {
      return await applyRandomization(details);
    } catch (error) {
      console.error('[main] Failed to randomize model', error);
      return null;
    }
  };

  const loadModelForBundledTrack = async (track) => {
    const snapshotUrl = getTrackSnapshotUrl(track, TRACK_SNAPSHOT_MAP);
    if (!snapshotUrl) {
      console.warn('[main] No curated model mapping found for track; falling back to random seed.', track);
      return randomizeModel({ reason: 'track', track });
    }
    try {
      const snapshot = await loadModelSnapshot(snapshotUrl);
      return await applyRandomization({
        reason: 'track',
        track,
        snapshot,
        snapshotUrl,
      });
    } catch (error) {
      console.error('[main] Failed to load curated snapshot for track. Falling back to random model.', error);
      return randomizeModel({ reason: 'track', track });
    }
  };

  const loadNextCuratedModel = async () => {
    if (!modelCycler.hasNext()) {
      throw new Error('No curated models are available.');
    }
    const url = modelCycler.peek();
    if (!url) {
      throw new Error('No curated models are available.');
    }
    const snapshot = await loadModelSnapshot(url);
    await applyRandomization({
      reason: 'next-model',
      snapshot,
      snapshotUrl: url,
    });
  };

  await randomizeModel({ reason: 'startup' });

  const dropOverlay = createDropOverlay();
  document.body.appendChild(dropOverlay);

  const transport = createTransportControls(audioManager, DEFAULT_VOLUME_PERCENT, {
    onNextModel: loadNextCuratedModel,
  });
  document.body.appendChild(transport);
  attachIdleVisibilityController(transport, {
    idleMs: 3000,
    hiddenClass: 'audio-transport--hidden',
  });

  const gate = createAudioGate(audioManager);
  document.body.appendChild(gate.element);

  const onboarding = createOnboardingOverlay();
  if (onboarding) {
    document.body.appendChild(onboarding.element);
  }

  audioManager.on(AudioManagerEvents.STATE, (state) => {
    gate.setVisible(!state.unlocked);
    if (state.playing || state.unlocked) {
      onboarding?.complete?.();
    }
  });

  audioManager.on(AudioManagerEvents.TRACK_LOADED, ({ track, isResume }) => {
    applyTrackLayout(track);
    if (isResume) {
      return;
    }
    if (track?.source === 'upload') {
      randomizeModel({ reason: 'upload', track });
      return;
    }
    loadModelForBundledTrack(track);
  });

  let last = performance.now();
  function loop(timestamp) {
    const delta = (timestamp - last) * 0.001;
    last = timestamp;
    if (delta > 0) {
      const instant = 1 / delta;
      liveStats.fps = Number.isFinite(liveStats.fps)
        ? liveStats.fps * 0.9 + instant * 0.1
        : instant;
      liveStats.fps = Math.round(liveStats.fps);
    }
    featureExtractor.sample(delta);
    mlpController.update(delta);
    const stats = mlpController.getStats();
    const inference = stats?.lastInferenceMs;
    liveStats.inferenceMs = Number.isFinite(inference) ? Math.round(inference * 100) / 100 : 0;
    renderer.update(delta);
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

async function loadInitialModelSnapshot() {
  try {
    const snapshot = await loadDefaultModelSnapshot();
    return upgradeModelSnapshot(snapshot);
  } catch (error) {
    console.warn('[main] Unable to load default model snapshot.', error);
    return null;
  }
}

function configureTrainingManagerFromSnapshot(trainingManager, snapshot) {
  if (!trainingManager || !snapshot) {
    return;
  }
  const correlations = snapshot.metadata?.correlations;
  if (Array.isArray(correlations) && correlations.length) {
    trainingManager.setCorrelations(correlations);
  }
  const trainingOptions = extractTrainingOptions(snapshot);
  if (trainingOptions) {
    trainingManager.updateTrainingOptions(trainingOptions);
  }
}

function extractTrainingOptions(snapshot) {
  const source = snapshot?.metadata?.training?.metadata || snapshot?.metadata?.training;
  if (!source) {
    return null;
  }
  const map = {
    epochs: 'epochs',
    batchSize: 'batchSize',
    learningRate: 'learningRate',
    sampleCount: 'sampleCount',
    noise: 'noise',
    seed: 'seed',
  };
  const options = {};
  let updated = false;
  Object.entries(map).forEach(([optionKey, sourceKey]) => {
    const value = source[sourceKey];
    if (Number.isFinite(value)) {
      options[optionKey] = value;
      updated = true;
    }
  });
  return updated ? options : null;
}

function createAudioGate(manager) {
  const gate = document.createElement('div');
  gate.className = 'audio-gate';

  const text = document.createElement('p');
  text.textContent = 'Click to enable audio playback.';

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'audio-gate__button';
  button.textContent = 'Start Audio';

  gate.append(text, button);

  const handleClick = async () => {
    try {
      gate.classList.add('is-loading');
      await manager.play();
    } catch (error) {
      console.error('[AudioGate] Failed to start audio', error);
    } finally {
      gate.classList.remove('is-loading');
    }
  };

  button.addEventListener('click', handleClick);

  let removed = false;
  const destroy = () => {
    if (removed) return;
    removed = true;
    button.removeEventListener('click', handleClick);
    gate.remove();
  };

  return {
    element: gate,
    setVisible(visible) {
      if (!visible) {
        destroy();
      } else if (!removed) {
        gate.removeAttribute('hidden');
      }
    },
  };
}

function createTransportControls(manager, defaultVolumePercent = DEFAULT_VOLUME_PERCENT, options = {}) {
  const wrapper = document.createElement('section');
  wrapper.className = 'audio-transport';
  wrapper.setAttribute('aria-label', 'Audio transport controls');
  const onNextModel = typeof options.onNextModel === 'function' ? options.onNextModel : null;
  const primaryRow = document.createElement('div');
  primaryRow.className = 'audio-transport__row audio-transport__row--primary';
  const sliderRow = document.createElement('div');
  sliderRow.className = 'audio-transport__slider-row';
  const metaRow = document.createElement('div');
  metaRow.className = 'audio-transport__row audio-transport__row--meta';

  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.className = 'audio-btn';
  playButton.textContent = 'Play';

  const stopButton = document.createElement('button');
  stopButton.type = 'button';
  stopButton.className = 'audio-btn';
  stopButton.textContent = 'Stop';

  const nextTrackButton = document.createElement('button');
  nextTrackButton.type = 'button';
  nextTrackButton.className = 'audio-btn';
  nextTrackButton.setAttribute('aria-label', 'Skip to next track');
  nextTrackButton.textContent = 'Next';

  const trackSelect = document.createElement('select');
  trackSelect.className = 'audio-select';
  trackSelect.setAttribute('aria-label', 'Track selector');
  const trackSelectWrapper = document.createElement('div');
  trackSelectWrapper.className = 'audio-select-wrapper';
  const trackSelectArrow = document.createElement('span');
  trackSelectArrow.className = 'audio-select__arrow';
  trackSelectArrow.setAttribute('aria-hidden', 'true');
  trackSelectWrapper.append(trackSelect, trackSelectArrow);

  const albumGroup = document.createElement('optgroup');
  albumGroup.label = 'Album tracks';
  trackSelect.append(albumGroup);

  const customGroup = document.createElement('optgroup');
  customGroup.label = 'Custom uploads';
  customGroup.hidden = true;
  trackSelect.append(customGroup);

  const uploadButton = document.createElement('button');
  uploadButton.type = 'button';
  uploadButton.className = 'audio-btn';
  uploadButton.textContent = 'Upload';

  const status = document.createElement('span');
  status.className = 'audio-status';
  status.textContent = 'Pick a track to begin';

  const volumeWrapper = document.createElement('div');
  volumeWrapper.className = 'audio-volume';

  const volumeLabel = document.createElement('span');
  volumeLabel.className = 'audio-volume__label';
  volumeLabel.textContent = 'Vol';

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.className = 'audio-volume__slider';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.step = '1';
  volumeSlider.setAttribute('aria-label', 'Volume');
  const initialVolumePercent = volumeToPercent(manager.getState().volume ?? percentToVolume(defaultVolumePercent));
  volumeSlider.value = String(initialVolumePercent);

  const volumeValue = document.createElement('span');
  volumeValue.className = 'audio-volume__value';
  volumeValue.textContent = `${initialVolumePercent}%`;

  const updateVolumeUi = (percent) => {
    const safePercent = clampPercent(percent);
    const nextValue = String(safePercent);
    if (volumeSlider.value !== nextValue) {
      volumeSlider.value = nextValue;
    }
    volumeValue.textContent = `${safePercent}%`;
  };

  volumeSlider.addEventListener('input', (event) => {
    const percent = Number(event.target.value);
    manager.setVolume(percentToVolume(percent));
    updateVolumeUi(percent);
  });

  volumeWrapper.append(volumeLabel, volumeSlider, volumeValue);

  const repeatLabel = document.createElement('label');
  repeatLabel.className = 'audio-repeat';
  repeatLabel.setAttribute('title', 'Repeat current track');

  const repeatToggle = document.createElement('input');
  repeatToggle.type = 'checkbox';
  repeatToggle.className = 'audio-repeat__input';
  repeatToggle.setAttribute('aria-label', 'Repeat current track');
  repeatToggle.checked = Boolean(manager.getState().repeat);

  const repeatText = document.createElement('span');
  repeatText.className = 'audio-repeat__text';
  repeatText.textContent = 'Repeat';

  repeatLabel.append(repeatToggle, repeatText);

  const nextModelButton = document.createElement('button');
  nextModelButton.type = 'button';
  nextModelButton.className = 'audio-btn audio-btn--ghost';
  nextModelButton.setAttribute('aria-label', 'Load next curated model');
  nextModelButton.textContent = 'Next model';

  const tracks = manager.getTracks();
  const customOptions = new Map();

  const ensureCustomGroupVisibility = () => {
    customGroup.hidden = customGroup.children.length === 0;
  };

  const createOption = (track, prefix = '') => {
    const option = document.createElement('option');
    option.value = track.id;
    option.textContent = `${prefix}${track.title}`;
    option.dataset.source = track.source || 'bundled';
    return option;
  };

  const addCustomTrackOption = (track) => {
    if (!track?.id) return;
    const existing = customOptions.get(track.id);
    if (existing) {
      existing.textContent = track.title;
    } else {
      const option = createOption(track);
      customGroup.append(option);
      customOptions.set(track.id, option);
    }
    ensureCustomGroupVisibility();
  };

  tracks.forEach((track, index) => {
    albumGroup.append(createOption(track, `${index + 1}. `));
  });

  const uploadedTracks =
    typeof manager.getUploadedTracks === 'function' ? manager.getUploadedTracks() : [];
  uploadedTracks.forEach((track) => addCustomTrackOption(track));
  ensureCustomGroupVisibility();

  if (tracks[0]) {
    trackSelect.value = tracks[0].id;
  } else if (uploadedTracks[0]) {
    trackSelect.value = uploadedTracks[0].id;
  }

  playButton.addEventListener('click', () => {
    const state = manager.getState();
    if (state.playing) {
      manager.pause();
    } else {
      manager.play();
    }
  });

  stopButton.addEventListener('click', () => manager.stop());
  uploadButton.addEventListener('click', () => manager.triggerFilePicker());
  trackSelect.addEventListener('change', (event) => manager.playTrack(event.target.value));
  repeatToggle.addEventListener('change', (event) => manager.setRepeat(event.target.checked));
  const updateNextTrackButtonState = (state = manager.getState()) => {
    const baseCount = manager.getTracks().length;
    const uploadCount =
      typeof manager.getUploadedTracks === 'function' ? manager.getUploadedTracks().length : 0;
    const hasChoice = baseCount + uploadCount > 1;
    nextTrackButton.disabled = !hasChoice || state.isLoading;
  };

  nextTrackButton.addEventListener('click', async () => {
    if (nextTrackButton.disabled) {
      return;
    }
    nextTrackButton.disabled = true;
    try {
      await manager.playNextTrack();
    } catch (error) {
      console.error('[Transport] Failed to skip track', error);
      status.textContent = error?.message || 'Unable to skip track';
      wrapper.classList.add('has-error');
      setTimeout(() => wrapper.classList.remove('has-error'), 2000);
    } finally {
      updateNextTrackButtonState();
    }
  });

  updateNextTrackButtonState();
  if (onNextModel) {
    nextModelButton.addEventListener('click', async () => {
      if (nextModelButton.disabled) {
        return;
      }
      const previousLabel = nextModelButton.textContent;
      nextModelButton.disabled = true;
      nextModelButton.textContent = 'Loading...';
      try {
        await onNextModel();
      } catch (error) {
        console.error('[Transport] Failed to load next model', error);
        status.textContent = error?.message || 'Failed to load model';
        wrapper.classList.add('has-error');
        setTimeout(() => wrapper.classList.remove('has-error'), 2000);
      } finally {
        nextModelButton.textContent = previousLabel;
        nextModelButton.disabled = false;
      }
    });
  } else {
    nextModelButton.disabled = true;
  }

  manager.on(AudioManagerEvents.STATE, (state) => {
    playButton.textContent = state.playing ? 'Pause' : 'Play';
    playButton.disabled = state.isLoading;
    stopButton.disabled = !state.playing && !state.currentTrack;
    updateNextTrackButtonState(state);
    uploadButton.disabled = state.isLoading;
    wrapper.classList.toggle('is-lockable', !state.unlocked);
    if (state.currentTrack?.id && !customOptions.has(state.currentTrack.id)) {
      if (state.currentTrack.source === 'upload') {
        addCustomTrackOption(state.currentTrack);
      }
    }
    if (state.currentTrack?.id) {
      trackSelect.value = state.currentTrack.id;
    }
    if (state.isLoading) {
      status.textContent = 'Loading track...';
    } else if (state.currentTrack) {
      status.textContent = `Now playing: ${state.currentTrack.title}`;
    } else {
      status.textContent = 'Pick a track to begin';
    }
    if (typeof state.volume === 'number') {
      updateVolumeUi(volumeToPercent(state.volume));
    } else {
      updateVolumeUi(defaultVolumePercent);
    }
    const repeatEnabled = Boolean(state.repeat);
    if (repeatToggle.checked !== repeatEnabled) {
      repeatToggle.checked = repeatEnabled;
    }
  });

  manager.on(AudioManagerEvents.ERROR, ({ error }) => {
    status.textContent = error?.message || 'Audio error';
    wrapper.classList.add('has-error');
    setTimeout(() => wrapper.classList.remove('has-error'), 2000);
  });

  manager.on(AudioManagerEvents.TRACK_LOADED, ({ track }) => {
    status.textContent = `Now playing: ${track.title}`;
  });

  manager.on(AudioManagerEvents.UPLOAD, ({ track }) => {
    addCustomTrackOption(track);
    trackSelect.value = track.id;
    updateNextTrackButtonState();
  });

  primaryRow.append(playButton, stopButton, nextTrackButton, trackSelectWrapper);
  sliderRow.append(volumeWrapper);
  metaRow.append(status, repeatLabel, uploadButton, nextModelButton);
  wrapper.append(primaryRow, sliderRow, metaRow);
  return wrapper;
}

function createDropOverlay() {
  const overlay = document.createElement('div');
  overlay.className = 'audio-drop-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  overlay.innerHTML = '<p>Drop audio files to play them instantly</p>';
  return overlay;
}

function createOnboardingOverlay(storageKey = 'ep-onboarding-dismissed') {
  if (typeof document === 'undefined') {
    return null;
  }
  const storage = typeof window !== 'undefined' ? window.localStorage : null;
  const dismissed = storage?.getItem(storageKey) === '1';
  if (dismissed) {
    return null;
  }
  const wrapper = document.createElement('aside');
  wrapper.className = 'onboarding-tip';
  wrapper.setAttribute('role', 'status');
  const heading = document.createElement('h3');
  heading.textContent = 'Welcome to Emergent Properties';
  const intro = document.createElement('p');
  intro.textContent = 'Drop a track or pick a bundled song, then use the presets panel (right) to sculpt the field.';
  const list = document.createElement('ul');
  [
    'Drag & drop audio anywhere on the screen or use the Upload button.',
    'Open the “Presets” folder in the UI to swap between curated vibes.',
    'Tweak audio, render, and model folders to dial in your own look.',
  ].forEach((tip) => {
    const li = document.createElement('li');
    li.textContent = tip;
    list.append(li);
  });
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'onboarding-tip__button';
  button.textContent = "Let's play";

  const dismiss = () => {
    wrapper.classList.add('is-hidden');
    storage?.setItem(storageKey, '1');
    setTimeout(() => wrapper.remove(), 400);
  };

  button.addEventListener('click', dismiss);
  wrapper.append(heading, intro, list, button);

  return {
    element: wrapper,
    complete: dismiss,
  };
}
