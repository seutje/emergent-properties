import { AudioManager, AudioManagerEvents } from './audio/AudioManager.js';
import { FeatureExtractor, FEATURE_KEYS } from './audio/FeatureExtractor.js';
import { Renderer } from './render/Renderer.js';
import { MLPModel } from './ml/MLPModel.js';
import { MLPOrchestrator } from './ml/MLPOrchestrator.js';
import { MLPTrainingManager } from './ml/MLPTrainingManager.js';
import { UIController } from './ui/UIController.js';

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

bootstrap().catch((error) => {
  console.error('[main] Failed to bootstrap application', error);
});

async function bootstrap() {
  const appRoot = document.getElementById('app');
  const fileInput = document.getElementById('file-input');
  if (!appRoot || !fileInput) {
    throw new Error('Required root elements (#app, #file-input) are missing from the DOM.');
  }

  const renderer = new Renderer(appRoot);
  renderer.init();
  const particleField = renderer.getParticleField();

  const audioManager = new AudioManager({
    tracks: BUNDLED_TRACKS,
    fileInput,
    dropTarget: document.body,
  });
  audioManager.init();

  const featureExtractor = new FeatureExtractor(audioManager.getAnalyser());
  featureExtractor.init();

  const mlpModel = new MLPModel({
    inputSize: 17,
    outputSize: 9,
    hiddenLayers: [32],
    backend: 'webgl',
    activation: 'relu',
  });
  await mlpModel.init();

  const mlpController = new MLPOrchestrator({
    model: mlpModel,
    particleField,
    featureExtractor,
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
  });
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

  const dropOverlay = createDropOverlay();
  document.body.appendChild(dropOverlay);

  const transport = createTransportControls(audioManager);
  document.body.appendChild(transport);

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

function createTransportControls(manager) {
  const wrapper = document.createElement('section');
  wrapper.className = 'audio-transport';
  wrapper.setAttribute('aria-label', 'Audio transport controls');

  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.className = 'audio-btn';
  playButton.textContent = 'Play';

  const stopButton = document.createElement('button');
  stopButton.type = 'button';
  stopButton.className = 'audio-btn';
  stopButton.textContent = 'Stop';

  const trackSelect = document.createElement('select');
  trackSelect.className = 'audio-select';
  trackSelect.setAttribute('aria-label', 'Bundled tracks');

  const uploadButton = document.createElement('button');
  uploadButton.type = 'button';
  uploadButton.className = 'audio-btn';
  uploadButton.textContent = 'Upload';

  const status = document.createElement('span');
  status.className = 'audio-status';
  status.textContent = 'Pick a track to begin';

  const tracks = manager.getTracks();
  tracks.forEach((track, index) => {
    const option = document.createElement('option');
    option.value = track.id;
    option.textContent = `${index + 1}. ${track.title}`;
    trackSelect.append(option);
  });
  if (tracks[0]) {
    trackSelect.value = tracks[0].id;
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

  manager.on(AudioManagerEvents.STATE, (state) => {
    playButton.textContent = state.playing ? 'Pause' : 'Play';
    playButton.disabled = state.isLoading;
    stopButton.disabled = !state.playing && !state.currentTrack;
    uploadButton.disabled = state.isLoading;
    wrapper.classList.toggle('is-lockable', !state.unlocked);
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
  });

  manager.on(AudioManagerEvents.ERROR, ({ error }) => {
    status.textContent = error?.message || 'Audio error';
    wrapper.classList.add('has-error');
    setTimeout(() => wrapper.classList.remove('has-error'), 2000);
  });

  manager.on(AudioManagerEvents.TRACK_LOADED, ({ track }) => {
    status.textContent = `Now playing: ${track.title}`;
  });

  wrapper.append(playButton, stopButton, trackSelect, uploadButton, status);
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
