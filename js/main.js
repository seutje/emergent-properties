import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';
import { AudioManager, AudioManagerEvents } from './audio/AudioManager.js';
import { FeatureExtractor, FEATURE_KEYS } from './audio/FeatureExtractor.js';
import { Renderer } from './render/Renderer.js';
import { MLPModel } from './ml/MLPModel.js';
import { MLPOrchestrator } from './ml/MLPOrchestrator.js';

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
    backend: 'auto',
  });
  await mlpModel.init();

  const mlpController = new MLPOrchestrator({
    model: mlpModel,
    particleField,
    featureExtractor,
    options: {
      rateHz: 24,
      blend: 0.85,
    },
  });
  await mlpController.init();

  const gui = new GUI();
  gui.title('Emergent Properties');
  const fpsStats = { fps: 0 };
  createFeatureGui(gui, featureExtractor, fpsStats);
  createMlpGui(gui, mlpModel, mlpController);

  const dropOverlay = createDropOverlay();
  document.body.appendChild(dropOverlay);

  const transport = createTransportControls(audioManager);
  document.body.appendChild(transport);

  const gate = createAudioGate(audioManager);
  document.body.appendChild(gate.element);

  audioManager.on(AudioManagerEvents.STATE, (state) => {
    gate.setVisible(!state.unlocked);
  });

  let last = performance.now();
  function loop(timestamp) {
    const delta = (timestamp - last) * 0.001;
    last = timestamp;
    if (delta > 0) {
      const instant = 1 / delta;
      fpsStats.fps = Number.isFinite(fpsStats.fps)
        ? fpsStats.fps * 0.9 + instant * 0.1
        : instant;
      fpsStats.fps = Math.round(fpsStats.fps);
    }
    featureExtractor.sample(delta);
    mlpController.update(delta);
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

function createFeatureGui(guiInstance, extractor, liveStats) {
  const folder = guiInstance.addFolder('Audio Features');
  const settings = {
    sampleRate: extractor.options.sampleRate,
    decimate: extractor.options.decimation.enabled,
    smoothing: extractor.options.smoothing.enabled,
    smoothingAlpha: extractor.options.smoothing.alpha,
  };

  folder
    .add(settings, 'sampleRate', 5, 120, 1)
    .name('Sample rate (Hz)')
    .onChange((value) => extractor.setSampleRate(value));

  folder
    .add(settings, 'decimate')
    .name('Decimation enabled')
    .onChange((value) => extractor.setDecimationEnabled(value));

  folder
    .add(settings, 'smoothing')
    .name('EMA enabled')
    .onChange((value) => extractor.setSmoothingEnabled(value));

  folder
    .add(settings, 'smoothingAlpha', 0.01, 1, 0.01)
    .name('EMA alpha')
    .onChange((value) => extractor.setSmoothingAlpha(value));

  const liveFolder = folder.addFolder('Live Values');
  const labels = {
    rms: 'RMS',
    specCentroid: 'Spectral centroid',
    specRolloff: 'Spectral rolloff',
    bandLow: 'Low band',
    bandMid: 'Mid band',
    bandHigh: 'High band',
    peak: 'Peak',
    zeroCrossRate: 'Zero-cross rate',
    tempoProxy: 'Tempo proxy',
  };

  FEATURE_KEYS.forEach((key) => {
    const controller = liveFolder.add(extractor.getFeatures(), key).name(labels[key] || key).listen();
    controller.disable?.();
  });

  if (liveStats && Object.prototype.hasOwnProperty.call(liveStats, 'fps')) {
    const fpsController = liveFolder.add(liveStats, 'fps').name('FPS').listen();
    fpsController.disable?.();
  }

  folder.open();
  liveFolder.open();
  return folder;
}

function createMlpGui(guiInstance, model, orchestrator) {
  const folder = guiInstance.addFolder('MLP Model');
  const config = model.getConfig();
  const outputs = orchestrator.getOutputConfig();
  const state = {
    backend: model.getBackend(),
    activation: config.activation,
    layers: config.hiddenLayers.length,
    hiddenUnits: config.hiddenLayers[0] ?? 32,
    rateHz: orchestrator.getRate(),
    blend: orchestrator.getBlend(),
    deltaPosX: outputs.deltaPos.x,
    deltaPosY: outputs.deltaPos.y,
    deltaPosZ: outputs.deltaPos.z,
    sizeMin: outputs.size.min,
    sizeMax: outputs.size.max,
    colorMin: outputs.color.min,
    colorMax: outputs.color.max,
    flickerRateMin: outputs.flickerRate.min,
    flickerRateMax: outputs.flickerRate.max,
    flickerDepthMin: outputs.flickerDepth.min,
    flickerDepthMax: outputs.flickerDepth.max,
  };

  const applyModelUpdate = async () => {
    const layers = new Array(state.layers).fill(state.hiddenUnits);
    try {
      await model.rebuild({
        hiddenLayers: layers,
        activation: state.activation,
      });
      await orchestrator.syncModelDimensions();
    } catch (error) {
      console.error('[MLP GUI] Failed to rebuild model', error);
    }
  };

  folder
    .add(state, 'backend', ['auto', 'webgl', 'wasm', 'cpu'])
    .name('Backend')
    .onChange(async (value) => {
      try {
        await model.setBackend(value);
      } catch (error) {
        console.error('[MLP GUI] Failed to set backend', error);
      }
    });

  folder
    .add(state, 'activation', ['tanh', 'relu', 'elu'])
    .name('Activation')
    .onChange((value) => {
      state.activation = value;
      applyModelUpdate();
    });

  folder
    .add(state, 'layers', 1, 3, 1)
    .name('Hidden layers')
    .onChange((value) => {
      state.layers = value;
      applyModelUpdate();
    });

  folder
    .add(state, 'hiddenUnits', 8, 96, 4)
    .name('Hidden units')
    .onChange((value) => {
      state.hiddenUnits = value;
      applyModelUpdate();
    });

  folder
    .add(state, 'rateHz', 5, 90, 1)
    .name('Inference rate (Hz)')
    .onChange((value) => orchestrator.setRate(value));

  folder
    .add(state, 'blend', 0, 1, 0.05)
    .name('Model blend')
    .onChange((value) => orchestrator.setBlend(value));

  const outputFolder = folder.addFolder('Output clamps');

  outputFolder
    .add(state, 'deltaPosX', 0.1, 2, 0.05)
    .name('ΔX range')
    .onChange((value) => orchestrator.updateOutput('deltaPos', { x: value }));

  outputFolder
    .add(state, 'deltaPosY', 0.1, 2, 0.05)
    .name('ΔY range')
    .onChange((value) => orchestrator.updateOutput('deltaPos', { y: value }));

  outputFolder
    .add(state, 'deltaPosZ', 0.1, 2, 0.05)
    .name('ΔZ range')
    .onChange((value) => orchestrator.updateOutput('deltaPos', { z: value }));

  outputFolder
    .add(state, 'sizeMin', -5, 0, 0.1)
    .name('Size min')
    .onChange((value) => {
      state.sizeMin = Math.min(value, state.sizeMax - 0.1);
      orchestrator.updateOutput('size', { min: state.sizeMin });
    });

  outputFolder
    .add(state, 'sizeMax', 0, 5, 0.1)
    .name('Size max')
    .onChange((value) => {
      state.sizeMax = Math.max(value, state.sizeMin + 0.1);
      orchestrator.updateOutput('size', { max: state.sizeMax });
    });

  outputFolder
    .add(state, 'colorMin', -1, 0, 0.01)
    .name('Color min')
    .onChange((value) => {
      state.colorMin = Math.min(value, state.colorMax - 0.01);
      orchestrator.updateOutput('color', { min: state.colorMin });
    });

  outputFolder
    .add(state, 'colorMax', 0, 1, 0.01)
    .name('Color max')
    .onChange((value) => {
      state.colorMax = Math.max(value, state.colorMin + 0.01);
      orchestrator.updateOutput('color', { max: state.colorMax });
    });

  outputFolder
    .add(state, 'flickerRateMin', 0, 5, 0.05)
    .name('Flicker rate min')
    .onChange((value) => {
      state.flickerRateMin = Math.min(value, state.flickerRateMax - 0.05);
      orchestrator.updateOutput('flickerRate', { min: state.flickerRateMin });
    });

  outputFolder
    .add(state, 'flickerRateMax', 0.1, 8, 0.05)
    .name('Flicker rate max')
    .onChange((value) => {
      state.flickerRateMax = Math.max(value, state.flickerRateMin + 0.05);
      orchestrator.updateOutput('flickerRate', { max: state.flickerRateMax });
    });

  outputFolder
    .add(state, 'flickerDepthMin', 0, 1, 0.01)
    .name('Flicker depth min')
    .onChange((value) => {
      state.flickerDepthMin = Math.min(value, state.flickerDepthMax - 0.01);
      orchestrator.updateOutput('flickerDepth', { min: state.flickerDepthMin });
    });

  outputFolder
    .add(state, 'flickerDepthMax', 0, 1, 0.01)
    .name('Flicker depth max')
    .onChange((value) => {
      state.flickerDepthMax = Math.max(value, state.flickerDepthMin + 0.01);
      orchestrator.updateOutput('flickerDepth', { max: state.flickerDepthMax });
    });

  folder.open();
  outputFolder.open();
  return folder;
}
