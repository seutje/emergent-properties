import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';
import { BaseModule } from '../core/BaseModule.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PresetStore } from './PresetStore.js';
import { TrainingPanel } from './TrainingPanel.js';
import { PARTICLE_POSITIONAL_FEATURES } from '../ml/MLPTrainingFeatures.js';

const FEATURE_LABELS = {
  rms: 'RMS',
  specCentroid: 'Spectral centroid',
  specRolloff: 'Spectral rolloff',
  bandSub: 'Sub',
  bandBass: 'Bass',
  bandLowMid: 'Low-mid',
  bandMid: 'Mid band',
  bandHigh: 'High band',
  peak: 'Peak',
  zeroCrossRate: 'Zero-cross rate',
  tempoProxy: 'Tempo proxy',
};

const BUILTIN_PRESETS = [
  {
    name: 'Chill Bloom',
    builtin: true,
    data: {
      features: { sampleRate: 24, smoothing: false, smoothingAlpha: 0.4, decimate: true },
      particles: {
        rotationSpeed: 0.04,
        wobbleStrength: 0.06,
        wobbleFrequency: 0.25,
        colorMix: 0.55,
        alphaScale: 0.8,
        pointScale: 1.1,
        seed: 311,
      },
      mlp: {
        activation: 'tanh',
        hiddenLayers: [28],
        rateHz: 18,
        blend: 0.82,
        backend: 'auto',
        seed: 1447,
      },
      outputs: {
        deltaPos: { x: 0.4, y: 0.55, z: 0.3 },
        size: { min: -1.5, max: 2 },
        color: { min: -0.2, max: 0.4 },
        flickerRate: { min: 0.2, max: 2.5 },
        flickerDepth: { min: 0.05, max: 0.4 },
        rotationSpeed: { min: 0.01, max: 0.18 },
        wobbleStrength: { min: 0, max: 0.2 },
        wobbleFrequency: { min: 0.05, max: 0.8 },
        colorMix: { min: 0.3, max: 1.2 },
        alphaScale: { min: 0.5, max: 1.4 },
        pointScale: { min: 0.8, max: 2 },
        cameraZoom: { min: 4.5, max: 9 },
      },
    },
  },
  {
    name: 'Pulsar Storm',
    builtin: true,
    data: {
      features: { sampleRate: 36, smoothing: false, smoothingAlpha: 0.2, decimate: false },
      particles: {
        rotationSpeed: 0.12,
        wobbleStrength: 0.16,
        wobbleFrequency: 0.65,
        colorMix: 0.85,
        alphaScale: 1.2,
        pointScale: 1.4,
        seed: 777,
      },
      mlp: {
        activation: 'relu',
        hiddenLayers: [36, 24],
        rateHz: 30,
        blend: 0.65,
        backend: 'auto',
        seed: 927,
      },
      outputs: {
        deltaPos: { x: 0.75, y: 0.9, z: 0.65 },
        size: { min: -2.5, max: 3.5 },
        color: { min: -0.35, max: 0.6 },
        flickerRate: { min: 0.35, max: 4.8 },
        flickerDepth: { min: 0.1, max: 0.65 },
        rotationSpeed: { min: 0.05, max: 0.3 },
        wobbleStrength: { min: 0, max: 0.3 },
        wobbleFrequency: { min: 0.1, max: 1.1 },
        colorMix: { min: 0.4, max: 1.4 },
        alphaScale: { min: 0.6, max: 1.6 },
        pointScale: { min: 1, max: 2.3 },
        cameraZoom: { min: 3.5, max: 11 },
      },
    },
  },
  {
    name: 'Minimal Drift',
    builtin: true,
    data: {
      features: { sampleRate: 18, smoothing: false, smoothingAlpha: 0.6, decimate: true },
      particles: {
        rotationSpeed: 0.015,
        wobbleStrength: 0.02,
        wobbleFrequency: 0.15,
        colorMix: 0.35,
        alphaScale: 0.65,
        pointScale: 0.85,
        seed: 1337,
      },
      mlp: {
        activation: 'tanh',
        hiddenLayers: [20],
        rateHz: 12,
        blend: 0.92,
        backend: 'auto',
        seed: 2024,
      },
      outputs: {
        deltaPos: { x: 0.25, y: 0.35, z: 0.25 },
        size: { min: -0.8, max: 1.2 },
        color: { min: -0.1, max: 0.2 },
        flickerRate: { min: 0.15, max: 1.5 },
        flickerDepth: { min: 0.02, max: 0.25 },
        rotationSpeed: { min: 0, max: 0.08 },
        wobbleStrength: { min: 0, max: 0.12 },
        wobbleFrequency: { min: 0.02, max: 0.45 },
        colorMix: { min: 0.2, max: 0.9 },
        alphaScale: { min: 0.4, max: 1 },
        pointScale: { min: 0.6, max: 1.6 },
        cameraZoom: { min: 5.5, max: 12 },
      },
    },
  },
];

export class UIController extends BaseModule {
  constructor(options = {}) {
    super('UIController');
    this.renderer = options.renderer || null;
    this.particleField = options.particleField || null;
    this.audioManager = options.audioManager || null;
    this.featureExtractor = options.featureExtractor || null;
    this.mlpModel = options.mlpModel || null;
    this.mlpController = options.mlpController || null;
    this.stats = options.stats || {};
    this.options = options;
    this.gui = null;
    this.sections = {};
    this.trainingManager = options.trainingManager || null;
    this.trainingPanel = null;
    this.startClosed = Object.prototype.hasOwnProperty.call(options, 'startClosed')
      ? Boolean(options.startClosed)
      : true;
    const hasCustomStorage = Object.prototype.hasOwnProperty.call(options, 'storage');
    this.store = new PresetStore({
      storageKey: options.storageKey || 'emergent-presets',
      storage: hasCustomStorage ? options.storage : undefined,
    });
    this.presets = [];
    this.defaultPresets = (options.defaultPresets && options.defaultPresets.length
      ? options.defaultPresets
      : BUILTIN_PRESETS
    ).map((preset) => ({ ...preset, builtin: preset.builtin ?? true }));
  }

  dispose() {
    this.trainingPanel?.dispose?.();
    super.dispose();
  }

  init() {
    if (this.initialized) {
      return this;
    }

    this.gui = new GUI({ title: this.options.title || 'Emergent Properties' });
    if (this.startClosed && typeof this.gui.close === 'function') {
      this.gui.close();
    }
    this._buildFeatureFolder();
    this._buildRenderFolder();
    this._buildMlpFolder();
    this._buildPresetFolder();
    this._loadPresets();
    this._mountTrainingPanel();
    super.init();
    return this;
  }

  notifyModelRandomized(details = {}) {
    this.trainingPanel?.handleModelRandomized?.(details);
  }

  addFolder(name) {
    return this.gui?.addFolder(name);
  }

  capturePreset() {
    return {
      features: this._captureFeatureSettings(),
      particles: this._captureParticleSettings(),
      mlp: this._captureMlpSettings(),
      outputs: this.mlpController?.getOutputConfig?.() || {},
    };
  }

  async applyPresetByName(name) {
    const preset = this.presets.find((item) => item.name === name);
    if (!preset) return false;
    await this.applyPreset(preset.data);
    this._selectPreset(name);
    return true;
  }

  async applyPreset(data = {}) {
    if (data.features) {
      this._applyFeatureSettings(data.features);
    }
    if (data.particles) {
      this._applyParticleSettings(data.particles);
    }
    if (data.outputs) {
      this._applyOutputSettings(data.outputs);
    }
    if (data.reactivity) {
      this._applyReactivitySettings(data.reactivity);
    }
    if (data.mlp) {
      await this._applyMlpSettings(data.mlp);
    }
  }

  async savePreset(name) {
    const finalName = PresetStore.sanitizeName(name) || this._generatePresetName();
    const data = this.capturePreset();
    this.store.upsert({ name: finalName, data });
    this._loadPresets();
    this._selectPreset(finalName);
    if (this.sections.presets) {
      this.sections.presets.state.presetName = '';
      this.sections.presets.controllers.name?.updateDisplay?.();
    }
    return finalName;
  }

  deletePreset(name) {
    if (!name || this._isBuiltin(name)) {
      if (name && this._isBuiltin(name)) {
        console.warn('[UIController] Built-in presets cannot be removed.');
      }
      return false;
    }
    const removed = this.store.remove(name);
    if (removed) {
      this._loadPresets();
    }
    return removed;
  }

  _buildFeatureFolder() {
    if (!this.featureExtractor || !this.gui) return;
    const extractor = this.featureExtractor;
    const folder = this.gui.addFolder('Audio & Features');
    const state = {
      sampleRate: extractor.options.sampleRate,
      decimate: extractor.options.decimation.enabled,
      smoothing: extractor.options.smoothing.enabled,
      smoothingAlpha: extractor.options.smoothing.alpha,
    };

    const controllers = {
      sampleRate: folder
        .add(state, 'sampleRate', 5, 120, 1)
        .name('Sample rate (Hz)')
        .onChange((value) => extractor.setSampleRate(value)),
      decimate: folder
        .add(state, 'decimate')
        .name('Decimation enabled')
        .onChange((value) => extractor.setDecimationEnabled(value)),
      smoothing: folder
        .add(state, 'smoothing')
        .name('EMA enabled')
        .onChange((value) => extractor.setSmoothingEnabled(value)),
      smoothingAlpha: folder
        .add(state, 'smoothingAlpha', 0.01, 1, 0.01)
        .name('EMA alpha')
        .onChange((value) => extractor.setSmoothingAlpha(value)),
    };

    const liveFolder = folder.addFolder('Live values');
    const liveFeatures = extractor.getFeatures();
    FEATURE_KEYS.forEach((key) => {
      const controller = liveFolder.add(liveFeatures, key).name(FEATURE_LABELS[key] || key).listen();
      controller.disable?.();
    });
    if (Object.prototype.hasOwnProperty.call(this.stats, 'fps')) {
      const controller = liveFolder.add(this.stats, 'fps').name('FPS').listen();
      controller.disable?.();
    }
    if (Object.prototype.hasOwnProperty.call(this.stats, 'inferenceMs')) {
      const controller = liveFolder.add(this.stats, 'inferenceMs').name('Inference (ms)').listen();
      controller.disable?.();
    }

    folder.open();
    liveFolder.open();
    this.sections.features = { folder, state, controllers };
  }

  _buildRenderFolder() {
    if (!this.particleField || !this.gui) return;
    const field = this.particleField;
    const folder = this.gui.addFolder('Renderer & Particles');
    const state = {
      rotationSpeed: field.options.rotationSpeed,
      wobbleStrength: field.options.wobbleStrength,
      wobbleFrequency: field.options.wobbleFrequency,
      colorMix: field.options.colorMix,
      alphaScale: field.options.alphaScale,
      pointScale: field.options.pointScale,
      seed: field.options.seed,
    };

    const controllers = {
      rotationSpeed: folder
        .add(state, 'rotationSpeed', 0, 0.4, 0.005)
        .name('Rotation speed')
        .onChange((value) => field.setRotationSpeed(value)),
      wobbleStrength: folder
        .add(state, 'wobbleStrength', 0, 0.4, 0.01)
        .name('Wobble strength')
        .onChange((value) => field.setWobbleStrength(value)),
      wobbleFrequency: folder
        .add(state, 'wobbleFrequency', 0, 1.2, 0.01)
        .name('Wobble frequency')
        .onChange((value) => field.setWobbleFrequency(value)),
      colorMix: folder
        .add(state, 'colorMix', 0.1, 1.5, 0.01)
        .name('Color mix')
        .onChange((value) => field.setColorMix(value)),
      alphaScale: folder
        .add(state, 'alphaScale', 0.2, 1.8, 0.01)
        .name('Alpha scale')
        .onChange((value) => field.setAlphaScale(value)),
      pointScale: folder
        .add(state, 'pointScale', 0.5, 2.5, 0.05)
        .name('Point scale')
        .onChange((value) => field.setPointScale(value)),
      seed: folder
        .add(state, 'seed', 0, 100000, 1)
        .name('Particle seed')
        .onFinishChange((value) => {
          field.setSeed(value);
          this._handleParticleSeedChange();
        }),
    };

    folder.add({ recenter: () => this.renderer?.focusOnParticles?.() }, 'recenter').name('Recenter camera');
    folder.open();
    this.sections.render = { folder, state, controllers };
  }

  _buildMlpFolder() {
    if (!this.mlpModel || !this.mlpController || !this.gui) return;
    const folder = this.gui.addFolder('MLP & Outputs');
    const config = this.mlpModel.getConfig();
    const outputs = this.mlpController.getOutputConfig();
    const reactivity = this.mlpController.getReactivityConfig();
    const state = {
      backend: this.mlpModel.getBackend(),
      activation: config.activation,
      layers: config.hiddenLayers.length,
      hiddenUnits: config.hiddenLayers[0] ?? 32,
      rateHz: this.mlpController.getRate(),
      blend: this.mlpController.getBlend(),
      reactAttack: reactivity.attack,
      reactRelease: reactivity.release,
      reactBoost: reactivity.boost,
      reactCurve: reactivity.curve,
      reactFloor: reactivity.floor,
      reactCeiling: reactivity.ceiling,
      reactBlendDrop: reactivity.blendDrop,
      reactMinBlend: reactivity.minBlend,
      reactFlickerBoost: reactivity.flickerBoost,
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
      rotationSpeedMin: outputs.rotationSpeed.min,
      rotationSpeedMax: outputs.rotationSpeed.max,
      wobbleStrengthMin: outputs.wobbleStrength.min,
      wobbleStrengthMax: outputs.wobbleStrength.max,
      wobbleFrequencyMin: outputs.wobbleFrequency.min,
      wobbleFrequencyMax: outputs.wobbleFrequency.max,
      colorMixMin: outputs.colorMix.min,
      colorMixMax: outputs.colorMix.max,
      alphaScaleMin: outputs.alphaScale.min,
      alphaScaleMax: outputs.alphaScale.max,
      pointScaleMin: outputs.pointScale.min,
      pointScaleMax: outputs.pointScale.max,
      cameraZoomMin: outputs.cameraZoom.min,
      cameraZoomMax: outputs.cameraZoom.max,
    };

    const applyModelUpdate = async () => {
      const layers = new Array(state.layers).fill(state.hiddenUnits);
      try {
        await this.mlpModel.rebuild({ hiddenLayers: layers, activation: state.activation });
        await this.mlpController.syncModelDimensions();
      } catch (error) {
        console.error('[UIController] Failed to rebuild model', error);
      }
    };

    const controllers = {
      backend: folder
        .add(state, 'backend', ['auto', 'webgl', 'wasm', 'cpu'])
        .name('Backend')
        .onChange(async (value) => {
          try {
            await this.mlpModel.setBackend(value);
          } catch (error) {
            console.error('[UIController] Failed to set backend', error);
          }
        }),
      activation: folder
        .add(state, 'activation', ['tanh', 'relu', 'elu'])
        .name('Activation')
        .onChange((value) => {
          state.activation = value;
          applyModelUpdate();
        }),
      layers: folder
        .add(state, 'layers', 1, 3, 1)
        .name('Hidden layers')
        .onChange((value) => {
          state.layers = value;
          applyModelUpdate();
        }),
      hiddenUnits: folder
        .add(state, 'hiddenUnits', 8, 96, 4)
        .name('Hidden units')
        .onChange((value) => {
          state.hiddenUnits = value;
          applyModelUpdate();
        }),
      rateHz: folder
        .add(state, 'rateHz', 5, 90, 1)
        .name('Inference rate (Hz)')
        .onChange((value) => this.mlpController.setRate(value)),
      blend: folder
        .add(state, 'blend', 0, 1, 0.05)
        .name('Model blend')
        .onChange((value) => this.mlpController.setBlend(value)),
    };

    const outputFolder = folder.addFolder('Output clamps');
    const outputControllers = {
      deltaPosX: outputFolder
        .add(state, 'deltaPosX', 0.1, 2, 0.05)
        .name('ΔX range')
        .onChange((value) => this.mlpController.updateOutput('deltaPos', { x: value })),
      deltaPosY: outputFolder
        .add(state, 'deltaPosY', 0.1, 2, 0.05)
        .name('ΔY range')
        .onChange((value) => this.mlpController.updateOutput('deltaPos', { y: value })),
      deltaPosZ: outputFolder
        .add(state, 'deltaPosZ', 0.1, 2, 0.05)
        .name('ΔZ range')
        .onChange((value) => this.mlpController.updateOutput('deltaPos', { z: value })),
      sizeMin: outputFolder
        .add(state, 'sizeMin', -5, 0, 0.1)
        .name('Size min')
        .onChange((value) => {
          state.sizeMin = Math.min(value, state.sizeMax - 0.1);
          this.mlpController.updateOutput('size', { min: state.sizeMin });
        }),
      sizeMax: outputFolder
        .add(state, 'sizeMax', 0, 5, 0.1)
        .name('Size max')
        .onChange((value) => {
          state.sizeMax = Math.max(value, state.sizeMin + 0.1);
          this.mlpController.updateOutput('size', { max: state.sizeMax });
        }),
      colorMin: outputFolder
        .add(state, 'colorMin', -1, 0, 0.01)
        .name('Color min')
        .onChange((value) => {
          state.colorMin = Math.min(value, state.colorMax - 0.01);
          this.mlpController.updateOutput('color', { min: state.colorMin });
        }),
      colorMax: outputFolder
        .add(state, 'colorMax', 0, 1, 0.01)
        .name('Color max')
        .onChange((value) => {
          state.colorMax = Math.max(value, state.colorMin + 0.01);
          this.mlpController.updateOutput('color', { max: state.colorMax });
        }),
      flickerRateMin: outputFolder
        .add(state, 'flickerRateMin', 0, 5, 0.05)
        .name('Flicker rate min')
        .onChange((value) => {
          state.flickerRateMin = Math.min(value, state.flickerRateMax - 0.05);
          this.mlpController.updateOutput('flickerRate', { min: state.flickerRateMin });
        }),
      flickerRateMax: outputFolder
        .add(state, 'flickerRateMax', 0.1, 8, 0.05)
        .name('Flicker rate max')
        .onChange((value) => {
          state.flickerRateMax = Math.max(value, state.flickerRateMin + 0.05);
          this.mlpController.updateOutput('flickerRate', { max: state.flickerRateMax });
        }),
      flickerDepthMin: outputFolder
        .add(state, 'flickerDepthMin', 0, 1, 0.01)
        .name('Flicker depth min')
        .onChange((value) => {
          state.flickerDepthMin = Math.min(value, state.flickerDepthMax - 0.01);
          this.mlpController.updateOutput('flickerDepth', { min: state.flickerDepthMin });
        }),
      flickerDepthMax: outputFolder
        .add(state, 'flickerDepthMax', 0, 1, 0.01)
        .name('Flicker depth max')
        .onChange((value) => {
          state.flickerDepthMax = Math.max(value, state.flickerDepthMin + 0.01);
          this.mlpController.updateOutput('flickerDepth', { max: state.flickerDepthMax });
        }),
      rotationSpeedMin: outputFolder
        .add(state, 'rotationSpeedMin', 0, 0.4, 0.005)
        .name('Rotation speed min')
        .onChange((value) => {
          state.rotationSpeedMin = Math.min(value, state.rotationSpeedMax - 0.005);
          this.mlpController.updateOutput('rotationSpeed', { min: state.rotationSpeedMin });
        }),
      rotationSpeedMax: outputFolder
        .add(state, 'rotationSpeedMax', 0.01, 0.4, 0.005)
        .name('Rotation speed max')
        .onChange((value) => {
          state.rotationSpeedMax = Math.max(value, state.rotationSpeedMin + 0.005);
          this.mlpController.updateOutput('rotationSpeed', { max: state.rotationSpeedMax });
        }),
      wobbleStrengthMin: outputFolder
        .add(state, 'wobbleStrengthMin', 0, 0.4, 0.01)
        .name('Wobble strength min')
        .onChange((value) => {
          state.wobbleStrengthMin = Math.min(value, state.wobbleStrengthMax - 0.01);
          this.mlpController.updateOutput('wobbleStrength', { min: state.wobbleStrengthMin });
        }),
      wobbleStrengthMax: outputFolder
        .add(state, 'wobbleStrengthMax', 0.01, 0.4, 0.01)
        .name('Wobble strength max')
        .onChange((value) => {
          state.wobbleStrengthMax = Math.max(value, state.wobbleStrengthMin + 0.01);
          this.mlpController.updateOutput('wobbleStrength', { max: state.wobbleStrengthMax });
        }),
      wobbleFrequencyMin: outputFolder
        .add(state, 'wobbleFrequencyMin', 0, 1.2, 0.01)
        .name('Wobble frequency min')
        .onChange((value) => {
          state.wobbleFrequencyMin = Math.min(value, state.wobbleFrequencyMax - 0.01);
          this.mlpController.updateOutput('wobbleFrequency', { min: state.wobbleFrequencyMin });
        }),
      wobbleFrequencyMax: outputFolder
        .add(state, 'wobbleFrequencyMax', 0.01, 1.2, 0.01)
        .name('Wobble frequency max')
        .onChange((value) => {
          state.wobbleFrequencyMax = Math.max(value, state.wobbleFrequencyMin + 0.01);
          this.mlpController.updateOutput('wobbleFrequency', { max: state.wobbleFrequencyMax });
        }),
      colorMixMin: outputFolder
        .add(state, 'colorMixMin', 0.1, 1.4, 0.01)
        .name('Color mix min')
        .onChange((value) => {
          state.colorMixMin = Math.min(value, state.colorMixMax - 0.01);
          this.mlpController.updateOutput('colorMix', { min: state.colorMixMin });
        }),
      colorMixMax: outputFolder
        .add(state, 'colorMixMax', 0.2, 1.5, 0.01)
        .name('Color mix max')
        .onChange((value) => {
          state.colorMixMax = Math.max(value, state.colorMixMin + 0.01);
          this.mlpController.updateOutput('colorMix', { max: state.colorMixMax });
        }),
      alphaScaleMin: outputFolder
        .add(state, 'alphaScaleMin', 0.2, 1.8, 0.01)
        .name('Alpha scale min')
        .onChange((value) => {
          state.alphaScaleMin = Math.min(value, state.alphaScaleMax - 0.01);
          this.mlpController.updateOutput('alphaScale', { min: state.alphaScaleMin });
        }),
      alphaScaleMax: outputFolder
        .add(state, 'alphaScaleMax', 0.3, 1.9, 0.01)
        .name('Alpha scale max')
        .onChange((value) => {
          state.alphaScaleMax = Math.max(value, state.alphaScaleMin + 0.01);
          this.mlpController.updateOutput('alphaScale', { max: state.alphaScaleMax });
        }),
      pointScaleMin: outputFolder
        .add(state, 'pointScaleMin', 0.4, 2.5, 0.05)
        .name('Point scale min')
        .onChange((value) => {
          state.pointScaleMin = Math.min(value, state.pointScaleMax - 0.05);
          this.mlpController.updateOutput('pointScale', { min: state.pointScaleMin });
        }),
      pointScaleMax: outputFolder
        .add(state, 'pointScaleMax', 0.5, 3, 0.05)
        .name('Point scale max')
        .onChange((value) => {
          state.pointScaleMax = Math.max(value, state.pointScaleMin + 0.05);
          this.mlpController.updateOutput('pointScale', { max: state.pointScaleMax });
        }),
      cameraZoomMin: outputFolder
        .add(state, 'cameraZoomMin', 2, 15, 0.1)
        .name('Camera zoom min')
        .onChange((value) => {
          state.cameraZoomMin = Math.min(value, state.cameraZoomMax - 0.1);
          this.mlpController.updateOutput('cameraZoom', { min: state.cameraZoomMin });
        }),
      cameraZoomMax: outputFolder
        .add(state, 'cameraZoomMax', 3, 20, 0.1)
        .name('Camera zoom max')
        .onChange((value) => {
          state.cameraZoomMax = Math.max(value, state.cameraZoomMin + 0.1);
          this.mlpController.updateOutput('cameraZoom', { max: state.cameraZoomMax });
        }),
    };

    const setReactivityOption = (key, value) => {
      this.mlpController.updateReactivity({ [key]: value });
    };

    const reactivityFolder = folder.addFolder('Reactivity');
    const reactivityControllers = {
      reactAttack: reactivityFolder
        .add(state, 'reactAttack', 0.05, 1, 0.01)
        .name('Attack')
        .onChange((value) => {
          state.reactAttack = value;
          setReactivityOption('attack', value);
        }),
      reactRelease: reactivityFolder
        .add(state, 'reactRelease', 0.01, 0.5, 0.01)
        .name('Release')
        .onChange((value) => {
          state.reactRelease = value;
          setReactivityOption('release', value);
        }),
      reactBoost: reactivityFolder
        .add(state, 'reactBoost', 0.5, 2.5, 0.05)
        .name('Boost')
        .onChange((value) => {
          state.reactBoost = value;
          setReactivityOption('boost', value);
        }),
      reactCurve: reactivityFolder
        .add(state, 'reactCurve', 0.4, 1.6, 0.05)
        .name('Curve')
        .onChange((value) => {
          state.reactCurve = value;
          setReactivityOption('curve', value);
        }),
      reactFloor: reactivityFolder
        .add(state, 'reactFloor', 0.4, 1.5, 0.05)
        .name('Floor')
        .onChange((value) => {
          state.reactFloor = value;
          setReactivityOption('floor', value);
        }),
      reactCeiling: reactivityFolder
        .add(state, 'reactCeiling', 1, 3, 0.05)
        .name('Ceiling')
        .onChange((value) => {
          state.reactCeiling = value;
          setReactivityOption('ceiling', value);
        }),
      reactBlendDrop: reactivityFolder
        .add(state, 'reactBlendDrop', 0, 0.8, 0.02)
        .name('Blend drop')
        .onChange((value) => {
          state.reactBlendDrop = value;
          setReactivityOption('blendDrop', value);
        }),
      reactMinBlend: reactivityFolder
        .add(state, 'reactMinBlend', 0, 0.9, 0.02)
        .name('Min blend')
        .onChange((value) => {
          state.reactMinBlend = value;
          setReactivityOption('minBlend', value);
        }),
      reactFlickerBoost: reactivityFolder
        .add(state, 'reactFlickerBoost', 1, 2, 0.05)
        .name('Flicker boost')
        .onChange((value) => {
          state.reactFlickerBoost = value;
          setReactivityOption('flickerBoost', value);
        }),
    };

    folder.open();
    outputFolder.open();
    reactivityFolder.open();
    this.sections.mlp = {
      folder,
      state,
      controllers: { ...controllers, ...outputControllers },
      reactivityControllers,
    };
  }

  _buildPresetFolder() {
    if (!this.gui) return;
    const folder = this.gui.addFolder('Presets');
    const state = {
      selectedPreset: '',
      presetName: '',
    };

    const controllers = {};
    controllers.select = folder
      .add(state, 'selectedPreset', this._buildPresetOptions())
      .name('Available presets')
      .onChange((value) => {
        state.selectedPreset = value;
      });

    controllers.apply = folder
      .add({ apply: () => this.applyPresetByName(state.selectedPreset) }, 'apply')
      .name('Load preset');

    controllers.name = folder
      .add(state, 'presetName')
      .name('Save as');

    controllers.save = folder
      .add({ save: () => this.savePreset(state.presetName || state.selectedPreset || 'New Preset') }, 'save')
      .name('Save preset');

    controllers.delete = folder
      .add({ delete: () => this.deletePreset(state.selectedPreset) }, 'delete')
      .name('Delete preset');

    folder.open();
    this.sections.presets = { folder, state, controllers };
  }

  _captureFeatureSettings() {
    const extractor = this.featureExtractor;
    if (!extractor) return {};
    return {
      sampleRate: extractor.options.sampleRate,
      smoothing: extractor.options.smoothing.enabled,
      smoothingAlpha: extractor.options.smoothing.alpha,
      decimate: extractor.options.decimation.enabled,
    };
  }

  _captureParticleSettings() {
    const field = this.particleField;
    if (!field) return {};
    return {
      rotationSpeed: field.options.rotationSpeed,
      wobbleStrength: field.options.wobbleStrength,
      wobbleFrequency: field.options.wobbleFrequency,
      colorMix: field.options.colorMix,
      alphaScale: field.options.alphaScale,
      pointScale: field.options.pointScale,
      seed: field.options.seed,
    };
  }

  _captureMlpSettings() {
    if (!this.mlpModel || !this.mlpController) return {};
    const config = this.mlpModel.getConfig();
    return {
      activation: config.activation,
      hiddenLayers: [...config.hiddenLayers],
      backend: this.mlpModel.getBackend(),
      rateHz: this.mlpController.getRate(),
      blend: this.mlpController.getBlend(),
      seed: config.seed,
      reactivity: this.mlpController.getReactivityConfig(),
    };
  }

  _applyFeatureSettings(settings) {
    if (!this.featureExtractor || !this.sections.features) return;
    if (Object.prototype.hasOwnProperty.call(settings, 'sampleRate')) {
      this.featureExtractor.setSampleRate(settings.sampleRate);
      this._setControllerValue(this.sections.features.controllers.sampleRate, settings.sampleRate);
    }
    if (Object.prototype.hasOwnProperty.call(settings, 'decimate')) {
      this.featureExtractor.setDecimationEnabled(settings.decimate);
      this._setControllerValue(this.sections.features.controllers.decimate, settings.decimate);
    }
    if (Object.prototype.hasOwnProperty.call(settings, 'smoothing')) {
      this.featureExtractor.setSmoothingEnabled(settings.smoothing);
      this._setControllerValue(this.sections.features.controllers.smoothing, settings.smoothing);
    }
    if (Object.prototype.hasOwnProperty.call(settings, 'smoothingAlpha')) {
      this.featureExtractor.setSmoothingAlpha(settings.smoothingAlpha);
      this._setControllerValue(this.sections.features.controllers.smoothingAlpha, settings.smoothingAlpha);
    }
  }

  _applyParticleSettings(settings) {
    if (!this.particleField || !this.sections.render) return;
    const map = {
      rotationSpeed: 'setRotationSpeed',
      wobbleStrength: 'setWobbleStrength',
      wobbleFrequency: 'setWobbleFrequency',
      colorMix: 'setColorMix',
      alphaScale: 'setAlphaScale',
      pointScale: 'setPointScale',
      seed: 'setSeed',
    };
    Object.entries(map).forEach(([key, method]) => {
      if (Object.prototype.hasOwnProperty.call(settings, key) && typeof this.particleField[method] === 'function') {
        this.particleField[method](settings[key]);
        this._setControllerValue(this.sections.render.controllers[key], settings[key]);
        if (key === 'seed') {
          this._handleParticleSeedChange();
        }
      }
    });
  }

  _handleParticleSeedChange() {
    this.mlpController?.refreshParticleState?.();
  }

  async _applyMlpSettings(settings) {
    if (!this.mlpModel || !this.mlpController || !this.sections.mlp) return;
    const config = this.mlpModel.getConfig();
    const nextLayers = Array.isArray(settings.hiddenLayers) && settings.hiddenLayers.length
      ? settings.hiddenLayers
      : config.hiddenLayers;
    const nextActivation = settings.activation || config.activation;
    await this.mlpModel.rebuild({ hiddenLayers: nextLayers, activation: nextActivation, seed: settings.seed ?? config.seed });
    if (settings.backend) {
      await this.mlpModel.setBackend(settings.backend);
      this._setControllerValue(this.sections.mlp.controllers.backend, settings.backend);
    }
    await this.mlpController.syncModelDimensions();
    if (Object.prototype.hasOwnProperty.call(settings, 'rateHz')) {
      this.mlpController.setRate(settings.rateHz);
      this._setControllerValue(this.sections.mlp.controllers.rateHz, settings.rateHz);
    }
    if (Object.prototype.hasOwnProperty.call(settings, 'blend')) {
      this.mlpController.setBlend(settings.blend);
      this._setControllerValue(this.sections.mlp.controllers.blend, settings.blend);
    }
    if (settings.reactivity) {
      this._applyReactivitySettings(settings.reactivity);
    }
    this._setControllerValue(this.sections.mlp.controllers.activation, nextActivation);
    this._setControllerValue(this.sections.mlp.controllers.layers, nextLayers.length);
    this._setControllerValue(this.sections.mlp.controllers.hiddenUnits, nextLayers[0] ?? config.hiddenLayers[0]);
  }

  _applyOutputSettings(settings) {
    if (!this.mlpController || !this.sections.mlp) return;
    Object.entries(settings).forEach(([key, value]) => {
      this.mlpController.updateOutput(key, value);
      Object.entries(value).forEach(([axis, axisValue]) => {
        const property = this._outputKeyToStateKey(key, axis);
        if (property && this.sections.mlp.controllers[property]) {
          this._setControllerValue(this.sections.mlp.controllers[property], axisValue);
        }
      });
    });
  }

  _outputKeyToStateKey(key, axis) {
    const map = {
      deltaPos: { x: 'deltaPosX', y: 'deltaPosY', z: 'deltaPosZ' },
      size: { min: 'sizeMin', max: 'sizeMax' },
      color: { min: 'colorMin', max: 'colorMax' },
      flickerRate: { min: 'flickerRateMin', max: 'flickerRateMax' },
      flickerDepth: { min: 'flickerDepthMin', max: 'flickerDepthMax' },
      rotationSpeed: { min: 'rotationSpeedMin', max: 'rotationSpeedMax' },
      wobbleStrength: { min: 'wobbleStrengthMin', max: 'wobbleStrengthMax' },
      wobbleFrequency: { min: 'wobbleFrequencyMin', max: 'wobbleFrequencyMax' },
      colorMix: { min: 'colorMixMin', max: 'colorMixMax' },
      alphaScale: { min: 'alphaScaleMin', max: 'alphaScaleMax' },
      pointScale: { min: 'pointScaleMin', max: 'pointScaleMax' },
      cameraZoom: { min: 'cameraZoomMin', max: 'cameraZoomMax' },
    };
    return map[key]?.[axis] || null;
  }

  _buildPresetOptions() {
    const names = this.presets?.map((preset) => preset.name) || [];
    if (!names.length) {
      return { '(none)': '' };
    }
    return names.reduce((acc, name) => {
      acc[name] = name;
      return acc;
    }, {});
  }

  _loadPresets() {
    const byName = new Map();
    this.defaultPresets.forEach((preset) => {
      byName.set(preset.name, { ...preset, builtin: true });
    });
    this.store.getAll().forEach((preset) => {
      byName.set(preset.name, { ...preset, builtin: false });
    });
    this.presets = Array.from(byName.values());
    if (this.sections.presets?.controllers.select) {
      const options = this._buildPresetOptions();
      const select = this.sections.presets.controllers.select;
      if (typeof select.options === 'function') {
        select.options(options);
      }
      const first = Object.values(options)[0] || '';
      if (!options[this.sections.presets.state.selectedPreset]) {
        this.sections.presets.state.selectedPreset = first;
        if (typeof select.setValue === 'function') {
          select.setValue(first);
        } else {
          select.object[select.property] = first;
          select.updateDisplay?.();
        }
      }
    }
  }

  _selectPreset(name) {
    if (!this.sections.presets?.controllers.select) return;
    if (!name) return;
    const select = this.sections.presets.controllers.select;
    this.sections.presets.state.selectedPreset = name;
    if (typeof select.setValue === 'function') {
      select.setValue(name);
    } else {
      select.object[select.property] = name;
      select.updateDisplay?.();
    }
  }

  _generatePresetName() {
    const base = 'Preset';
    let index = this.presets.length + 1;
    let candidate = `${base} ${index}`;
    const names = new Set(this.presets.map((preset) => preset.name));
    while (names.has(candidate)) {
      index += 1;
      candidate = `${base} ${index}`;
    }
    return candidate;
  }

  _isBuiltin(name) {
    return this.presets.some((preset) => preset.name === name && preset.builtin);
  }

  _setControllerValue(controller, value) {
    if (!controller) return;
    if (typeof controller.setValue === 'function') {
      controller.setValue(value);
    } else {
      controller.object[controller.property] = value;
      controller.updateDisplay?.();
    }
  }
  _applyReactivitySettings(settings) {
    if (!this.mlpController || !this.sections.mlp?.reactivityControllers) return;
    this.mlpController.updateReactivity(settings);
    const map = [
      ['attack', 'reactAttack'],
      ['release', 'reactRelease'],
      ['boost', 'reactBoost'],
      ['curve', 'reactCurve'],
      ['floor', 'reactFloor'],
      ['ceiling', 'reactCeiling'],
      ['blendDrop', 'reactBlendDrop'],
      ['minBlend', 'reactMinBlend'],
      ['flickerBoost', 'reactFlickerBoost'],
    ];
    map.forEach(([optionKey, stateKey]) => {
      if (
        Object.prototype.hasOwnProperty.call(settings, optionKey) &&
        this.sections.mlp.reactivityControllers[stateKey]
      ) {
        this._setControllerValue(this.sections.mlp.reactivityControllers[stateKey], settings[optionKey]);
      }
    });
  }

  _mountTrainingPanel() {
    if (!this.trainingManager || this.trainingPanel) {
      return;
    }
    const positionalLabels = PARTICLE_POSITIONAL_FEATURES.reduce((acc, feature) => {
      acc[feature.id] = feature.label;
      return acc;
    }, {});
    this.trainingPanel = new TrainingPanel({
      trainingManager: this.trainingManager,
      mlpModel: this.mlpModel,
      mlpController: this.mlpController,
      featureKeys: FEATURE_KEYS,
      featureLabels: {
        ...FEATURE_LABELS,
        ...positionalLabels,
      },
      positionalFeatures: PARTICLE_POSITIONAL_FEATURES,
      trainingOptions: this.options.mlpTrainingOptions || {},
    });
    this.trainingPanel.init();
  }
}
