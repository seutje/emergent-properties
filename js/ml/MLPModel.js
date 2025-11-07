import { BaseModule } from '../core/BaseModule.js';
import { serializeModelWeights, applySerializedWeights } from './MLPWeightUtils.js';

const DEFAULT_CONFIG = {
  inputSize: 17,
  outputSize: 9,
  hiddenLayers: [32],
  activation: 'tanh',
  outputActivation: 'tanh',
  backend: 'auto',
  seed: 1337,
};

const VALID_BACKENDS = ['auto', 'webgl', 'wasm', 'cpu', 'tensorflow'];

const toArray = (value) => {
  if (Array.isArray(value) && value.length) {
    return value.map((v) => Math.max(1, Math.floor(v))).filter((v) => v > 0);
  }
  const asNumber = Number.isFinite(value) ? Number(value) : null;
  if (asNumber && asNumber > 0) {
    return [Math.floor(asNumber)];
  }
  return [...DEFAULT_CONFIG.hiddenLayers];
};

export class MLPModel extends BaseModule {
  constructor(config = {}) {
    super('MLPModel');
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      hiddenLayers: toArray(config.hiddenLayers ?? DEFAULT_CONFIG.hiddenLayers),
    };
    this.model = null;
    this.tf = null;
    this.backend = 'auto';
  }

  async init() {
    if (this.initialized) {
      return this;
    }
    await this._ensureTensorFlow();
    await this._configureBackend(this.config.backend);
    this._buildModel();
    super.init();
    return this;
  }

  async rebuild(config = {}) {
    this.config = {
      ...this.config,
      ...config,
      hiddenLayers: toArray(config.hiddenLayers ?? this.config.hiddenLayers),
    };
    if (!this.tf) {
      await this._ensureTensorFlow();
    }
    if (config.backend) {
      await this._configureBackend(config.backend);
    }
    this._buildModel();
    return this;
  }

  async setBackend(backend = 'auto') {
    await this._configureBackend(backend);
    return this.backend;
  }

  getBackend() {
    return this.backend;
  }

  getConfig() {
    return { ...this.config, hiddenLayers: [...this.config.hiddenLayers] };
  }

  getTF() {
    if (!this.tf) {
      throw new Error('[MLPModel] TensorFlow is not ready. Call init() first.');
    }
    return this.tf;
  }

  predict(inputTensor) {
    if (!this.model) {
      throw new Error('[MLPModel] Model not initialized. Call init() before predict().');
    }
    return this.tf.tidy(() => this.model.predict(inputTensor));
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.tf = null;
    super.dispose();
  }

  async exportSnapshot(metadata = {}) {
    if (!this.model) {
      throw new Error('[MLPModel] Cannot export snapshot before init().');
    }
    const weights = await serializeModelWeights(this.model);
    return {
      version: 1,
      createdAt: Date.now(),
      backend: this.getBackend(),
      config: this.getConfig(),
      metadata,
      weights,
    };
  }

  async applyWeights(serialized = []) {
    if (!this.model) {
      throw new Error('[MLPModel] Model not initialized.');
    }
    if (!serialized?.length) {
      return;
    }
    const tf = this.getTF();
    applySerializedWeights(this.model, tf, serialized);
  }

  async importSnapshot(snapshot = {}) {
    if (!snapshot || !snapshot.weights) {
      throw new Error('[MLPModel] Invalid model snapshot.');
    }
    const config = snapshot.config || this.getConfig();
    await this.rebuild(config);
    await this.applyWeights(snapshot.weights);
    return snapshot.metadata || null;
  }

  async _ensureTensorFlow() {
    if (this.tf) {
      return this.tf;
    }
    if (globalThis.tf) {
      this.tf = globalThis.tf;
      return this.tf;
    }
    if (typeof window === 'undefined') {
      const tryLoad = async (specifier, { quiet = false } = {}) => {
        try {
          const module = await import(specifier);
          return module?.default ?? module;
        } catch (error) {
          if (!quiet) {
            console.warn(`[MLPModel] Failed to load ${specifier}`, error);
          }
          return null;
        }
      };

      // Prefer the accelerated native backend when running under Node.js tests.
      const tfNode = await tryLoad('@tensorflow/tfjs-node', { quiet: true });
      if (tfNode) {
        this.tf = tfNode;
        return this.tf;
      }

      const tf = await tryLoad('@tensorflow/tfjs');
      if (tf) {
        this.tf = tf;
        return this.tf;
      }

      throw new Error('[MLPModel] Unable to load TensorFlow.js in Node environment.');
    }
    throw new Error('[MLPModel] TensorFlow.js not found on window. Include tf.min.js before main.js.');
  }

  async _configureBackend(target = 'auto') {
    if (!this.tf) {
      await this._ensureTensorFlow();
    }
    const normalized = VALID_BACKENDS.includes(target) ? target : 'auto';
    if (normalized !== 'auto' && this.tf.setBackend) {
      try {
        await this.tf.setBackend(normalized);
      } catch (error) {
        console.warn(`[MLPModel] Failed to set backend "${normalized}"`, error);
      }
    }
    await this.tf.ready?.();
    this.backend = this.tf?.getBackend?.() || normalized;
  }

  _buildModel() {
    if (!this.tf) {
      throw new Error('[MLPModel] TensorFlow is not ready.');
    }
    if (this.model) {
      this.model.dispose();
    }
    const tf = this.tf;
    const { inputSize, outputSize, hiddenLayers, activation, outputActivation, seed } = this.config;
    const initializer = tf.initializers.randomUniform({
      minval: -0.5,
      maxval: 0.5,
      seed,
    });

    const layers = [];
    hiddenLayers.forEach((units, index) => {
      layers.push(
        tf.layers.dense({
          units,
          activation,
          useBias: true,
          inputShape: index === 0 ? [inputSize] : undefined,
          kernelInitializer: initializer,
          biasInitializer: 'zeros',
        }),
      );
    });

    layers.push(
      tf.layers.dense({
        units: outputSize,
        activation: outputActivation,
        useBias: true,
        kernelInitializer: initializer,
        biasInitializer: 'zeros',
      }),
    );

    this.model = tf.sequential({ layers });
    return this.model;
  }
}
