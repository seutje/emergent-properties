import { BaseModule } from '../core/BaseModule.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { sanitizeCorrelationList } from './MLPTrainingUtils.js';

const DEFAULT_TRAINING_OPTIONS = {
  epochs: 24,
  batchSize: 128,
  learningRate: 0.0015,
  sampleCount: 4096,
  noise: 0.04,
  seed: Date.now(),
};

const noop = () => {};

export class MLPTrainingManager extends BaseModule {
  constructor(options = {}) {
    super('MLPTrainingManager');
    this.model = options.model || null;
    this.featureKeys = options.featureKeys || FEATURE_KEYS;
    this.baseDims = options.baseDims || 0;
    this.audioDims = options.audioDims || this.featureKeys.length;
    this.correlations = [];
    this.worker = null;
    this.trainingOptions = { ...DEFAULT_TRAINING_OPTIONS, ...(options.training || {}) };
    this.listeners = new Map();
    this.state = {
      status: 'idle',
      epoch: 0,
      epochs: this.trainingOptions.epochs,
      loss: null,
      best: null,
      metadata: null,
    };
    this.options = options;
    this.pendingStart = null;
    this.lastWeights = null;
    this.targets = [];
  }

  init() {
    if (this.initialized) {
      return this;
    }
    this._spawnWorker();
    super.init();
    return this;
  }

  dispose() {
    this.worker?.terminate?.();
    this.worker = null;
    this.listeners.clear();
    super.dispose();
  }

  on(event, handler = noop) {
    if (typeof handler !== 'function') {
      return noop;
    }
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    const bucket = this.listeners.get(event);
    bucket.add(handler);
    return () => bucket.delete(handler);
  }

  off(event, handler) {
    const bucket = this.listeners.get(event);
    bucket?.delete(handler);
  }

  setCorrelations(list = []) {
    this.correlations = sanitizeCorrelationList(list, this.featureKeys);
    this._emit('correlations', this.correlations);
    return this.correlations;
  }

  getCorrelations() {
    return this.correlations.slice();
  }

  updateTrainingOptions(changes = {}) {
    this.trainingOptions = {
      ...this.trainingOptions,
      ...(changes || {}),
    };
    return this.trainingOptions;
  }

  getTrainingOptions() {
    return { ...this.trainingOptions };
  }

  async startTraining(overrides = {}) {
    if (!this.worker) {
      this._spawnWorker();
    }
    if (!this.worker) {
      throw new Error('Training worker is unavailable in this environment.');
    }
    if (this.state.status === 'running') {
      throw new Error('Training already in progress.');
    }
    if (!this.correlations.length) {
      throw new Error('At least one correlation must be defined before training.');
    }
    const modelConfig = this.model?.getConfig?.();
    if (!modelConfig) {
      throw new Error('MLP model configuration is unavailable.');
    }
    const baseSampleProvider = this.options.getBaseSamples || this.options.baseSampleProvider || null;
    let baseSamples = { buffer: null, count: 0, dims: this.baseDims };
    if (typeof baseSampleProvider === 'function') {
      baseSamples = (await baseSampleProvider()) || baseSamples;
    }
    const { baseWeights = null, ...trainingOverrides } = overrides || {};
    const mergedOptions = {
      ...this.trainingOptions,
      ...trainingOverrides,
    };

    const payload = {
      correlations: this.correlations,
      featureKeys: this.featureKeys,
      baseSamples: baseSamples?.buffer ? baseSamples.buffer.slice(0) : null,
      baseSampleCount: baseSamples?.count || 0,
      baseDims: baseSamples?.dims ?? this.baseDims ?? 0,
      audioDims: this.audioDims,
      model: {
        inputSize: modelConfig.inputSize,
        outputSize: modelConfig.outputSize,
        hiddenLayers: modelConfig.hiddenLayers,
        activation: modelConfig.activation,
        outputActivation: modelConfig.outputActivation,
      },
      training: mergedOptions,
    };
    if (Array.isArray(baseWeights) && baseWeights.length) {
      payload.baseWeights = baseWeights;
    }

    const transfer = payload.baseSamples ? [payload.baseSamples.buffer] : [];
    this.worker.postMessage({ type: 'start', payload }, transfer);
    this._setState({
      status: 'running',
      epoch: 0,
      epochs: mergedOptions.epochs ?? this.trainingOptions.epochs,
      loss: null,
      metadata: null,
    });
  }

  pauseTraining() {
    if (!this.worker || this.state.status !== 'running') {
      return;
    }
    this.worker.postMessage({ type: 'pause' });
  }

  resumeTraining() {
    if (!this.worker || this.state.status !== 'paused') {
      return;
    }
    this.worker.postMessage({ type: 'resume' });
  }

  abortTraining() {
    if (!this.worker) {
      return;
    }
    this.worker.postMessage({ type: 'abort' });
  }

  getState() {
    return { ...this.state };
  }

  getLastWeights() {
    return this.lastWeights ? JSON.parse(JSON.stringify(this.lastWeights)) : null;
  }

  _spawnWorker() {
    if (typeof Worker === 'undefined') {
      console.warn('[MLPTrainingManager] Web Workers are not available in this environment.');
      return;
    }
    this.worker = new Worker(new URL('./MLPTrainingWorker.js', import.meta.url), {
      type: 'module',
    });
    this.worker.addEventListener('message', (event) => this._handleWorkerMessage(event));
  }

  _handleWorkerMessage(event) {
    const { type, payload } = event.data || {};
    switch (type) {
      case 'ready':
        this.targets = payload?.targets || [];
        this._emit('ready', this.targets);
        break;
      case 'status':
        this._setState({
          status: payload?.status || this.state.status,
          epoch: payload?.epoch ?? this.state.epoch,
          loss: payload?.loss ?? this.state.loss,
        });
        break;
      case 'progress':
        this._setState({
          status: 'running',
          epoch: payload?.epoch ?? this.state.epoch,
          epochs: payload?.epochs ?? this.state.epochs,
          loss: payload?.loss ?? this.state.loss,
        });
        this._emit('progress', payload);
        break;
      case 'best':
        this._emit('best', payload);
        break;
      case 'result':
        this.lastWeights = payload?.weights || null;
        this._setState({
          status: payload?.status || 'idle',
          metadata: payload?.metadata || null,
          loss: payload?.metadata?.loss ?? this.state.loss,
        });
        this._emit('result', payload);
        break;
      case 'error':
        console.error('[MLPTrainingManager] Worker error', payload?.message);
        this._setState({ status: 'error' });
        this._emit('error', payload);
        break;
      default:
        break;
    }
  }

  _setState(partial = {}) {
    this.state = {
      ...this.state,
      ...partial,
    };
    this._emit('state', this.getState());
  }

  _emit(event, payload) {
    const handlers = this.listeners.get(event);
    if (!handlers || !handlers.size) {
      return;
    }
    handlers.forEach((handler) => {
      try {
        handler(payload);
      } catch (error) {
        console.error(`[MLPTrainingManager] Listener for ${event} failed`, error);
      }
    });
  }
}
