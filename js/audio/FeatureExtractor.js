import { BaseModule } from '../core/BaseModule.js';

export const FEATURE_KEYS = [
  'rms',
  'specCentroid',
  'specRolloff',
  'bandSub',
  'bandBass',
  'bandLowMid',
  'bandMid',
  'bandHigh',
  'peak',
  'zeroCrossRate',
  'tempoProxy',
];

const DEFAULT_BANDS = Object.freeze({
  bandSub: [20, 60],
  bandBass: [60, 250],
  bandLowMid: [250, 500],
  bandMid: [500, 2000],
  bandHigh: [2000, 8000],
});

const LEGACY_BAND_KEYS = {
  low: 'bandBass',
  bandLow: 'bandBass',
  mid: 'bandMid',
  bandMid: 'bandMid',
  high: 'bandHigh',
  bandHigh: 'bandHigh',
};

const DEFAULT_OPTIONS = {
  sampleRate: 30, // target feature refresh rate (Hz)
  smoothing: {
    enabled: false,
    alpha: 0.25,
  },
  decimation: {
    enabled: true,
  },
  bands: DEFAULT_BANDS,
  rolloffPercent: 0.85,
  tempo: {
    minBpm: 60,
    maxBpm: 180,
    sensitivity: 0.08,
    energyAlpha: 0.15,
    bpmAlpha: 0.35,
  },
  fallbackSampleRate: 44100,
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const clamp01 = (value) => clamp(value, 0, 1);
const createFeatureTemplate = () =>
  FEATURE_KEYS.reduce((acc, key) => {
    acc[key] = 0;
    return acc;
  }, {});

const isValidBandRange = (value) =>
  Array.isArray(value) &&
  value.length === 2 &&
  value.every((entry) => Number.isFinite(entry));

const sanitizeBands = (custom = {}) => {
  const normalized = { ...DEFAULT_BANDS };
  Object.entries(custom || {}).forEach(([rawKey, range]) => {
    const key = LEGACY_BAND_KEYS[rawKey] || rawKey;
    if (!normalized[key] || !isValidBandRange(range)) {
      return;
    }
    const [startHz, endHz] = range;
    if (Math.max(0, endHz) <= Math.max(0, startHz)) {
      return;
    }
    normalized[key] = [Math.max(0, startHz), Math.max(0, endHz)];
  });
  return normalized;
};

const mergeOptions = (overrides = {}) => ({
  ...DEFAULT_OPTIONS,
  ...overrides,
  smoothing: {
    ...DEFAULT_OPTIONS.smoothing,
    ...(overrides.smoothing || {}),
  },
  decimation: {
    ...DEFAULT_OPTIONS.decimation,
    ...(overrides.decimation || {}),
  },
  bands: sanitizeBands(overrides.bands),
  tempo: {
    ...DEFAULT_OPTIONS.tempo,
    ...(overrides.tempo || {}),
  },
});

export class FeatureExtractor extends BaseModule {
  constructor(analyser, options = {}) {
    super('FeatureExtractor');
    this.analyser = analyser || null;
    this.options = mergeOptions(options);
    this.features = createFeatureTemplate();
    this.frequencyData = null;
    this.timeDomainData = null;
    this._listeners = new Set();
    this._clock = 0;
    this._accumulator = 0;
    this._hasSampled = false;
    this._binWidth = 0;
    this._nyquist = 0;
    this._bandBinRanges = Object.keys(this.options.bands).reduce((acc, key, index) => {
      acc[key] = [index, index + 1];
      return acc;
    }, {});
    this._tempoState = {
      energyEMA: 0,
      lastBeatTime: 0,
      bpm: 0,
    };
    this._sampleInterval = 1 / Math.max(1, this.options.sampleRate);
    if (this.analyser) {
      this._configureBuffers();
    }
  }

  init() {
    if (this.initialized) {
      return this;
    }
    super.init();
    this._configureBuffers();
    return this;
  }

  dispose() {
    this._listeners.clear();
    super.dispose();
  }

  onUpdate(handler) {
    if (typeof handler === 'function') {
      this._listeners.add(handler);
      return () => this.offUpdate(handler);
    }
    return () => {};
  }

  offUpdate(handler) {
    this._listeners.delete(handler);
  }

  getFeatures() {
    return this.features;
  }

  setAnalyser(analyser) {
    this.analyser = analyser;
    if (this.initialized) {
      this._configureBuffers();
    }
  }

  setSampleRate(value) {
    const next = clamp(value || DEFAULT_OPTIONS.sampleRate, 1, 120);
    this.options.sampleRate = next;
    this._sampleInterval = 1 / next;
  }

  setSmoothingEnabled(enabled) {
    this.options.smoothing.enabled = Boolean(enabled);
  }

  setSmoothingAlpha(alpha) {
    this.options.smoothing.alpha = clamp(alpha || 0.25, 0.01, 1);
  }

  setDecimationEnabled(enabled) {
    this.options.decimation.enabled = Boolean(enabled);
    if (!enabled) {
      this._accumulator = 0;
    }
  }

  sample(deltaSeconds = 0) {
    const dt = Number.isFinite(deltaSeconds) && deltaSeconds > 0 ? deltaSeconds : 0;
    this._clock += dt;

    if (!this.initialized && this.analyser) {
      this.init();
    }

    if (!this.analyser || !this.frequencyData || !this.timeDomainData) {
      return this.features;
    }

    let shouldSample = !this._hasSampled;
    if (this.options.decimation.enabled) {
      this._accumulator += dt;
      if (this._accumulator >= this._sampleInterval || !this._hasSampled) {
        shouldSample = true;
        this._accumulator = 0;
      }
    } else {
      shouldSample = true;
    }

    if (!shouldSample) {
      return this.features;
    }

    this._hasSampled = true;
    this.analyser.getByteFrequencyData(this.frequencyData);
    this.analyser.getByteTimeDomainData(this.timeDomainData);

    const freqData = this.frequencyData;
    const timeData = this.timeDomainData;

    const totalEnergy =
      freqData.reduce((sum, value) => sum + value, 0) || 1;

    let centroidNumerator = 0;
    let cumulativeEnergy = 0;
    let rolloffFrequency = this._nyquist;
    let peak = 0;
    const rolloffTarget = totalEnergy * this.options.rolloffPercent;

    for (let i = 0; i < freqData.length; i++) {
      const magnitude = freqData[i];
      if (magnitude > peak) peak = magnitude;
      const freq = i * this._binWidth;
      centroidNumerator += freq * magnitude;
      cumulativeEnergy += magnitude;
      if (rolloffFrequency === this._nyquist && cumulativeEnergy >= rolloffTarget) {
        rolloffFrequency = freq;
      }
    }

    const centroidHz = totalEnergy > 0 ? centroidNumerator / totalEnergy : 0;
    const bandFeatures = Object.entries(this._bandBinRanges).reduce((acc, [key, range]) => {
      acc[key] = this._computeBandEnergy(freqData, range);
      return acc;
    }, {});

    const rms = this._computeRms(timeData);
    const zeroCrossRate = this._computeZeroCrossRate(timeData);
    const tempoProxy = this._updateTempo(
      Math.max(
        bandFeatures.bandBass ?? 0,
        bandFeatures.bandSub ?? 0,
      ),
    );

    const nextFeatures = {
      rms,
      specCentroid: clamp01(this._nyquist ? centroidHz / this._nyquist : 0),
      specRolloff: clamp01(this._nyquist ? rolloffFrequency / this._nyquist : 0),
      peak: clamp01(peak / 255),
      zeroCrossRate,
      tempoProxy,
      ...bandFeatures,
    };

    FEATURE_KEYS.forEach((key) => {
      const current = this.features[key];
      const next = nextFeatures[key];
      this.features[key] = this._smoothValue(current, next);
    });

    this._emitUpdate();
    return this.features;
  }

  _configureBuffers() {
    if (!this.analyser) {
      return;
    }
    const bins = this.analyser.frequencyBinCount || 0;
    if (!bins) {
      return;
    }
    const timeSize = this.analyser.fftSize || bins * 2;
    this.frequencyData = new Uint8Array(bins);
    this.timeDomainData = new Uint8Array(timeSize);
    const sampleRate = this._getSampleRate();
    this._nyquist = sampleRate / 2;
    this._binWidth = bins ? this._nyquist / bins : 0;
    this._bandBinRanges = this._computeBandBins();
  }

  _computeBandBins() {
    if (!this.analyser || !this._binWidth) {
      return { ...this._bandBinRanges };
    }
    const maxBin = this.analyser.frequencyBinCount || 1;
    return Object.entries(this.options.bands).reduce((acc, [key, [startHz, endHz]]) => {
      const start = clamp(Math.floor(startHz / this._binWidth), 0, maxBin - 1);
      const end = clamp(Math.ceil(endHz / this._binWidth), start + 1, maxBin);
      acc[key] = [start, end];
      return acc;
    }, {});
  }

  _computeBandEnergy(data, range = [0, data.length]) {
    if (!data || !data.length) {
      return 0;
    }
    const [start, end] = range;
    const length = Math.max(end - start, 1);
    let sum = 0;
    for (let i = start; i < end && i < data.length; i++) {
      sum += data[i];
    }
    return clamp01((sum / length) / 255);
  }

  _computeRms(timeData) {
    if (!timeData || !timeData.length) {
      return 0;
    }
    let sumSquares = 0;
    for (let i = 0; i < timeData.length; i++) {
      const sample = (timeData[i] - 128) / 128;
      sumSquares += sample * sample;
    }
    return clamp01(Math.sqrt(sumSquares / timeData.length));
  }

  _computeZeroCrossRate(timeData) {
    if (!timeData || timeData.length <= 1) {
      return 0;
    }
    let zeroCrossings = 0;
    let prev = (timeData[0] - 128) / 128;
    for (let i = 1; i < timeData.length; i++) {
      const sample = (timeData[i] - 128) / 128;
      if ((sample >= 0 && prev < 0) || (sample < 0 && prev >= 0)) {
        zeroCrossings += 1;
      }
      prev = sample;
    }
    return clamp01(zeroCrossings / (timeData.length - 1));
  }

  _updateTempo(lowEnergy) {
    const tempo = this.options.tempo;
    const state = this._tempoState;
    state.energyEMA += tempo.energyAlpha * (lowEnergy - state.energyEMA);
    const threshold = state.energyEMA + tempo.sensitivity;
    const now = this._clock;
    const minInterval = 60 / tempo.maxBpm;
    const maxInterval = 60 / tempo.minBpm;

    if (lowEnergy > threshold && (now - state.lastBeatTime >= minInterval || state.lastBeatTime === 0)) {
      const interval = state.lastBeatTime ? now - state.lastBeatTime : 0;
      state.lastBeatTime = now;
      if (interval >= minInterval && interval <= maxInterval) {
        const bpm = clamp(60 / interval, tempo.minBpm, tempo.maxBpm);
        state.bpm = state.bpm
          ? state.bpm + tempo.bpmAlpha * (bpm - state.bpm)
          : bpm;
      }
    } else if (state.bpm > 0) {
      // Gentle decay if no beats are detected for a while
      state.bpm *= 0.995;
      if (state.bpm < tempo.minBpm * 0.5) {
        state.bpm = 0;
      }
    }

    return state.bpm ? clamp01(state.bpm / tempo.maxBpm) : 0;
  }

  _smoothValue(previous, next) {
    if (!this.options.smoothing.enabled || !Number.isFinite(previous)) {
      return clamp01(next);
    }
    const alpha = this.options.smoothing.alpha;
    const prev = Number.isFinite(previous) ? previous : next;
    const value = prev + alpha * (next - prev);
    return clamp01(value);
  }

  _getSampleRate() {
    return (
      this.analyser?.context?.sampleRate ||
      this.options.fallbackSampleRate ||
      DEFAULT_OPTIONS.fallbackSampleRate
    );
  }

  _emitUpdate() {
    for (const handler of this._listeners) {
      try {
        handler(this.features);
      } catch (error) {
        console.error('[FeatureExtractor] update listener failed', error);
      }
    }
  }
}
