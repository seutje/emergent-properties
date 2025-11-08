import { BaseModule } from '../core/BaseModule.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';

const OUTPUT_INDEX = PARTICLE_PARAMETER_TARGETS.reduce((acc, target) => {
  acc[target.id] = target.outputIndex;
  return acc;
}, {});

const GLOBAL_OUTPUT_SETTERS = {
  rotationSpeed: 'setRotationSpeed',
  wobbleStrength: 'setWobbleStrength',
  wobbleFrequency: 'setWobbleFrequency',
  colorMix: 'setColorMix',
  alphaScale: 'setAlphaScale',
  pointScale: 'setPointScale',
};

const GLOBAL_OUTPUT_KEYS = Object.keys(GLOBAL_OUTPUT_SETTERS);

const DEFAULT_OUTPUTS = {
  deltaPos: { x: 1.25, y: 1.6, z: 0.45 },
  size: { min: -4.3, max: 4.5 },
  color: { min: -0.91, max: 0.99 },
  flickerRate: { min: 0.25, max: 7.55 },
  flickerDepth: { min: 0.06, max: 0.84 },
  rotationSpeed: { min: 0.005, max: 0.22 },
  wobbleStrength: { min: 0, max: 0.25 },
  wobbleFrequency: { min: 0.05, max: 0.95 },
  colorMix: { min: 0.25, max: 1.4 },
  alphaScale: { min: 0.4, max: 1.6 },
  pointScale: { min: 0.75, max: 2.35 },
};

export const DEFAULT_REACTIVITY = {
  attack: 0.54,
  release: 0.08,
  boost: 1.7,
  curve: 0.85,
  floor: 0.75,
  ceiling: 1.9,
  blendDrop: 0.35,
  minBlend: 0.35,
  flickerBoost: 1.65,
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (a, b, t) => a + (b - a) * clamp(t, 0, 1);
const mapMinusOneToOne = (value, min, max) => {
  const normalized = (clamp(value, -1, 1) + 1) * 0.5;
  return lerp(min, max, normalized);
};
const clamp01 = (value) => clamp(value, 0, 1);

export function deriveReactivity(features = {}, prevState = {}, options = DEFAULT_REACTIVITY, baseBlend = 1) {
  const opts = {
    ...DEFAULT_REACTIVITY,
    ...(options || {}),
  };
  const prev = {
    envelope: clamp01(prevState?.envelope ?? 0),
    prevPeak: clamp01(prevState?.prevPeak ?? 0),
  };
  const rms = clamp01(features.rms ?? 0);
  const low = clamp01(features.bandLow ?? 0);
  const mid = clamp01(features.bandMid ?? 0);
  const high = clamp01(features.bandHigh ?? 0);
  const peak = clamp01(features.peak ?? 0);
  const tempo = clamp01(features.tempoProxy ?? 0);

  const drive = clamp01(rms * 0.45 + low * 0.3 + mid * 0.15 + high * 0.1 + tempo * 0.12);
  const transient = Math.max(0, peak - prev.prevPeak) * 0.5;
  const target = clamp01(drive + transient);
  const attack = clamp(opts.attack ?? DEFAULT_REACTIVITY.attack, 0.01, 1);
  const release = clamp(Math.min(opts.release ?? DEFAULT_REACTIVITY.release, attack), 0.01, 1);
  const delta = target - prev.envelope;
  const coeff = delta >= 0 ? attack : release;
  const envelope = clamp01(prev.envelope + delta * coeff);

  const boost = clamp(opts.boost ?? DEFAULT_REACTIVITY.boost, 0.1, 5);
  const curve = clamp(opts.curve ?? DEFAULT_REACTIVITY.curve, 0.25, 2);
  const normalized = clamp01(envelope * boost);
  const curved = Math.pow(normalized, curve);

  const floor = Math.max(0, opts.floor ?? DEFAULT_REACTIVITY.floor);
  const ceiling = Math.max(floor + 0.01, opts.ceiling ?? DEFAULT_REACTIVITY.ceiling);
  const gain = lerp(floor, ceiling, curved);

  const blendDrop = clamp01(opts.blendDrop ?? DEFAULT_REACTIVITY.blendDrop);
  const minBlend = clamp(opts.minBlend ?? DEFAULT_REACTIVITY.minBlend, 0, 1);
  const base = clamp(baseBlend ?? 0.9, 0, 1);
  const blend = clamp(base - curved * blendDrop, minBlend, base);

  const flickerBoost = lerp(1, Math.max(1, opts.flickerBoost ?? DEFAULT_REACTIVITY.flickerBoost), curved);

  return {
    envelope,
    curved,
    gain,
    blend,
    flickerBoost,
    prevPeak: peak,
  };
}

export class MLPOrchestrator extends BaseModule {
  constructor({ model, particleField, featureExtractor, options = {} } = {}) {
    super('MLPOrchestrator');
    this.model = model;
    this.particleField = particleField;
    this.featureExtractor = featureExtractor;
    this.options = {
      enabled: true,
      rateHz: 24,
      blend: 1,
      outputs: { ...DEFAULT_OUTPUTS, ...(options.outputs || {}) },
      reactivity: { ...DEFAULT_REACTIVITY, ...(options.reactivity || {}) },
      ...options,
    };

    this.count = 0;
    this.baseDims = 8;
    this.audioDims = FEATURE_KEYS.length;
    this.totalInputSize = this.baseDims + this.audioDims;
    this.outputDims = PARTICLE_PARAMETER_TARGETS.length;
    this.interval = 1 / this.options.rateHz;
    this.accumulator = 0;
    this._pending = null;
    this._tf = null;
    this.baseTensor = null;
    this.baseBuffer = null;
    this.featureVector = new Float32Array(this.audioDims);
    this.attributeHandles = null;
    this.lastInferenceMs = 0;
    this.baseFlickerRate = null;
    this.baseFlickerDepth = null;
    this.lastFeatures = {};
    this._reactivityState = {
      envelope: 0,
      prevPeak: 0,
    };
  }

  async init() {
    if (!this.model || !this.particleField) {
      throw new Error('[MLPOrchestrator] model and particleField are required.');
    }

    await this.model.init();
    this._tf = this.model.getTF();
    this.attributeHandles = this.particleField.getAttributeHandles();
    this._captureFlickerDefaults();
    this._prepareStaticInputs();
    await this.syncModelDimensions();
    super.init();
    return this;
  }

  dispose() {
    this.baseTensor?.dispose?.();
    this.baseTensor = null;
    this._pending = null;
    super.dispose();
  }

  update(delta = 0) {
    if (!this.initialized || !this.options.enabled) {
      return;
    }
    this.accumulator += Math.max(0, delta);
    if (this.accumulator < this.interval || this._pending) {
      return;
    }
    this.accumulator = 0;
    this._pending = this._computeInference()
      .catch((error) => {
        console.error('[MLPOrchestrator] inference failed', error);
      })
      .finally(() => {
        this._pending = null;
      });
  }

  async runOnce() {
    if (!this.initialized) {
      return;
    }
    await this._computeInference();
  }

  setRate(rateHz = 24) {
    const clamped = clamp(Number(rateHz) || 0, 5, 90);
    this.options.rateHz = clamped;
    this.interval = 1 / clamped;
  }

  getRate() {
    return this.options.rateHz;
  }

  setBlend(value = 1) {
    this.options.blend = clamp(Number(value) || 1, 0, 1);
  }

  getBlend() {
    return this.options.blend;
  }

  getReactivityConfig() {
    return { ...this.options.reactivity };
  }

  updateReactivity(changes = {}) {
    this.options.reactivity = {
      ...this.options.reactivity,
      ...(changes || {}),
    };
  }

  getOutputConfig() {
    return JSON.parse(JSON.stringify(this.options.outputs));
  }

  updateOutput(key, changes = {}) {
    if (!this.options.outputs[key]) {
      return;
    }
    this.options.outputs[key] = {
      ...this.options.outputs[key],
      ...changes,
    };
  }

  getStats() {
    return {
      lastInferenceMs: this.lastInferenceMs,
      count: this.count,
      pending: Boolean(this._pending),
    };
  }

  async syncModelDimensions() {
    const totalInputs = this.baseDims + this.audioDims;
    this.totalInputSize = totalInputs;
    const config = this.model.getConfig();
    if (config.inputSize !== totalInputs) {
      await this.model.rebuild({ inputSize: totalInputs });
    }
    this.outputDims = this.model.getConfig().outputSize;
  }

  refreshParticleState() {
    if (!this.particleField || !this._tf) {
      return;
    }
    this.attributeHandles = this.particleField.getAttributeHandles();
    this._captureFlickerDefaults();
    this._prepareStaticInputs();
  }

  _prepareStaticInputs() {
    const state = this.particleField.getParticleState();
    if (!state) {
      throw new Error('[MLPOrchestrator] ParticleField state is missing.');
    }
    const { positions, distOrigin, idHash, phase } = state;
    this.count = idHash.length;
    const buffer = new Float32Array(this.count * this.baseDims);
    for (let i = 0; i < this.count; i += 1) {
      const baseIndex = i * this.baseDims;
      const posIndex = i * 3;
      const px = positions[posIndex] ?? 0;
      const py = positions[posIndex + 1] ?? 0;
      const pz = positions[posIndex + 2] ?? 0;
      buffer[baseIndex] = px;
      buffer[baseIndex + 1] = py;
      buffer[baseIndex + 2] = pz;
      buffer[baseIndex + 3] = distOrigin[i] ?? Math.hypot(px, py, pz);
      buffer[baseIndex + 4] = idHash[i] ?? 0;
      const phaseValue = phase[i] ?? 0;
      buffer[baseIndex + 5] = Math.sin(phaseValue);
      buffer[baseIndex + 6] = Math.cos(phaseValue);
      buffer[baseIndex + 7] = 0; // prev speed placeholder
    }
    this.baseTensor?.dispose?.();
    this.baseTensor = this._tf.tensor2d(buffer, [this.count, this.baseDims]);
    this.baseBuffer = buffer;
  }

  _captureFlickerDefaults() {
    if (!this.attributeHandles) return;
    const rateHandle = this.attributeHandles.flickerRate;
    const depthHandle = this.attributeHandles.flickerDepth;
    if (rateHandle) {
      this.baseFlickerRate = new Float32Array(rateHandle.array);
    }
    if (depthHandle) {
      this.baseFlickerDepth = new Float32Array(depthHandle.array);
    }
  }

  async _computeInference() {
    if (!this.baseTensor) {
      return;
    }
    const tf = this._tf;
    const features = this._buildFeatureVector();
    const featureTensor = tf.tensor2d(features, [1, this.audioDims]);
    const tiled = featureTensor.tile([this.count, 1]);
    const inputTensor = tf.concat([this.baseTensor, tiled], 1);
    const start = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const outputTensor = this.model.predict(inputTensor);
    const data = await outputTensor.data();
    this.lastInferenceMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - start;
    const modifiers = this._computeReactivityModifiers();
    this._applyOutputs(data, modifiers);
    featureTensor.dispose();
    tiled.dispose();
    inputTensor.dispose();
    outputTensor.dispose();
  }

  _buildFeatureVector() {
    const source = (this.featureExtractor && this.featureExtractor.getFeatures?.()) || {};
    for (let i = 0; i < this.audioDims; i += 1) {
      const key = FEATURE_KEYS[i];
      const value = source[key];
      this.featureVector[i] = Number.isFinite(value) ? value : 0;
      this.lastFeatures[key] = this.featureVector[i];
    }
    return this.featureVector;
  }

  _computeReactivityModifiers() {
    const result = deriveReactivity(this.lastFeatures, this._reactivityState, this.options.reactivity, this.getBlend());
    this._reactivityState.envelope = result.envelope;
    this._reactivityState.prevPeak = result.prevPeak;
    return result;
  }

  _applyOutputs(buffer, modifiers = null) {
    if (!buffer || !buffer.length || !this.attributeHandles) {
      return;
    }
    const {
      deltaPos,
      colorDelta,
      sizeDelta,
      flickerRate,
      flickerDepth,
    } = this.attributeHandles;
    const gain = clamp(modifiers?.gain ?? 1, 0.1, 4);
    const flickerBoost = clamp(modifiers?.flickerBoost ?? 1, 0.5, 4);
    const blend = clamp(modifiers?.blend ?? this.options.blend, 0, 1);
    const outputs = this.options.outputs;
    const activeGlobalKeys = GLOBAL_OUTPUT_KEYS.filter(
      (key) => typeof OUTPUT_INDEX[key] === 'number' && OUTPUT_INDEX[key] < this.outputDims,
    );
    const globalAccum = activeGlobalKeys.reduce((acc, key) => {
      acc[key] = 0;
      return acc;
    }, {});

    for (let i = 0; i < this.count; i += 1) {
      const baseIndex = i * this.outputDims;
      if (deltaPos) {
        const stride = i * 3;
        const scaleX = outputs.deltaPos.x;
        const scaleY = outputs.deltaPos.y;
        const scaleZ = outputs.deltaPos.z;
        const xRange = scaleX * gain;
        const yRange = scaleY * gain;
        const zRange = scaleZ * gain;
        deltaPos.array[stride] = lerp(
          deltaPos.array[stride],
          clamp(buffer[baseIndex] * xRange, -xRange, xRange),
          blend,
        );
        deltaPos.array[stride + 1] = lerp(
          deltaPos.array[stride + 1],
          clamp(buffer[baseIndex + 1] * yRange, -yRange, yRange),
          blend,
        );
        deltaPos.array[stride + 2] = lerp(
          deltaPos.array[stride + 2],
          clamp(buffer[baseIndex + 2] * zRange, -zRange, zRange),
          blend,
        );
      }

      if (sizeDelta) {
        const sizeRange = outputs.size;
        const mapped = mapMinusOneToOne(
          buffer[baseIndex + 3],
          sizeRange.min * gain,
          sizeRange.max * gain,
        );
        sizeDelta.array[i] = lerp(sizeDelta.array[i], mapped, blend);
      }

      if (colorDelta) {
        const colorRange = outputs.color;
        const cIdx = i * 3;
        const r = mapMinusOneToOne(buffer[baseIndex + 4], colorRange.min, colorRange.max);
        const g = mapMinusOneToOne(buffer[baseIndex + 5], colorRange.min, colorRange.max);
        const b = mapMinusOneToOne(buffer[baseIndex + 6], colorRange.min, colorRange.max);
        const rScaled = r * gain;
        const gScaled = g * gain;
        const bScaled = b * gain;
        colorDelta.array[cIdx] = lerp(colorDelta.array[cIdx], rScaled, blend);
        colorDelta.array[cIdx + 1] = lerp(colorDelta.array[cIdx + 1], gScaled, blend);
        colorDelta.array[cIdx + 2] = lerp(colorDelta.array[cIdx + 2], bScaled, blend);
      }

      if (flickerRate) {
        const range = outputs.flickerRate;
        const mapped = mapMinusOneToOne(
          buffer[baseIndex + 7],
          range.min,
          range.max * flickerBoost,
        );
        const base = this.baseFlickerRate ? this.baseFlickerRate[i] : 1;
        const value = lerp(base, mapped, blend);
        flickerRate.array[i] = value;
      }

      if (flickerDepth) {
        const range = outputs.flickerDepth;
        const mapped = mapMinusOneToOne(
          buffer[baseIndex + 8],
          range.min,
          range.max * flickerBoost,
        );
        const base = this.baseFlickerDepth ? this.baseFlickerDepth[i] : 0.2;
        flickerDepth.array[i] = lerp(base, mapped, blend);
      }

      activeGlobalKeys.forEach((key) => {
        const index = OUTPUT_INDEX[key];
        if (typeof index === 'number') {
          globalAccum[key] += buffer[baseIndex + index];
        }
      });
    }

    deltaPos?.markNeedsUpdate();
    sizeDelta?.markNeedsUpdate();
    colorDelta?.markNeedsUpdate();
    flickerRate?.markNeedsUpdate();
    flickerDepth?.markNeedsUpdate();
    if (activeGlobalKeys.length) {
      this._applyGlobalOutputs(globalAccum, blend);
    }
  }

  getBaseSamples(sampleCount = 512) {
    if (!this.baseBuffer || !this.baseDims || !this.count) {
      return null;
    }
    const total = this.count;
    const clamped = Math.max(1, Math.min(sampleCount, total));
    const indices = new Set();
    while (indices.size < clamped) {
      indices.add(Math.floor(Math.random() * total));
    }
    const result = new Float32Array(clamped * this.baseDims);
    let offset = 0;
    indices.forEach((index) => {
      const src = index * this.baseDims;
      for (let i = 0; i < this.baseDims; i += 1) {
        result[offset + i] = this.baseBuffer[src + i];
      }
      offset += this.baseDims;
    });
    return {
      buffer: result,
      count: clamped,
      dims: this.baseDims,
    };
  }

  _applyGlobalOutputs(accum = {}, blend = 1) {
    if (!this.particleField || !accum) {
      return;
    }
    const outputs = this.options.outputs;
    const divisor = Math.max(1, this.count);
    GLOBAL_OUTPUT_KEYS.forEach((key) => {
      const index = OUTPUT_INDEX[key];
      if (typeof index !== 'number' || index >= this.outputDims) {
        return;
      }
      const range = outputs[key];
      if (!range) {
        return;
      }
      const sum = accum[key];
      if (!Number.isFinite(sum)) {
        return;
      }
      const normalized = clamp(sum / divisor, -1, 1);
      const mapped = mapMinusOneToOne(normalized, range.min, range.max);
      const current = this._readGlobalValue(key);
      const next = lerp(current, mapped, blend);
      this._writeGlobalValue(key, next);
    });
  }

  _readGlobalValue(key) {
    const field = this.particleField;
    if (!field?.options) {
      return 0;
    }
    return field.options[key] ?? 0;
  }

  _writeGlobalValue(key, value) {
    const field = this.particleField;
    if (!field) {
      return;
    }
    const method = GLOBAL_OUTPUT_SETTERS[key];
    if (method && typeof field[method] === 'function') {
      field[method](value);
    } else if (field.options) {
      field.options[key] = value;
    }
  }
}
