import { BaseModule } from '../core/BaseModule.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';

const DEFAULT_OUTPUTS = {
  deltaPos: { x: 0.45, y: 0.65, z: 0.45 },
  size: { min: -2.5, max: 2.5 },
  color: { min: -0.35, max: 0.35 },
  flickerRate: { min: 0.25, max: 3.5 },
  flickerDepth: { min: 0.05, max: 0.6 },
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (a, b, t) => a + (b - a) * clamp(t, 0, 1);
const mapMinusOneToOne = (value, min, max) => {
  const normalized = (clamp(value, -1, 1) + 1) * 0.5;
  return lerp(min, max, normalized);
};

export class MLPOrchestrator extends BaseModule {
  constructor({ model, particleField, featureExtractor, options = {} } = {}) {
    super('MLPOrchestrator');
    this.model = model;
    this.particleField = particleField;
    this.featureExtractor = featureExtractor;
    this.options = {
      enabled: true,
      rateHz: 20,
      blend: 0.9,
      outputs: { ...DEFAULT_OUTPUTS, ...(options.outputs || {}) },
      ...options,
    };

    this.count = 0;
    this.baseDims = 8;
    this.audioDims = FEATURE_KEYS.length;
    this.totalInputSize = this.baseDims + this.audioDims;
    this.outputDims = 9;
    this.interval = 1 / this.options.rateHz;
    this.accumulator = 0;
    this._pending = null;
    this._tf = null;
    this.baseTensor = null;
    this.featureVector = new Float32Array(this.audioDims);
    this.attributeHandles = null;
    this.lastInferenceMs = 0;
    this.baseFlickerRate = null;
    this.baseFlickerDepth = null;
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

  setRate(rateHz = 20) {
    const clamped = clamp(Number(rateHz) || 0, 5, 90);
    this.options.rateHz = clamped;
    this.interval = 1 / clamped;
  }

  getRate() {
    return this.options.rateHz;
  }

  setBlend(value = 0.9) {
    this.options.blend = clamp(Number(value) || 0.9, 0, 1);
  }

  getBlend() {
    return this.options.blend;
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
    this._applyOutputs(data);
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
    }
    return this.featureVector;
  }

  _applyOutputs(buffer) {
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
    const blend = clamp(this.options.blend, 0, 1);
    const outputs = this.options.outputs;

    for (let i = 0; i < this.count; i += 1) {
      const baseIndex = i * this.outputDims;
      if (deltaPos) {
        const stride = i * 3;
        const scaleX = outputs.deltaPos.x;
        const scaleY = outputs.deltaPos.y;
        const scaleZ = outputs.deltaPos.z;
        deltaPos.array[stride] = lerp(
          deltaPos.array[stride],
          clamp(buffer[baseIndex] * scaleX, -scaleX, scaleX),
          blend,
        );
        deltaPos.array[stride + 1] = lerp(
          deltaPos.array[stride + 1],
          clamp(buffer[baseIndex + 1] * scaleY, -scaleY, scaleY),
          blend,
        );
        deltaPos.array[stride + 2] = lerp(
          deltaPos.array[stride + 2],
          clamp(buffer[baseIndex + 2] * scaleZ, -scaleZ, scaleZ),
          blend,
        );
      }

      if (sizeDelta) {
        const sizeRange = outputs.size;
        const mapped = mapMinusOneToOne(buffer[baseIndex + 3], sizeRange.min, sizeRange.max);
        sizeDelta.array[i] = lerp(sizeDelta.array[i], mapped, blend);
      }

      if (colorDelta) {
        const colorRange = outputs.color;
        const cIdx = i * 3;
        const r = mapMinusOneToOne(buffer[baseIndex + 4], colorRange.min, colorRange.max);
        const g = mapMinusOneToOne(buffer[baseIndex + 5], colorRange.min, colorRange.max);
        const b = mapMinusOneToOne(buffer[baseIndex + 6], colorRange.min, colorRange.max);
        colorDelta.array[cIdx] = lerp(colorDelta.array[cIdx], r, blend);
        colorDelta.array[cIdx + 1] = lerp(colorDelta.array[cIdx + 1], g, blend);
        colorDelta.array[cIdx + 2] = lerp(colorDelta.array[cIdx + 2], b, blend);
      }

      if (flickerRate) {
        const range = outputs.flickerRate;
        const mapped = mapMinusOneToOne(buffer[baseIndex + 7], range.min, range.max);
        const base = this.baseFlickerRate ? this.baseFlickerRate[i] : 1;
        const value = lerp(base, mapped, blend);
        flickerRate.array[i] = value;
      }

      if (flickerDepth) {
        const range = outputs.flickerDepth;
        const mapped = mapMinusOneToOne(buffer[baseIndex + 8], range.min, range.max);
        const base = this.baseFlickerDepth ? this.baseFlickerDepth[i] : 0.2;
        flickerDepth.array[i] = lerp(base, mapped, blend);
      }
    }

    deltaPos?.markNeedsUpdate();
    sizeDelta?.markNeedsUpdate();
    colorDelta?.markNeedsUpdate();
    flickerRate?.markNeedsUpdate();
    flickerDepth?.markNeedsUpdate();
  }
}
