import { PARTICLE_PARAMETER_TARGETS, getParticleParameterTarget } from './MLPTrainingTargets.js';
import { CORRELATION_FEATURE_SOURCES, getParticlePositionalFeature } from './MLPTrainingFeatures.js';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const clamp01 = (value) => clamp(value, 0, 1);

const createRng = (seed = Date.now()) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

export const DEFAULT_CORRELATION = (featureKey, targetId) => ({
  id: `${featureKey}-${targetId}`.toLowerCase(),
  featureKey,
  targetId,
  strength: 0.65,
  polarity: 'direct',
});

const buildPositionalFeatureMap = (list = []) => {
  const map = new Map();
  (Array.isArray(list) ? list : []).forEach((feature) => {
    if (feature?.id != null && Number.isFinite(feature.baseIndex)) {
      map.set(feature.id, {
        id: feature.id,
        label: feature.label,
        baseIndex: feature.baseIndex,
      });
    }
  });
  return map;
};

const normalizeBaseFeatureValue = (value, index, stats = null) => {
  if (!stats || !stats[index]) {
    return clamp(value, -1, 1);
  }
  const { min, max } = stats[index];
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min < 1e-5) {
    return clamp(value, -1, 1);
  }
  const normalized = (value - min) / (max - min);
  return clamp(normalized * 2 - 1, -1, 1);
};

const computeBaseStats = (buffer, count, dims) => {
  if (!buffer || !count || !dims) {
    return null;
  }
  const stats = Array.from({ length: dims }, () => ({ min: Infinity, max: -Infinity }));
  for (let row = 0; row < count; row += 1) {
    const baseOffset = row * dims;
    for (let dim = 0; dim < dims; dim += 1) {
      const value = buffer[baseOffset + dim];
      const entry = stats[dim];
      if (value < entry.min) {
        entry.min = value;
      }
      if (value > entry.max) {
        entry.max = value;
      }
    }
  }
  return stats;
};

export function sanitizeCorrelation(entry = {}, featureKeys = [], options = {}) {
  const features = Array.isArray(featureKeys) && featureKeys.length ? featureKeys : ['rms'];
  const positionalFeatures = Array.isArray(options.positionalFeatures) ? options.positionalFeatures : [];
  const positionalMap = buildPositionalFeatureMap(positionalFeatures);
  const positionalKeys = Array.from(positionalMap.keys());
  const allKeys = [...features, ...positionalKeys];
  const fallbackFeature = allKeys[0] || features[0];
  const featureKey = allKeys.includes(entry.featureKey) ? entry.featureKey : fallbackFeature;
  const featureIndex = Math.max(0, features.indexOf(featureKey));
  const target = getParticleParameterTarget(entry.targetId) || PARTICLE_PARAMETER_TARGETS[0];
  const strength = clamp01(Number(entry.strength) || 0);
  const polarity = entry.polarity === 'inverse' ? 'inverse' : 'direct';
  const id = entry.id || `${featureKey}-${target.id}`;
  const positionalMeta = positionalMap.get(featureKey) || getParticlePositionalFeature(featureKey);
  const source = positionalMeta ? CORRELATION_FEATURE_SOURCES.PARTICLE : CORRELATION_FEATURE_SOURCES.AUDIO;
  return {
    id,
    featureKey,
    targetId: target.id,
    targetIndex: target.outputIndex,
    axis: target.axis || null,
    group: target.group,
    featureIndex: source === CORRELATION_FEATURE_SOURCES.AUDIO ? featureIndex : null,
    baseIndex: positionalMeta?.baseIndex ?? null,
    strength,
    polarity,
    source,
  };
}

export function sanitizeCorrelationList(list = [], featureKeys = [], options = {}) {
  const result = [];
  const seen = new Set();
  (Array.isArray(list) ? list : []).forEach((entry) => {
    const sanitized = sanitizeCorrelation(entry, featureKeys, options);
    if (sanitized && !seen.has(sanitized.id)) {
      seen.add(sanitized.id);
      result.push(sanitized);
    }
  });
  return result;
}

export function generateSyntheticDataset({
  correlations = [],
  featureKeys = [],
  positionalFeatures = [],
  baseSampleBuffer = null,
  baseSampleCount = 0,
  baseDims = 0,
  audioDims = 0,
  outputSize = 0,
  sampleCount = 2048,
  noise = 0.05,
  seed = Date.now(),
}) {
  const sanitized = sanitizeCorrelationList(correlations, featureKeys, {
    positionalFeatures,
  });
  const totalInputs = baseDims + audioDims;
  const inputs = new Float32Array(sampleCount * totalInputs);
  const targets = new Float32Array(sampleCount * outputSize);
  const featureBuffer = new Float32Array(sampleCount * audioDims);
  const baseFeatureBuffer = baseDims ? new Float32Array(sampleCount * baseDims) : null;
  const rng = createRng(seed);
  const baseStats = computeBaseStats(baseSampleBuffer, baseSampleCount, baseDims);

  const pickBaseRow = () => {
    if (!baseSampleBuffer || !baseSampleCount || !baseDims) {
      return null;
    }
    const row = Math.floor(rng() * baseSampleCount) % baseSampleCount;
    const offset = row * baseDims;
    return baseSampleBuffer.subarray(offset, offset + baseDims);
  };

  for (let sample = 0; sample < sampleCount; sample += 1) {
    const baseRow = pickBaseRow();
    for (let i = 0; i < baseDims; i += 1) {
      const value = baseRow ? baseRow[i] : rng() * 2 - 1;
      inputs[sample * totalInputs + i] = value;
      if (baseFeatureBuffer) {
        baseFeatureBuffer[sample * baseDims + i] = value;
      }
    }

    for (let f = 0; f < audioDims; f += 1) {
      const value = rng();
      const audioIndex = sample * audioDims + f;
      featureBuffer[audioIndex] = value;
      inputs[sample * totalInputs + baseDims + f] = value;
    }

    for (let out = 0; out < outputSize; out += 1) {
      targets[sample * outputSize + out] = 0;
    }

    sanitized.forEach((corr) => {
      let normalized = 0;
      if (corr.source === CORRELATION_FEATURE_SOURCES.PARTICLE) {
        if (!baseFeatureBuffer || !Number.isInteger(corr.baseIndex)) {
          return;
        }
        const baseIndex = corr.baseIndex;
        if (baseIndex < 0 || baseIndex >= baseDims) {
          return;
        }
        const baseValue = baseFeatureBuffer[sample * baseDims + baseIndex];
        normalized = normalizeBaseFeatureValue(baseValue, baseIndex, baseStats);
      } else {
        const featureIndex = corr.featureIndex ?? featureKeys.indexOf(corr.featureKey);
        if (featureIndex < 0) return;
        const featureValue = featureBuffer[sample * audioDims + featureIndex];
        normalized = (featureValue - 0.5) * 2;
      }
      const signed = corr.polarity === 'inverse' ? -normalized : normalized;
      const contribution = signed * corr.strength;
      const outputIndex = corr.targetIndex;
      const current = targets[sample * outputSize + outputIndex];
      const value = clamp(current + contribution + (rng() - 0.5) * noise, -1, 1);
      targets[sample * outputSize + outputIndex] = value;
    });
  }

  return {
    inputs,
    targets,
    featureBuffer,
    baseFeatureBuffer,
    sampleCount,
    totalInputs,
    audioDims,
    baseDims,
    outputSize,
    correlations: sanitized,
    baseStats,
  };
}

export function computePearsonCorrelation(seriesA = [], seriesB = []) {
  const length = Math.min(seriesA.length, seriesB.length);
  if (!length) {
    return 0;
  }
  let sumA = 0;
  let sumB = 0;
  let sumA2 = 0;
  let sumB2 = 0;
  let sumAB = 0;
  for (let i = 0; i < length; i += 1) {
    const a = seriesA[i];
    const b = seriesB[i];
    sumA += a;
    sumB += b;
    sumA2 += a * a;
    sumB2 += b * b;
    sumAB += a * b;
  }
  const numerator = length * sumAB - sumA * sumB;
  const denominator = Math.sqrt(
    (length * sumA2 - sumA * sumA) * (length * sumB2 - sumB * sumB),
  );
  if (!denominator || !Number.isFinite(denominator)) {
    return 0;
  }
  return clamp(numerator / denominator, -1, 1);
}

export function evaluateCorrelationAchievement({
  featureBuffer,
  baseFeatureBuffer = null,
  predictionBuffer,
  sampleCount,
  audioDims,
  baseDims = 0,
  outputSize,
  featureKeys,
  correlations = [],
  baseStats = null,
}) {
  if (!sampleCount || !audioDims || !outputSize) {
    return [];
  }
  return correlations.map((corr) => {
    const featureSeries = new Array(sampleCount);
    const predictionSeries = new Array(sampleCount);
    if (corr.source === CORRELATION_FEATURE_SOURCES.PARTICLE) {
      if (!baseFeatureBuffer || !Number.isInteger(corr.baseIndex)) {
        return { ...corr, achieved: 0 };
      }
      const baseIndex = corr.baseIndex;
      if (baseIndex < 0 || baseIndex >= baseDims) {
        return { ...corr, achieved: 0 };
      }
      for (let i = 0; i < sampleCount; i += 1) {
        const raw = baseFeatureBuffer[i * baseDims + baseIndex];
        featureSeries[i] = normalizeBaseFeatureValue(raw, baseIndex, baseStats);
        predictionSeries[i] = predictionBuffer[i * outputSize + corr.targetIndex];
      }
    } else {
      const featureIndex = corr.featureIndex ?? featureKeys.indexOf(corr.featureKey);
      if (featureIndex < 0) {
        return { ...corr, achieved: 0 };
      }
      for (let i = 0; i < sampleCount; i += 1) {
        featureSeries[i] = featureBuffer[i * audioDims + featureIndex];
        predictionSeries[i] = predictionBuffer[i * outputSize + corr.targetIndex];
      }
    }
    const achieved = computePearsonCorrelation(featureSeries, predictionSeries);
    return { ...corr, achieved };
  });
}
