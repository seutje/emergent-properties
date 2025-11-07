import { PARTICLE_PARAMETER_TARGETS, getParticleParameterTarget } from './MLPTrainingTargets.js';

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

export function sanitizeCorrelation(entry = {}, featureKeys = []) {
  const features = Array.isArray(featureKeys) && featureKeys.length ? featureKeys : ['rms'];
  const fallbackFeature = features[0];
  const featureKey = features.includes(entry.featureKey) ? entry.featureKey : fallbackFeature;
  const featureIndex = Math.max(0, features.indexOf(featureKey));
  const target = getParticleParameterTarget(entry.targetId) || PARTICLE_PARAMETER_TARGETS[0];
  const strength = clamp01(Number(entry.strength) || 0);
  const polarity = entry.polarity === 'inverse' ? 'inverse' : 'direct';
  const id = entry.id || `${featureKey}-${target.id}`;
  return {
    id,
    featureKey,
    targetId: target.id,
    targetIndex: target.outputIndex,
    axis: target.axis || null,
    group: target.group,
    featureIndex,
    strength,
    polarity,
  };
}

export function sanitizeCorrelationList(list = [], featureKeys = []) {
  const result = [];
  const seen = new Set();
  (Array.isArray(list) ? list : []).forEach((entry) => {
    const sanitized = sanitizeCorrelation(entry, featureKeys);
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
  baseSampleBuffer = null,
  baseSampleCount = 0,
  baseDims = 0,
  audioDims = 0,
  outputSize = 0,
  sampleCount = 2048,
  noise = 0.05,
  seed = Date.now(),
}) {
  const sanitized = sanitizeCorrelationList(correlations, featureKeys);
  const totalInputs = baseDims + audioDims;
  const inputs = new Float32Array(sampleCount * totalInputs);
  const targets = new Float32Array(sampleCount * outputSize);
  const featureBuffer = new Float32Array(sampleCount * audioDims);
  const rng = createRng(seed);

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
      const featureIndex = corr.featureIndex ?? featureKeys.indexOf(corr.featureKey);
      if (featureIndex < 0) return;
      const featureValue = featureBuffer[sample * audioDims + featureIndex];
      const normalized = (featureValue - 0.5) * 2;
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
    sampleCount,
    totalInputs,
    audioDims,
    baseDims,
    outputSize,
    correlations: sanitized,
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
  predictionBuffer,
  sampleCount,
  audioDims,
  outputSize,
  featureKeys,
  correlations = [],
}) {
  if (!sampleCount || !audioDims || !outputSize) {
    return [];
  }
  return correlations.map((corr) => {
    const featureIndex = corr.featureIndex ?? featureKeys.indexOf(corr.featureKey);
    if (featureIndex < 0) {
      return { ...corr, achieved: 0 };
    }
    const featureSeries = new Array(sampleCount);
    const predictionSeries = new Array(sampleCount);
    for (let i = 0; i < sampleCount; i += 1) {
      featureSeries[i] = featureBuffer[i * audioDims + featureIndex];
      predictionSeries[i] = predictionBuffer[i * outputSize + corr.targetIndex];
    }
    const achieved = computePearsonCorrelation(featureSeries, predictionSeries);
    return { ...corr, achieved };
  });
}
