import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_COUNT, PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';

const GLOBAL_CORRELATION_SEEDS = [
  { featureKey: 'tempoProxy', targetId: 'rotationSpeed', strength: 0.65 },
  { featureKey: 'bandLow', targetId: 'wobbleStrength', strength: 0.6 },
  { featureKey: 'bandHigh', targetId: 'wobbleFrequency', strength: 0.6 },
  { featureKey: 'specCentroid', targetId: 'colorMix', strength: 0.55 },
  { featureKey: 'rms', targetId: 'alphaScale', strength: 0.5 },
  { featureKey: 'peak', targetId: 'pointScale', strength: 0.55 },
];

export function upgradeModelSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') {
    return snapshot || null;
  }
  snapshot.config = snapshot.config || {};
  const requiredOutputs = PARTICLE_PARAMETER_COUNT;
  const currentOutputs = Number(snapshot.config.outputSize) || requiredOutputs;
  if (currentOutputs < requiredOutputs) {
    padOutputWeights(snapshot, currentOutputs, requiredOutputs);
    snapshot.config.outputSize = requiredOutputs;
  }
  snapshot.metadata = snapshot.metadata || {};
  snapshot.metadata.correlations = mergeRuntimeCorrelations(snapshot.metadata.correlations);
  if (snapshot.metadata.training?.metadata) {
    snapshot.metadata.training.metadata.correlations = mergeTrainingCorrelations(
      snapshot.metadata.training.metadata.correlations,
    );
  }
  return snapshot;
}

function padOutputWeights(snapshot, oldSize, newSize) {
  const diff = newSize - oldSize;
  if (!Array.isArray(snapshot.weights) || diff <= 0) {
    return;
  }
  snapshot.weights.forEach((entry) => {
    if (!entry || !Array.isArray(entry.shape) || !Array.isArray(entry.values)) {
      return;
    }
    // Bias vector for the output layer.
    if (entry.shape.length === 1 && entry.shape[0] === oldSize) {
      entry.shape[0] = newSize;
      entry.values = entry.values.concat(new Array(diff).fill(0));
      return;
    }
    // Dense kernel where the output dimension is the last axis.
    if (entry.shape.length === 2 && entry.shape[1] === oldSize) {
      const rows = entry.shape[0];
      const newValues = new Array(rows * newSize).fill(0);
      for (let row = 0; row < rows; row += 1) {
        const oldOffset = row * oldSize;
        const newOffset = row * newSize;
        for (let col = 0; col < oldSize; col += 1) {
          newValues[newOffset + col] = entry.values[oldOffset + col];
        }
      }
      entry.shape[1] = newSize;
      entry.values = newValues;
    }
  });
}

function mergeRuntimeCorrelations(list = []) {
  const result = Array.isArray(list) ? list.map((entry) => ({ ...entry })) : [];
  const seen = new Set(result.map((entry) => entry.targetId));
  GLOBAL_CORRELATION_SEEDS.forEach((seed) => {
    if (seen.has(seed.targetId)) {
      return;
    }
    const target = PARTICLE_PARAMETER_TARGETS.find((item) => item.id === seed.targetId);
    if (!target) {
      return;
    }
    result.push({
      id: seed.id || `${seed.featureKey}-${seed.targetId}`.toLowerCase(),
      featureKey: seed.featureKey,
      targetId: seed.targetId,
      targetIndex: target.outputIndex,
      axis: target.axis || null,
      group: target.group,
      featureIndex: Math.max(0, FEATURE_KEYS.indexOf(seed.featureKey)),
      strength: seed.strength ?? 0.6,
      polarity: seed.polarity || 'direct',
    });
    seen.add(seed.targetId);
  });
  return result;
}

function mergeTrainingCorrelations(list = []) {
  const result = Array.isArray(list) ? list.map((entry) => ({ ...entry })) : [];
  const seen = new Set(result.map((entry) => entry.targetId));
  GLOBAL_CORRELATION_SEEDS.forEach((seed) => {
    if (seen.has(seed.targetId)) {
      return;
    }
    result.push({
      id: seed.id || `${seed.featureKey}-${seed.targetId}`.toLowerCase(),
      featureKey: seed.featureKey,
      targetId: seed.targetId,
      strength: seed.strength ?? 0.6,
      polarity: seed.polarity || 'direct',
      achieved: 0,
    });
    seen.add(seed.targetId);
  });
  return result;
}
