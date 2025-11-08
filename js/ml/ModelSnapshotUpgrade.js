import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_COUNT, PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';

const BASE_PARTICLE_DIMS = 8;
const REQUIRED_INPUT_SIZE = BASE_PARTICLE_DIMS + FEATURE_KEYS.length;

const FEATURE_KEY_MIGRATIONS = {
  bandLow: 'bandBass',
  bandMid: 'bandMid',
  bandHigh: 'bandHigh',
};

const GLOBAL_CORRELATION_SEEDS = [
  { featureKey: 'tempoProxy', targetId: 'rotationSpeed', strength: 0.65 },
  { featureKey: 'bandBass', targetId: 'wobbleStrength', strength: 0.6 },
  { featureKey: 'bandHigh', targetId: 'wobbleFrequency', strength: 0.6 },
  { featureKey: 'specCentroid', targetId: 'colorMix', strength: 0.55 },
  { featureKey: 'rms', targetId: 'alphaScale', strength: 0.5 },
  { featureKey: 'peak', targetId: 'pointScale', strength: 0.55 },
  { featureKey: 'bandLowMid', targetId: 'cameraZoom', strength: 0.6 },
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
  const currentInputs = Number(snapshot.config.inputSize) || REQUIRED_INPUT_SIZE;
  if (currentInputs < REQUIRED_INPUT_SIZE) {
    padInputWeights(snapshot, currentInputs, REQUIRED_INPUT_SIZE);
    snapshot.config.inputSize = REQUIRED_INPUT_SIZE;
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

function padInputWeights(snapshot, oldSize, newSize) {
  const diff = newSize - oldSize;
  if (!Array.isArray(snapshot.weights) || diff <= 0) {
    return;
  }
  snapshot.weights.forEach((entry) => {
    if (!entry || !Array.isArray(entry.shape) || !Array.isArray(entry.values)) {
      return;
    }
    if (entry.shape.length === 2 && entry.shape[0] === oldSize) {
      const cols = entry.shape[1];
      const next = new Array(newSize * cols).fill(0);
      for (let row = 0; row < oldSize; row += 1) {
        const srcOffset = row * cols;
        const dstOffset = row * cols;
        for (let col = 0; col < cols; col += 1) {
          next[dstOffset + col] = entry.values[srcOffset + col];
        }
      }
      entry.shape[0] = newSize;
      entry.values = next;
    }
  });
}

const normalizeFeatureKey = (key) => {
  if (!key || typeof key !== 'string') {
    return null;
  }
  const mapped = FEATURE_KEY_MIGRATIONS[key] || key;
  return FEATURE_KEYS.includes(mapped) ? mapped : null;
};

const mapCorrelationFeatureKey = (entry, { includeIndex = false } = {}) => {
  if (!entry || typeof entry !== 'object') {
    return null;
  }
  const featureKey = normalizeFeatureKey(entry.featureKey);
  if (!featureKey) {
    return null;
  }
  const next = { ...entry, featureKey };
  if (includeIndex) {
    next.featureIndex = Math.max(0, FEATURE_KEYS.indexOf(featureKey));
  }
  return next;
};

function mergeRuntimeCorrelations(list = []) {
  const result = [];
  const seen = new Set();
  (Array.isArray(list) ? list : []).forEach((entry) => {
    const normalized = mapCorrelationFeatureKey(entry, { includeIndex: true });
    if (!normalized) {
      return;
    }
    result.push(normalized);
    seen.add(normalized.targetId);
  });
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
  const result = [];
  const seen = new Set();
  (Array.isArray(list) ? list : []).forEach((entry) => {
    const normalized = mapCorrelationFeatureKey(entry);
    if (!normalized) {
      return;
    }
    result.push(normalized);
    seen.add(normalized.targetId);
  });
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
