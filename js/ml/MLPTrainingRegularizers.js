const clampPositive = (value, fallback = 0) => {
  if (!Number.isFinite(value)) {
    return Math.max(0, fallback);
  }
  return Math.max(0, value);
};

export function resolveTemporalNorm(mode = 'l2') {
  return mode === 'l1' ? 'l1' : 'l2';
}

export function computeCurriculumScale(epoch = 0, warmupEpochs = 0, boost = 1) {
  const warmup = Math.max(0, Math.floor(warmupEpochs));
  const targetBoost = Math.max(1, Number.isFinite(boost) ? boost : 1);
  if (!warmup) {
    return 1;
  }
  if (epoch >= warmup) {
    return 1;
  }
  const phase = Math.max(0, warmup - epoch) / warmup;
  return 1 + (targetBoost - 1) * phase;
}

export function buildRegularizerProfile(training = {}, epoch = 0) {
  const scale = computeCurriculumScale(
    epoch,
    training.curriculumWarmupEpochs,
    training.curriculumBoost,
  );
  const temporalWeight = clampPositive(training.temporalSmoothnessWeight) * scale;
  const slewWeight = clampPositive(training.slewRateWeight) * scale;
  const noiseStd = clampPositive(training.inputNoiseStd ?? training.jacobianNoise ?? 0);
  const jacobianNoise = clampPositive(training.jacobianNoise ?? noiseStd);
  return {
    curriculumScale: scale,
    temporalNorm: resolveTemporalNorm(training.temporalSmoothnessNorm),
    temporalWeight,
    slewWeight,
    slewThreshold: clampPositive(training.slewRateThreshold, 0),
    jacobianWeight: clampPositive(training.jacobianWeight),
    noiseConsistencyWeight: clampPositive(training.noiseConsistencyWeight),
    noiseStd: Math.max(noiseStd, jacobianNoise),
    weightDecay: clampPositive(training.weightDecay),
  };
}

export function sanitizeBatchSize(value, fallback = 32) {
  if (!Number.isFinite(value) || value <= 0) {
    return Math.max(1, Math.floor(fallback));
  }
  return Math.max(1, Math.floor(value));
}
