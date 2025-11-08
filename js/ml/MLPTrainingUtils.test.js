import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';
import {
  sanitizeCorrelationList,
  generateSyntheticDataset,
  computePearsonCorrelation,
  evaluateCorrelationAchievement,
  DEFAULT_CORRELATION,
} from './MLPTrainingUtils.js';
import { PARTICLE_POSITIONAL_FEATURES } from './MLPTrainingFeatures.js';

describe('MLPTrainingUtils', () => {
  it('sanitizes correlations and clamps strength', () => {
    const dirty = [
      { featureKey: 'unknown', targetId: 'deltaPosX', strength: 1.5, polarity: 'inverse' },
      { featureKey: FEATURE_KEYS[1], targetId: 'bogus', strength: -0.1 },
    ];
    const sanitized = sanitizeCorrelationList(dirty, FEATURE_KEYS);
    expect(sanitized).toHaveLength(2);
    expect(sanitized[0].featureKey).toBe(FEATURE_KEYS[0]);
    expect(sanitized[0].strength).toBeLessThanOrEqual(1);
    expect(sanitized[1].targetId).toBe(PARTICLE_PARAMETER_TARGETS[0].id);
  });

  it('creates synthetic dataset aligning features with targets', () => {
    const correlation = DEFAULT_CORRELATION(FEATURE_KEYS[0], PARTICLE_PARAMETER_TARGETS[0].id);
    correlation.strength = 0.9;
    const sampleCount = 512;
    const baseDims = 3;
    const audioDims = FEATURE_KEYS.length;
    const outputSize = PARTICLE_PARAMETER_TARGETS.length;
    const dataset = generateSyntheticDataset({
      correlations: [correlation],
      featureKeys: FEATURE_KEYS,
      baseSampleBuffer: new Float32Array(baseDims * sampleCount),
      baseSampleCount: sampleCount,
      baseDims,
      audioDims,
      outputSize,
      sampleCount,
      noise: 0.01,
      seed: 123,
    });

    expect(dataset.inputs.length).toBe(sampleCount * (baseDims + audioDims));
    expect(dataset.targets.length).toBe(sampleCount * outputSize);

    const featureSeries = [];
    const targetSeries = [];
    const [sanitized] = dataset.correlations;
    const featureIndex = sanitized.featureIndex;
    const targetIndex = sanitized.targetIndex;
    for (let i = 0; i < sampleCount; i += 1) {
      featureSeries.push(dataset.featureBuffer[i * audioDims + featureIndex]);
      targetSeries.push(dataset.targets[i * outputSize + targetIndex]);
    }
    const corr = computePearsonCorrelation(featureSeries, targetSeries);
    expect(Math.abs(corr)).toBeGreaterThan(0.6);

    const achieved = evaluateCorrelationAchievement({
      featureBuffer: dataset.featureBuffer,
      predictionBuffer: dataset.targets,
      sampleCount,
      audioDims,
      outputSize,
      featureKeys: FEATURE_KEYS,
      correlations: dataset.correlations,
    });
    expect(achieved[0].achieved).toBeGreaterThan(0.6);
  });

  it('supports positional particle features in correlations', () => {
    const positionalFeature = PARTICLE_POSITIONAL_FEATURES[0];
    const correlation = DEFAULT_CORRELATION(positionalFeature.id, PARTICLE_PARAMETER_TARGETS[1].id);
    correlation.strength = 0.8;
    const sampleCount = 256;
    const baseDims = 4;
    const audioDims = FEATURE_KEYS.length;
    const outputSize = PARTICLE_PARAMETER_TARGETS.length;
    const baseSampleBuffer = new Float32Array(baseDims * sampleCount);
    for (let i = 0; i < sampleCount; i += 1) {
      const offset = i * baseDims;
      baseSampleBuffer[offset] = -1 + (2 * i) / sampleCount;
      baseSampleBuffer[offset + 1] = (i % 16) / 16;
      baseSampleBuffer[offset + 2] = 0.5;
      baseSampleBuffer[offset + 3] = Math.abs(baseSampleBuffer[offset]);
    }
    const dataset = generateSyntheticDataset({
      correlations: [correlation],
      featureKeys: FEATURE_KEYS,
      positionalFeatures: PARTICLE_POSITIONAL_FEATURES,
      baseSampleBuffer,
      baseSampleCount: sampleCount,
      baseDims,
      audioDims,
      outputSize,
      sampleCount,
      noise: 0.02,
      seed: 77,
    });
    const [sanitized] = dataset.correlations;
    expect(sanitized.baseIndex).toBe(positionalFeature.baseIndex);
    const series = [];
    const targets = [];
    for (let i = 0; i < sampleCount; i += 1) {
      series.push(dataset.baseFeatureBuffer[i * baseDims + sanitized.baseIndex]);
      targets.push(dataset.targets[i * outputSize + sanitized.targetIndex]);
    }
    const corr = computePearsonCorrelation(series, targets);
    expect(Math.abs(corr)).toBeGreaterThan(0.5);
    const achieved = evaluateCorrelationAchievement({
      featureBuffer: dataset.featureBuffer,
      baseFeatureBuffer: dataset.baseFeatureBuffer,
      predictionBuffer: dataset.targets,
      sampleCount,
      audioDims,
      baseDims,
      outputSize,
      featureKeys: FEATURE_KEYS,
      correlations: dataset.correlations,
      baseStats: dataset.baseStats,
    });
    expect(achieved[0].featureKey).toBe(positionalFeature.id);
    expect(Math.abs(achieved[0].achieved)).toBeGreaterThan(0.5);
  });
});
