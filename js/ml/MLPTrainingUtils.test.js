import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';
import {
  sanitizeCorrelationList,
  generateSyntheticDataset,
  computePearsonCorrelation,
  evaluateCorrelationAchievement,
  DEFAULT_CORRELATION,
} from './MLPTrainingUtils.js';

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
});
