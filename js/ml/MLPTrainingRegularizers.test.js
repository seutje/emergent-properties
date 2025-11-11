import {
  computeCurriculumScale,
  buildRegularizerProfile,
  resolveTemporalNorm,
  sanitizeBatchSize,
} from './MLPTrainingRegularizers.js';

describe('MLPTrainingRegularizers', () => {
  it('computes curriculum scale that ramps down smoothly', () => {
    expect(computeCurriculumScale(0, 4, 2)).toBeCloseTo(2);
    expect(computeCurriculumScale(2, 4, 2)).toBeCloseTo(1.5);
    expect(computeCurriculumScale(4, 4, 2)).toBeCloseTo(1);
  });

  it('builds a clamped regularizer profile with sensible defaults', () => {
    const profile = buildRegularizerProfile({
      temporalSmoothnessWeight: 0.05,
      temporalSmoothnessNorm: 'l1',
      slewRateWeight: 0.1,
      slewRateThreshold: -0.2,
      jacobianWeight: 0.25,
      inputNoiseStd: 0.01,
      noiseConsistencyWeight: 0.5,
      weightDecay: 0.0005,
      curriculumWarmupEpochs: 5,
      curriculumBoost: 1.5,
    }, 2);

    expect(profile.temporalNorm).toBe('l1');
    expect(profile.temporalWeight).toBeGreaterThan(0);
    expect(profile.slewThreshold).toBe(0);
    expect(profile.jacobianWeight).toBeCloseTo(0.25);
    expect(profile.noiseStd).toBeGreaterThan(0);
    expect(profile.weightDecay).toBeCloseTo(0.0005);

    const clamped = buildRegularizerProfile({ temporalSmoothnessWeight: -1 }, 0);
    expect(clamped.temporalWeight).toBe(0);
  });

  it('resolves fallback values for invalid inputs', () => {
    expect(resolveTemporalNorm('unsupported')).toBe('l2');
    expect(sanitizeBatchSize(-5, 16)).toBe(16);
  });
});
