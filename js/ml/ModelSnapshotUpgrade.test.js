import { describe, expect, it } from '@jest/globals';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { upgradeModelSnapshot } from './ModelSnapshotUpgrade.js';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';

describe('upgradeModelSnapshot', () => {
  it('pads output weights and injects missing global correlations', () => {
    const oldOutputSize = 9;
    const kernel = {
      name: 'dense_Dense2/kernel',
      shape: [32, oldOutputSize],
      dtype: 'float32',
      values: Array.from({ length: 32 * oldOutputSize }, (_, index) => index + 1),
    };
    const bias = {
      name: 'dense_Dense2/bias',
      shape: [oldOutputSize],
      dtype: 'float32',
      values: Array.from({ length: oldOutputSize }, (_, index) => index + 1),
    };

    const snapshot = {
      config: { inputSize: 17, outputSize: oldOutputSize },
      metadata: {
        correlations: [],
        training: {
          metadata: { correlations: [] },
        },
      },
      weights: [kernel, bias],
    };

    const upgraded = upgradeModelSnapshot(snapshot);
    const expectedSize = PARTICLE_PARAMETER_TARGETS.length;
    expect(upgraded.config.outputSize).toBe(expectedSize);
    expect(kernel.shape[1]).toBe(expectedSize);
    expect(kernel.values.length).toBe(32 * expectedSize);
    const firstRow = kernel.values.slice(0, expectedSize);
    expect(firstRow.slice(0, oldOutputSize)).toEqual(Array.from({ length: oldOutputSize }, (_, index) => index + 1));
    expect(firstRow.slice(oldOutputSize)).toEqual(new Array(expectedSize - oldOutputSize).fill(0));
    expect(bias.shape[0]).toBe(expectedSize);
    expect(bias.values.length).toBe(expectedSize);
    expect(bias.values.slice(oldOutputSize)).toEqual(new Array(expectedSize - oldOutputSize).fill(0));

    const hasRotationCorrelation = upgraded.metadata.correlations.some(
      (corr) => corr.targetId === 'rotationSpeed',
    );
    expect(hasRotationCorrelation).toBe(true);

    const trainingCorrelations = upgraded.metadata.training.metadata.correlations;
    expect(trainingCorrelations.some((corr) => corr.targetId === 'rotationSpeed')).toBe(true);
    expect(trainingCorrelations.every((corr) => FEATURE_KEYS.includes(corr.featureKey))).toBe(true);
  });
});
