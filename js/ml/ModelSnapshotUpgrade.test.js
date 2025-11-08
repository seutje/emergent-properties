import { describe, expect, it } from '@jest/globals';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { upgradeModelSnapshot } from './ModelSnapshotUpgrade.js';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';

describe('upgradeModelSnapshot', () => {
  it('pads output weights and injects missing global correlations', () => {
    const oldOutputSize = 9;
    const oldInputSize = 12;
    const inputKernel = {
      name: 'dense/kernel',
      shape: [oldInputSize, 32],
      dtype: 'float32',
      values: Array.from({ length: oldInputSize * 32 }, (_, index) => index + 1),
    };
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
      config: { inputSize: oldInputSize, outputSize: oldOutputSize },
      metadata: {
        correlations: [
          {
            id: 'legacy-bandlow',
            featureKey: 'bandLow',
            targetId: 'wobbleStrength',
            targetIndex: 10,
            axis: null,
            group: 'wobbleStrength',
            featureIndex: 3,
            strength: 0.5,
            polarity: 'direct',
          },
        ],
        training: {
          metadata: {
            correlations: [
              {
                id: 'legacy-bandmid',
                featureKey: 'bandMid',
                targetId: 'cameraZoom',
                strength: 0.5,
                polarity: 'direct',
                achieved: 0.8,
              },
            ],
          },
        },
      },
      weights: [inputKernel, kernel, bias],
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

    const expectedInput = 8 + FEATURE_KEYS.length;
    expect(upgraded.config.inputSize).toBe(expectedInput);
    expect(inputKernel.shape[0]).toBe(expectedInput);
    expect(inputKernel.values.length).toBe(expectedInput * 32);
    const preservedRow = inputKernel.values.slice(0, 32);
    expect(preservedRow).toEqual(Array.from({ length: 32 }, (_, index) => index + 1));
    const paddedSection = inputKernel.values.slice(oldInputSize * 32);
    expect(paddedSection).toEqual(new Array((expectedInput - oldInputSize) * 32).fill(0));

    const hasRotationCorrelation = upgraded.metadata.correlations.some(
      (corr) => corr.targetId === 'rotationSpeed',
    );
    const hasCameraZoomCorrelation = upgraded.metadata.correlations.some(
      (corr) => corr.targetId === 'cameraZoom',
    );
    const hasWobbleStrengthMigration = upgraded.metadata.correlations.some(
      (corr) => corr.targetId === 'wobbleStrength' && corr.featureKey === 'bandBass',
    );
    expect(hasRotationCorrelation).toBe(true);
    expect(hasCameraZoomCorrelation).toBe(true);
    expect(hasWobbleStrengthMigration).toBe(true);

    const trainingCorrelations = upgraded.metadata.training.metadata.correlations;
    expect(trainingCorrelations.some((corr) => corr.targetId === 'rotationSpeed')).toBe(true);
    expect(trainingCorrelations.some((corr) => corr.targetId === 'cameraZoom')).toBe(true);
    expect(trainingCorrelations.every((corr) => FEATURE_KEYS.includes(corr.featureKey))).toBe(true);
  });
});
