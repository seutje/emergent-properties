import { describe, expect, it, jest } from '@jest/globals';
import { randomizeActiveModel } from './randomizeActiveModel.js';

describe('randomizeActiveModel', () => {
  it('rebuilds the model, syncs the orchestrator, and notifies dependents', async () => {
    const mlpModel = { rebuild: jest.fn().mockResolvedValue(null) };
    const mlpController = {
      syncModelDimensions: jest.fn().mockResolvedValue(null),
      refreshParticleState: jest.fn(),
      runOnce: jest.fn().mockResolvedValue(null),
    };
    const trainingManager = { updateTrainingOptions: jest.fn() };
    const uiController = { notifyModelRandomized: jest.fn() };

    const seed = await randomizeActiveModel({
      mlpModel,
      mlpController,
      trainingManager,
      uiController,
      seed: 42,
      reason: 'test',
      track: { id: 'track-01' },
    });

    expect(seed).toBe(42);
    expect(mlpModel.rebuild).toHaveBeenCalledWith(expect.objectContaining({ seed: 42 }));
    expect(mlpController.syncModelDimensions).toHaveBeenCalledTimes(1);
    expect(mlpController.refreshParticleState).toHaveBeenCalledTimes(1);
    expect(mlpController.runOnce).toHaveBeenCalledTimes(1);
    expect(trainingManager.updateTrainingOptions).toHaveBeenCalledWith({ seed: 42 });
    expect(uiController.notifyModelRandomized).toHaveBeenCalledWith({
      seed: 42,
      reason: 'test',
      track: { id: 'track-01' },
    });
  });

  it('throws when required dependencies are missing', async () => {
    await expect(randomizeActiveModel()).rejects.toThrow(/mlpModel/i);
    await expect(randomizeActiveModel({ mlpModel: {} })).rejects.toThrow(/mlpController/i);
  });
});
