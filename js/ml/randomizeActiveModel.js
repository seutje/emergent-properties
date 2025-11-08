/**
 * Rebuilds the active MLP model with a fresh random seed and keeps the rest
 * of the runtime (orchestrator, training manager, UI) in sync.
 *
 * @param {Object} options
 * @param {MLPModel} options.mlpModel
 * @param {MLPOrchestrator} options.mlpController
 * @param {MLPTrainingManager} [options.trainingManager]
 * @param {UIController} [options.uiController]
 * @param {number} [options.seed]
 * @param {string} [options.reason]
 * @param {Object} [options.track]
 * @returns {Promise<number>} The seed used for randomization.
 */
export async function randomizeActiveModel({
  mlpModel,
  mlpController,
  trainingManager = null,
  uiController = null,
  seed = null,
  reason = 'manual',
  track = null,
} = {}) {
  if (!mlpModel) {
    throw new Error('[randomizeActiveModel] mlpModel is required.');
  }
  if (!mlpController) {
    throw new Error('[randomizeActiveModel] mlpController is required.');
  }

  const nextSeed = Number.isFinite(seed) ? seed : Date.now();
  await mlpModel.rebuild({ seed: nextSeed });

  if (typeof mlpController.syncModelDimensions === 'function') {
    await mlpController.syncModelDimensions();
  }
  if (typeof mlpController.refreshParticleState === 'function') {
    mlpController.refreshParticleState();
  }
  if (typeof mlpController.runOnce === 'function') {
    await mlpController.runOnce();
  }

  if (trainingManager?.updateTrainingOptions) {
    trainingManager.updateTrainingOptions({ seed: nextSeed });
  }

  uiController?.notifyModelRandomized?.({
    seed: nextSeed,
    reason,
    track,
  });

  return nextSeed;
}
