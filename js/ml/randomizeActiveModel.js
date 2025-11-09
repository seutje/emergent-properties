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
 * @param {Object} [options.snapshot] Optional snapshot to import instead of rebuilding.
 * @param {string} [options.snapshotUrl] Optional source hint for UI notifications.
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
  snapshot = null,
  snapshotUrl = '',
} = {}) {
  if (!mlpModel) {
    throw new Error('[randomizeActiveModel] mlpModel is required.');
  }
  if (!mlpController) {
    throw new Error('[randomizeActiveModel] mlpController is required.');
  }

  let nextSeed = Number.isFinite(seed) ? seed : null;
  let snapshotLabel = null;

  if (snapshot) {
    const metadata = await mlpModel.importSnapshot(snapshot);
    snapshotLabel = metadata?.label || snapshot?.metadata?.label || null;
    if (!Number.isFinite(nextSeed)) {
      const configSeed = Number(snapshot?.config?.seed);
      const metadataSeed = Number(metadata?.seed);
      if (Number.isFinite(configSeed)) {
        nextSeed = configSeed;
      } else if (Number.isFinite(metadataSeed)) {
        nextSeed = metadataSeed;
      }
    }
  } else {
    nextSeed = Number.isFinite(nextSeed) ? nextSeed : Date.now();
    await mlpModel.rebuild({ seed: nextSeed });
  }

  if (!Number.isFinite(nextSeed)) {
    nextSeed = Date.now();
  }

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

  const notification = {
    seed: nextSeed,
    reason,
    track,
  };
  if (snapshotUrl) {
    notification.snapshotUrl = snapshotUrl;
  }
  if (snapshotLabel) {
    notification.snapshotLabel = snapshotLabel;
  }

  uiController?.notifyModelRandomized?.(notification);

  return nextSeed;
}
