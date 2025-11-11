/* eslint-disable no-restricted-globals */
import * as tfModule from 'https://esm.sh/@tensorflow/tfjs@4.21.0?bundle';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';
import {
  generateSyntheticDataset,
  evaluateCorrelationAchievement,
} from './MLPTrainingUtils.js';
import { serializeModelWeights, applySerializedWeights } from './MLPWeightUtils.js';
import { buildRegularizerProfile, sanitizeBatchSize } from './MLPTrainingRegularizers.js';

const tf = (typeof tfModule.default === 'object' && tfModule.default != null) ? tfModule.default : tfModule;

const ctx = self;

const trainingState = {
  context: null,
};

let tfReadyPromise = null;
const ensureTfReady = () => {
  if (!tfReadyPromise) {
    tfReadyPromise = tf.ready ? tf.ready() : Promise.resolve();
  }
  return tfReadyPromise;
};

const defaultTrainingConfig = {
  epochs: 20,
  batchSize: 64,
  learningRate: 0.001,
  sampleCount: 2048,
  noise: 0.05,
  seed: Date.now(),
  temporalSmoothnessWeight: 0.02,
  temporalSmoothnessNorm: 'l2',
  slewRateWeight: 0.015,
  slewRateThreshold: 0.2,
  jacobianWeight: 0.02,
  jacobianNoise: 0.02,
  inputNoiseStd: 0.02,
  noiseConsistencyWeight: 0.35,
  weightDecay: 5e-4,
  curriculumWarmupEpochs: 6,
  curriculumBoost: 1.6,
  shuffleBatches: true,
};

function postMessageSafe(type, payload = {}) {
  ctx.postMessage({ type, payload });
}

function buildModel(config = {}) {
  const {
    inputSize,
    outputSize,
    hiddenLayers = [32],
    activation = 'tanh',
    outputActivation = 'tanh',
  } = config;
  const model = tf.sequential();
  hiddenLayers.forEach((units, index) => {
    model.add(
      tf.layers.dense({
        units,
        activation,
        useBias: true,
        inputShape: index === 0 ? [inputSize] : undefined,
      }),
    );
  });
  model.add(
    tf.layers.dense({
      units: outputSize,
      activation: outputActivation,
      useBias: true,
    }),
  );
  return model;
}

async function ensureContextDisposed(context, { keepModel = false } = {}) {
  if (!context) return;
  context.inputsTensor?.dispose?.();
  context.targetsTensor?.dispose?.();
  context.optimizer?.dispose?.();
  context.optimizer = null;
  if (!keepModel) {
    context.model?.dispose?.();
  }
  trainingState.context = keepModel ? context : null;
}

async function finalizeSession(context, status = 'completed') {
  if (!context) return;

  if (!context.best || !context.best.weights) {
    context.best = {
      weights: await serializeModelWeights(context.model),
      loss: context.lastLoss ?? 0,
      epoch: context.epoch,
    };
  }

  applySerializedWeights(context.model, tf, context.best.weights);
  const predictionTensor = context.model.predict(context.inputsTensor);
  const predictionBuffer = await predictionTensor.data();
  predictionTensor.dispose();

  const achieved = evaluateCorrelationAchievement({
    featureBuffer: context.dataset.featureBuffer,
    baseFeatureBuffer: context.dataset.baseFeatureBuffer,
    predictionBuffer,
    sampleCount: context.dataset.sampleCount,
    audioDims: context.dataset.audioDims,
    baseDims: context.dataset.baseDims,
    outputSize: context.dataset.outputSize,
    featureKeys: context.featureKeys,
    correlations: context.dataset.correlations,
    baseStats: context.dataset.baseStats,
  });

  const metadata = {
    status,
    epochs: context.epoch,
    loss: context.best.loss ?? context.lastLoss ?? 0,
    sampleCount: context.dataset.sampleCount,
    batchSize: context.training.batchSize,
    learningRate: context.training.learningRate,
    noise: context.training.noise,
    seed: context.training.seed,
    createdAt: Date.now(),
    regularization: {
      temporalSmoothnessWeight: context.training.temporalSmoothnessWeight,
      temporalSmoothnessNorm: context.training.temporalSmoothnessNorm,
      slewRateWeight: context.training.slewRateWeight,
      slewRateThreshold: context.training.slewRateThreshold,
      jacobianWeight: context.training.jacobianWeight,
      jacobianNoise: context.training.jacobianNoise,
      inputNoiseStd: context.training.inputNoiseStd,
      noiseConsistencyWeight: context.training.noiseConsistencyWeight,
      weightDecay: context.training.weightDecay,
      curriculumWarmupEpochs: context.training.curriculumWarmupEpochs,
      curriculumBoost: context.training.curriculumBoost,
    },
    correlations: achieved.map((entry) => ({
      id: entry.id,
      featureKey: entry.featureKey,
      targetId: entry.targetId,
      strength: entry.strength,
      polarity: entry.polarity,
      achieved: entry.achieved,
    })),
  };

  postMessageSafe('result', {
    status,
    weights: context.best.weights,
    metadata,
  });

  await ensureContextDisposed(context);
}

function createBatchSchedule(sampleCount, batchSize, shuffle = true) {
  const clampedBatch = sanitizeBatchSize(batchSize, 32);
  const total = Math.max(0, Math.floor(sampleCount));
  if (!total || clampedBatch <= 0) {
    return [];
  }
  const indices = Array.from({ length: total }, (_, i) => i);
  if (shuffle) {
    for (let i = indices.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
  }
  const batches = [];
  for (let start = 0; start < total; start += clampedBatch) {
    batches.push(indices.slice(start, Math.min(total, start + clampedBatch)));
  }
  return batches;
}

function gatherBatch(tensor, indices) {
  if (!tensor || !indices?.length) {
    return null;
  }
  const indexTensor = tf.tensor1d(indices, 'int32');
  const batch = tf.gather(tensor, indexTensor);
  indexTensor.dispose();
  return batch;
}

function computeBatchLoss(context, batchInputs, batchTargets, profile) {
  return tf.tidy(() => {
    const predictions = context.model.apply(batchInputs, { training: true });
    let loss = tf.losses.meanSquaredError(batchTargets, predictions).mean();

    const sequenceLength = predictions.shape?.[0] ?? 0;
    const outputDims = predictions.shape?.[1] ?? 0;
    if (sequenceLength > 1 && outputDims > 0 && (profile.temporalWeight > 0 || profile.slewWeight > 0)) {
      const prev = predictions.slice([0, 0], [sequenceLength - 1, outputDims]);
      const next = predictions.slice([1, 0], [sequenceLength - 1, outputDims]);
      const deltas = next.sub(prev);
      if (profile.temporalWeight > 0) {
        const penalty = profile.temporalNorm === 'l1'
          ? tf.mean(tf.abs(deltas))
          : tf.mean(tf.square(deltas));
        loss = loss.add(penalty.mul(profile.temporalWeight));
      }
      if (profile.slewWeight > 0) {
        const slewPenalty = tf.mean(
          tf.square(
            tf.relu(
              tf.abs(deltas).sub(profile.slewThreshold ?? 0),
            ),
          ),
        );
        loss = loss.add(slewPenalty.mul(profile.slewWeight));
      }
    }

    const wantsNoise = (profile.noiseConsistencyWeight > 0 || profile.jacobianWeight > 0) && profile.noiseStd > 0;
    let noisyPredictions = null;
    if (wantsNoise) {
      const noise = tf.randomNormal(batchInputs.shape, 0, profile.noiseStd);
      const noisyInputs = batchInputs.add(noise);
      noisyPredictions = context.model.apply(noisyInputs, { training: true });
    }

    if (noisyPredictions) {
      if (profile.noiseConsistencyWeight > 0) {
        const noisyLoss = tf.losses.meanSquaredError(batchTargets, noisyPredictions).mean();
        loss = loss.add(noisyLoss.mul(profile.noiseConsistencyWeight));
      }
      if (profile.jacobianWeight > 0) {
        const diff = noisyPredictions.sub(predictions);
        const jacobianPenalty = tf.mean(tf.square(diff));
        loss = loss.add(jacobianPenalty.mul(profile.jacobianWeight));
      }
    }

    if (profile.weightDecay > 0 && Array.isArray(context.model?.trainableWeights)) {
      const l2Terms = context.model.trainableWeights
        .map((param) => tf.mean(tf.square(param.val)));
      if (l2Terms.length) {
        const l2 = l2Terms.length > 1 ? tf.addN(l2Terms) : l2Terms[0];
        loss = loss.add(l2.mul(profile.weightDecay));
      }
    }

    return loss;
  });
}

async function trainBatch(context, indices, profile) {
  if (!indices?.length) {
    return context.lastLoss ?? 0;
  }
  const batchInputs = gatherBatch(context.inputsTensor, indices);
  const batchTargets = gatherBatch(context.targetsTensor, indices);
  if (!batchInputs || !batchTargets) {
    batchInputs?.dispose?.();
    batchTargets?.dispose?.();
    return context.lastLoss ?? 0;
  }
  const lossTensor = context.optimizer.minimize(
    () => computeBatchLoss(context, batchInputs, batchTargets, profile),
    true,
  );
  const lossValue = (await lossTensor.data())[0];
  lossTensor.dispose();
  batchInputs.dispose();
  batchTargets.dispose();
  return lossValue;
}

async function trainEpoch(context) {
  const profile = buildRegularizerProfile(context.training, context.epoch);
  const schedule = createBatchSchedule(
    context.dataset.sampleCount,
    context.training.batchSize,
    context.training.shuffleBatches !== false,
  );
  if (!schedule.length) {
    return context.lastLoss ?? 0;
  }
  let totalLoss = 0;
  let processed = 0;
  for (let i = 0; i < schedule.length; i += 1) {
    if (context.requestAbort || context.requestPause) {
      break;
    }
    // eslint-disable-next-line no-await-in-loop
    const loss = await trainBatch(context, schedule[i], profile);
    totalLoss += loss;
    processed += 1;
  }
  if (!processed) {
    return context.lastLoss ?? 0;
  }
  return totalLoss / processed;
}

async function runTrainingLoop(context) {
  context.running = true;
  context.paused = false;
  postMessageSafe('status', {
    status: 'running',
    epoch: context.epoch,
    epochs: context.training.epochs,
  });

  while (context.epoch < context.training.epochs) {
    if (context.requestAbort) {
      context.running = false;
      await finalizeSession(context, 'aborted');
      return;
    }
    if (context.requestPause) {
      context.running = false;
      context.paused = true;
      context.requestPause = false;
      postMessageSafe('status', {
        status: 'paused',
        epoch: context.epoch,
        loss: context.lastLoss ?? null,
      });
      return;
    }
    // eslint-disable-next-line no-await-in-loop
    const loss = await trainEpoch(context);
    if (context.requestAbort) {
      context.running = false;
      await finalizeSession(context, 'aborted');
      return;
    }
    if (context.requestPause) {
      context.running = false;
      context.paused = true;
      context.requestPause = false;
      postMessageSafe('status', {
        status: 'paused',
        epoch: context.epoch,
        loss: context.lastLoss ?? loss ?? null,
      });
      return;
    }

    context.lastLoss = loss;
    context.epoch += 1;

    if (loss < (context.best?.loss ?? Number.POSITIVE_INFINITY)) {
      context.best = {
        loss,
        epoch: context.epoch,
        weights: await serializeModelWeights(context.model),
      };
      postMessageSafe('best', {
        epoch: context.epoch,
        loss,
      });
    }

    postMessageSafe('progress', {
      epoch: context.epoch,
      epochs: context.training.epochs,
      loss,
    });
  }

  context.running = false;
  await finalizeSession(context, 'completed');
}

async function handleStart(payload = {}) {
  if (trainingState.context?.running) {
    postMessageSafe('error', { message: 'Training already in progress.' });
    return;
  }

  await ensureContextDisposed(trainingState.context);
  await ensureTfReady();

  const training = { ...defaultTrainingConfig, ...(payload.training || {}) };
  training.batchSize = sanitizeBatchSize(training.batchSize, defaultTrainingConfig.batchSize);
  const modelConfig = payload.model || {};
  const featureKeys = payload.featureKeys || [];

  const dataset = generateSyntheticDataset({
    correlations: payload.correlations || [],
    featureKeys,
    positionalFeatures: payload.positionalFeatures || [],
    baseSampleBuffer: payload.baseSamples || null,
    baseSampleCount: payload.baseSampleCount || 0,
    baseDims: payload.baseDims || 0,
    audioDims: payload.audioDims || 0,
    outputSize: modelConfig.outputSize || 0,
    sampleCount: training.sampleCount,
    noise: training.noise,
    seed: training.seed,
  });

  const inputsTensor = tf.tensor2d(dataset.inputs, [dataset.sampleCount, dataset.totalInputs]);
  const targetsTensor = tf.tensor2d(dataset.targets, [dataset.sampleCount, dataset.outputSize]);

  const model = buildModel(modelConfig);
  if (payload.baseWeights?.length) {
    try {
      applySerializedWeights(model, tf, payload.baseWeights);
    } catch (error) {
      console.warn('[MLPTrainingWorker] Failed to apply base weights, continuing with random init.', error);
    }
  }
  const optimizer = tf.train.adam(training.learningRate);

  trainingState.context = {
    dataset,
    model,
    inputsTensor,
    targetsTensor,
    optimizer,
    epoch: 0,
    lastLoss: null,
    best: null,
    running: false,
    paused: false,
    requestPause: false,
    requestAbort: false,
    training,
    modelConfig,
    featureKeys,
  };

  runTrainingLoop(trainingState.context).catch((error) => {
    console.error('[MLPTrainingWorker] Training failed', error);
    postMessageSafe('error', { message: error?.message || 'Training failed' });
    ensureContextDisposed(trainingState.context);
  });
}

function handlePause() {
  if (!trainingState.context?.running) {
    return;
  }
  trainingState.context.requestPause = true;
}

function handleResume() {
  if (!trainingState.context || !trainingState.context.paused) {
    return;
  }
  if (trainingState.context.running) {
    return;
  }
  trainingState.context.paused = false;
  runTrainingLoop(trainingState.context).catch((error) => {
    console.error('[MLPTrainingWorker] Resume failed', error);
    postMessageSafe('error', { message: error?.message || 'Resume failed' });
    ensureContextDisposed(trainingState.context);
  });
}

async function handleAbort() {
  if (!trainingState.context) {
    return;
  }
  if (!trainingState.context.running) {
    await finalizeSession(trainingState.context, 'aborted');
    return;
  }
  trainingState.context.requestAbort = true;
}

ctx.addEventListener('message', (event) => {
  const { type, payload } = event.data || {};
  switch (type) {
    case 'start':
      handleStart(payload);
      break;
    case 'pause':
      handlePause();
      break;
    case 'resume':
      handleResume();
      break;
    case 'abort':
      handleAbort();
      break;
    default:
      postMessageSafe('error', { message: `Unknown message type: ${type}` });
      break;
  }
});

postMessageSafe('ready', {
  targets: PARTICLE_PARAMETER_TARGETS,
});
