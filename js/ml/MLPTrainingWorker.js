/* eslint-disable no-restricted-globals */
import * as tfModule from 'https://esm.sh/@tensorflow/tfjs@4.21.0?bundle';
import { PARTICLE_PARAMETER_TARGETS } from './MLPTrainingTargets.js';
import {
  generateSyntheticDataset,
  evaluateCorrelationAchievement,
} from './MLPTrainingUtils.js';
import { serializeModelWeights, applySerializedWeights } from './MLPWeightUtils.js';

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
    predictionBuffer,
    sampleCount: context.dataset.sampleCount,
    audioDims: context.dataset.audioDims,
    outputSize: context.dataset.outputSize,
    featureKeys: context.featureKeys,
    correlations: context.dataset.correlations,
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
    const history = await context.model.fit(context.inputsTensor, context.targetsTensor, {
      epochs: 1,
      batchSize: context.training.batchSize,
      shuffle: true,
    });

    const loss = Array.isArray(history.history.loss)
      ? history.history.loss[0]
      : history.history.loss;
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
  const modelConfig = payload.model || {};
  const featureKeys = payload.featureKeys || [];

  const dataset = generateSyntheticDataset({
    correlations: payload.correlations || [],
    featureKeys,
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
  model.compile({
    optimizer,
    loss: 'meanSquaredError',
  });

  trainingState.context = {
    dataset,
    model,
    inputsTensor,
    targetsTensor,
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
