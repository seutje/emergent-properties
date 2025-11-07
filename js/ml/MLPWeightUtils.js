export async function serializeModelWeights(model) {
  if (!model?.getWeights) {
    throw new Error('[MLPWeightUtils] A TensorFlow.js model instance is required.');
  }
  const weights = model.getWeights();
  const payload = [];
  for (let i = 0; i < weights.length; i += 1) {
    const tensor = weights[i];
    // eslint-disable-next-line no-await-in-loop
    const data = await tensor.data();
    payload.push({
      name: tensor.name || `weight_${i}`,
      shape: [...tensor.shape],
      dtype: tensor.dtype,
      values: Array.from(data),
    });
  }
  return payload;
}

export function tensorsFromSerialized(tf, serialized = []) {
  if (!tf) {
    throw new Error('[MLPWeightUtils] TensorFlow.js reference is required.');
  }
  return serialized.map((item) =>
    tf.tensor(item.values, item.shape, item.dtype || undefined),
  );
}

export function applySerializedWeights(model, tf, serialized = []) {
  if (!model?.setWeights) {
    throw new Error('[MLPWeightUtils] Model with setWeights() is required.');
  }
  if (!serialized?.length) {
    return;
  }
  const tensors = tensorsFromSerialized(tf, serialized);
  model.setWeights(tensors);
  tensors.forEach((tensor) => tensor.dispose?.());
}
