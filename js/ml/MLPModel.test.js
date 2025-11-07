import { describe, expect, it } from '@jest/globals';
import { MLPModel } from './MLPModel.js';

describe('MLPModel', () => {
  it('initializes a sequential model with the requested backend and layers', async () => {
    const model = new MLPModel({
      inputSize: 4,
      outputSize: 2,
      hiddenLayers: [8],
      activation: 'relu',
      backend: 'tensorflow',
    });

    await model.init();
    expect(model.getBackend()).toBe('tensorflow');
    expect(model.model.layers).toHaveLength(2); // 1 hidden + 1 output
    expect(model.model.layers[0].units).toBe(8);
    model.dispose();
  });

  it('rebuilds when config changes and updates hidden layers', async () => {
    const model = new MLPModel({ inputSize: 6, outputSize: 3, hiddenLayers: [10], backend: 'tensorflow' });
    await model.init();
    expect(model.model.layers[0].units).toBe(10);

    await model.rebuild({ hiddenLayers: [12, 12], activation: 'tanh', backend: 'tensorflow' });
    expect(model.model.layers).toHaveLength(3); // 2 hidden + output
    expect(model.model.layers[0].units).toBe(12);
    expect(model.model.layers[1].units).toBe(12);
    model.dispose();
  });

  it('runs predictions that respect configured input/output shapes', async () => {
    const model = new MLPModel({ inputSize: 3, outputSize: 5, hiddenLayers: [4], backend: 'tensorflow' });
    await model.init();
    const tf = model.getTF();
    const input = tf.ones([2, 3]);
    const output = model.predict(input);

    expect(output.shape).toEqual([2, 5]);
    input.dispose();
    output.dispose();
    model.dispose();
  });
});
