import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js';
import { BaseModule } from '../core/BaseModule.js';

export class MLPModel extends BaseModule {
  constructor(config = {}) {
    super('MLPModel');
    this.config = { inputSize: 8, hidden: 16, outputSize: 4, ...config };
    this.model = null;
  }

  async init() {
    const { inputSize, hidden, outputSize } = this.config;
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({ units: hidden, inputShape: [inputSize], activation: 'tanh' }),
        tf.layers.dense({ units: hidden, activation: 'tanh' }),
        tf.layers.dense({ units: outputSize, activation: 'tanh' }),
      ],
    });
  }

  predict(tensor) {
    if (!this.model) return null;
    return tf.tidy(() => this.model.predict(tensor));
  }
}
