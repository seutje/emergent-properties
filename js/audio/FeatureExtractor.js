import { BaseModule } from '../core/BaseModule.js';

export class FeatureExtractor extends BaseModule {
  constructor(analyser) {
    super('FeatureExtractor');
    this.analyser = analyser;
    this.data = new Uint8Array(analyser?.frequencyBinCount || 0);
  }

  sample() {
    if (!this.analyser) return {};
    this.analyser.getByteFrequencyData(this.data);
    const avg =
      this.data.reduce((sum, value) => sum + value, 0) / Math.max(this.data.length, 1);
    return { rms: avg / 255 };
  }
}
