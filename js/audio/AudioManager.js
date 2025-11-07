import { BaseModule } from '../core/BaseModule.js';

export class AudioManager extends BaseModule {
  constructor() {
    super('AudioManager');
    this.context = null;
    this.analyser = null;
  }

  async init() {
    this.context = new (window.AudioContext || window.webkitAudioContext)();
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 2048;
  }
}
