import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.umd.min.js';
import { BaseModule } from '../core/BaseModule.js';

export class UIController extends BaseModule {
  constructor(config = {}) {
    super('UIController');
    this.gui = new GUI({ title: config.title || 'Emergent Properties' });
  }

  addFolder(name) {
    return this.gui.addFolder(name);
  }
}
