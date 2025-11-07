export class BaseModule {
  constructor(name = 'BaseModule') {
    this.name = name;
  }

  init() {
    /* optional override hook */
  }

  dispose() {
    /* optional override hook */
  }

  log(...args) {
    console.log(`[${this.name}]`, ...args);
  }
}
