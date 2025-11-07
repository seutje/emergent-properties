export class BaseModule {
  constructor(name = 'BaseModule') {
    this.name = name;
    this.initialized = false;
    this._disposables = new Set();
  }

  init() {
    this.initialized = true;
    return this;
  }

  dispose() {
    for (const cleanup of this._disposables) {
      try {
        cleanup();
      } catch (error) {
        console.warn(`[${this.name}] dispose error`, error);
      }
    }
    this._disposables.clear();
    this.initialized = false;
  }

  addDisposable(cleanup) {
    if (typeof cleanup === 'function') {
      this._disposables.add(cleanup);
    }
    return cleanup;
  }

  assertInitialized() {
    if (!this.initialized) {
      throw new Error(`[${this.name}] Module not initialized`);
    }
  }

  log(...args) {
    console.log(`[${this.name}]`, ...args);
  }
}
