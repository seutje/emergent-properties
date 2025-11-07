import { describe, expect, it, beforeEach } from '@jest/globals';
import { PresetStore } from '../PresetStore.js';

class MemoryStorage {
  constructor() {
    this.map = new Map();
  }
  getItem(key) {
    return this.map.has(key) ? this.map.get(key) : null;
  }
  setItem(key, value) {
    this.map.set(key, value);
  }
}

describe('PresetStore', () => {
  let storage;
  beforeEach(() => {
    storage = new MemoryStorage();
  });

  it('sanitizes names before writing', () => {
    const store = new PresetStore({ storage, storageKey: 'test' });
    const preset = store.upsert({
      name: '  Chill Bloom  ',
      data: { foo: 1 },
    });

    expect(preset.name).toBe('Chill Bloom');
    expect(store.getAll()).toHaveLength(1);
  });

  it('overwrites existing presets when names collide', () => {
    const store = new PresetStore({ storage, storageKey: 'test' });
    store.upsert({ name: 'Default', data: { a: 1 } });
    store.upsert({ name: 'Default', data: { a: 2 } });

    const presets = store.getAll();
    expect(presets).toHaveLength(1);
    expect(presets[0].data).toEqual({ a: 2 });
  });

  it('removes presets by name and reports status', () => {
    const store = new PresetStore({ storage, storageKey: 'test' });
    store.upsert({ name: 'Delete Me', data: {} });
    expect(store.remove('Delete Me')).toBe(true);
    expect(store.getAll()).toHaveLength(0);
    expect(store.remove('missing')).toBe(false);
  });

  it('falls back gracefully when storage is unavailable', () => {
    const store = new PresetStore({ storage: null, storageKey: 'test' });
    store.upsert({ name: 'Offline', data: { foo: 'bar' } });
    expect(store.getAll()[0].name).toBe('Offline');
    store.clear();
    expect(store.getAll()).toEqual([]);
  });
});
