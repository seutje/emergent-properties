export class PresetStore {
  constructor({ storageKey = 'emergent-presets', storage = null } = {}) {
    this.storageKey = storageKey;
    this.storage = storage || this._detectStorage();
    this.cache = null;
  }

  getAll() {
    if (!this.cache) {
      this.cache = this._read();
    }
    return this.cache.map((preset) => ({ ...preset, data: { ...preset.data } }));
  }

  upsert(preset) {
    const normalized = this._normalizePreset(preset);
    const presets = this.getAll();
    const index = presets.findIndex((item) => item.name === normalized.name);
    if (index >= 0) {
      presets[index] = normalized;
    } else {
      presets.push(normalized);
    }
    this._write(presets);
    return normalized;
  }

  remove(name) {
    if (!name) return false;
    const presets = this.getAll();
    const next = presets.filter((preset) => preset.name !== name);
    const changed = next.length !== presets.length;
    if (changed) {
      this._write(next);
    }
    return changed;
  }

  clear() {
    this._write([]);
  }

  _normalizePreset(preset = {}) {
    const name = PresetStore.sanitizeName(preset.name) || 'Untitled';
    const data = preset.data && typeof preset.data === 'object' ? preset.data : {};
    return {
      name,
      data,
      createdAt: preset.createdAt || new Date().toISOString(),
    };
  }

  _read() {
    if (!this.storage) return [];
    try {
      const raw = this.storage.getItem(this.storageKey) || '[]';
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .map((item) => this._normalizePreset(item))
        .filter((item) => Boolean(item.name));
    } catch (error) {
      console.warn('[PresetStore] Failed to read presets', error);
      return [];
    }
  }

  _write(list) {
    this.cache = list.map((item) => this._normalizePreset(item));
    if (!this.storage) return;
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.cache));
    } catch (error) {
      console.warn('[PresetStore] Failed to save presets', error);
    }
  }

  _detectStorage() {
    if (typeof window === 'undefined' || !window.localStorage) {
      return null;
    }
    return window.localStorage;
  }

  static sanitizeName(name) {
    if (typeof name !== 'string') {
      return '';
    }
    return name.trim().slice(0, 80);
  }
}
