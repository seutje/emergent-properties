const DEFAULT_MODEL_SNAPSHOT_URL = './assets/models/default.json';
const MODEL_POOL_SNAPSHOT_URLS = Array.from({ length: 29 }, (_, index) => {
  const id = String(index + 1).padStart(2, '0');
  return `./assets/models/${id}.json`;
});

const createError = (message) => new Error(`[ModelSnapshotLoader] ${message}`);

export async function loadModelSnapshot(url = DEFAULT_MODEL_SNAPSHOT_URL, { fetchImpl } = {}) {
  const fetchFn = fetchImpl || globalThis.fetch;
  if (typeof fetchFn !== 'function') {
    throw createError('Fetch API is not available in this environment.');
  }
  const response = await fetchFn(url, { cache: 'no-cache' });
  if (!response || !response.ok) {
    const status = response?.status ?? 'unknown';
    throw createError(`Failed to load snapshot from ${url} (status ${status}).`);
  }
  const snapshot = await response.json();
  if (!snapshot || !Array.isArray(snapshot.weights) || snapshot.weights.length === 0) {
    throw createError('Snapshot payload is missing serialized weights.');
  }
  return snapshot;
}

export function loadDefaultModelSnapshot(options) {
  return loadModelSnapshot(DEFAULT_MODEL_SNAPSHOT_URL, options);
}

export async function loadRandomSnapshotFromPool({
  urls = MODEL_POOL_SNAPSHOT_URLS,
  fetchImpl,
  rng = Math.random,
} = {}) {
  const pool = Array.isArray(urls) ? urls.filter((url) => typeof url === 'string' && url.trim()) : [];
  if (!pool.length) {
    throw createError('No model snapshot URLs are available for random selection.');
  }
  const randomFn = typeof rng === 'function' ? rng : Math.random;
  const value = randomFn();
  const normalized = Number.isFinite(value) ? Math.min(0.999999999, Math.max(0, value)) : Math.random();
  const index = Math.floor(normalized * pool.length);
  const url = pool[index];
  const snapshot = await loadModelSnapshot(url, { fetchImpl });
  return { snapshot, url };
}

export { DEFAULT_MODEL_SNAPSHOT_URL, MODEL_POOL_SNAPSHOT_URLS };
