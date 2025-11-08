const DEFAULT_MODEL_SNAPSHOT_URL = './assets/models/default.json';

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

export { DEFAULT_MODEL_SNAPSHOT_URL };
