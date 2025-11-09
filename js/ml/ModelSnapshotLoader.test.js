import { jest } from '@jest/globals';
import {
  DEFAULT_MODEL_SNAPSHOT_URL,
  MODEL_POOL_SNAPSHOT_URLS,
  loadDefaultModelSnapshot,
  loadModelSnapshot,
  loadRandomSnapshotFromPool,
} from './ModelSnapshotLoader.js';

const createResponse = (overrides = {}) => ({
  ok: true,
  status: 200,
  json: async () => ({ weights: [[0]], config: {} }),
  ...overrides,
});

describe('ModelSnapshotLoader', () => {
  afterEach(() => {
    delete global.fetch;
  });

  it('loads snapshot data from a given URL', async () => {
    const snapshot = { weights: [{ name: 'foo' }], config: { inputSize: 17 } };
    const fetchMock = jest.fn(() => createResponse({ json: async () => snapshot }));

    const result = await loadModelSnapshot('/foo.json', { fetchImpl: fetchMock });

    expect(fetchMock).toHaveBeenCalledWith('/foo.json', { cache: 'no-cache' });
    expect(result).toBe(snapshot);
  });

  it('throws when the response is not ok', async () => {
    const fetchMock = jest.fn(() => createResponse({ ok: false, status: 404 }));

    await expect(loadModelSnapshot('/missing.json', { fetchImpl: fetchMock })).rejects.toThrow(
      'Failed to load snapshot from /missing.json (status 404).',
    );
  });

  it('throws when weights are missing', async () => {
    const fetchMock = jest.fn(() => createResponse({ json: async () => ({ config: {} }) }));

    await expect(loadModelSnapshot('/bad.json', { fetchImpl: fetchMock })).rejects.toThrow(
      'Snapshot payload is missing serialized weights.',
    );
  });

  it('falls back to the global fetch implementation for the default loader', async () => {
    const snapshot = { weights: [{ name: 'bar' }], config: { hiddenLayers: [32] } };
    const fetchMock = jest.fn(() => createResponse({ json: async () => snapshot }));
    global.fetch = fetchMock;

    const result = await loadDefaultModelSnapshot();

    expect(fetchMock).toHaveBeenCalledWith(DEFAULT_MODEL_SNAPSHOT_URL, { cache: 'no-cache' });
    expect(result).toBe(snapshot);
  });

  it('exposes curated pool snapshot URLs', () => {
    expect(MODEL_POOL_SNAPSHOT_URLS).toHaveLength(29);
    expect(MODEL_POOL_SNAPSHOT_URLS[0]).toBe('./assets/models/01.json');
    expect(MODEL_POOL_SNAPSHOT_URLS[MODEL_POOL_SNAPSHOT_URLS.length - 1]).toBe('./assets/models/29.json');
  });

  it('loads a random snapshot from the pool with deterministic RNG', async () => {
    const snapshot = { weights: [{ name: 'pool' }], config: { inputSize: 17 } };
    const fetchMock = jest.fn(() => createResponse({ json: async () => snapshot }));
    const rng = jest.fn(() => 0.41);

    const { snapshot: result, url } = await loadRandomSnapshotFromPool({
      urls: ['a.json', 'b.json', 'c.json'],
      fetchImpl: fetchMock,
      rng,
    });

    expect(url).toBe('b.json');
    expect(result).toBe(snapshot);
    expect(fetchMock).toHaveBeenCalledWith('b.json', { cache: 'no-cache' });
  });

  it('throws when the pool is empty', async () => {
    await expect(loadRandomSnapshotFromPool({ urls: [] })).rejects.toThrow(
      'No model snapshot URLs are available for random selection.',
    );
  });
});
