import { MODEL_POOL_SNAPSHOT_URLS } from './ModelSnapshotLoader.js';

const normalizeSnapshotUrls = (urls = MODEL_POOL_SNAPSHOT_URLS) => {
  if (!Array.isArray(urls)) {
    return [];
  }
  return urls.filter((url) => typeof url === 'string' && url.trim());
};

export function createTrackSnapshotMap(tracks = [], snapshotUrls = MODEL_POOL_SNAPSHOT_URLS) {
  const map = new Map();
  if (!Array.isArray(tracks) || !tracks.length) {
    return map;
  }
  const pool = normalizeSnapshotUrls(snapshotUrls);
  tracks.forEach((track, index) => {
    const trackId = track?.id;
    const snapshotUrl = pool[index];
    if (trackId && snapshotUrl) {
      map.set(trackId, snapshotUrl);
    }
  });
  return map;
}

export function getTrackSnapshotUrl(track, trackMap) {
  if (!track || !trackMap || typeof trackMap.get !== 'function') {
    return null;
  }
  const trackId = track.id ?? track.trackId ?? null;
  if (!trackId) {
    return null;
  }
  return trackMap.get(trackId) || null;
}
