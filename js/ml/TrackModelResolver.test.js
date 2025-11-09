import { describe, expect, it } from '@jest/globals';
import { createTrackSnapshotMap, getTrackSnapshotUrl } from './TrackModelResolver.js';

describe('TrackModelResolver', () => {
  it('maps bundled tracks onto snapshot URLs by index order', () => {
    const tracks = [
      { id: 'track-01' },
      { id: 'track-02' },
      { id: 'track-03' },
    ];
    const snapshots = ['./assets/models/01.json', './assets/models/02.json', './assets/models/03.json'];
    const map = createTrackSnapshotMap(tracks, snapshots);

    expect(map.size).toBe(3);
    expect(map.get('track-01')).toBe('./assets/models/01.json');
    expect(map.get('track-03')).toBe('./assets/models/03.json');
  });

  it('ignores tracks that lack ids or when snapshots are missing', () => {
    const tracks = [{ title: 'Unnamed' }, { id: 'track-01' }];
    const snapshots = ['./assets/models/01.json'];
    const map = createTrackSnapshotMap(tracks, snapshots);

    expect(map.size).toBe(0);
  });

  it('resolves a snapshot URL for a given track meta', () => {
    const tracks = [
      { id: 'track-01' },
      { id: 'track-02' },
    ];
    const snapshots = ['./assets/models/01.json', './assets/models/02.json'];
    const map = createTrackSnapshotMap(tracks, snapshots);

    expect(getTrackSnapshotUrl({ id: 'track-02' }, map)).toBe('./assets/models/02.json');
    expect(getTrackSnapshotUrl({ id: 'missing' }, map)).toBeNull();
    expect(getTrackSnapshotUrl(null, map)).toBeNull();
  });
});
