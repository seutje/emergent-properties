import { describe, expect, it, jest } from '@jest/globals';
import { AudioManager, AudioManagerEvents } from './AudioManager.js';

const DEFAULT_TRACK = { id: 'track-01', title: 'Mock Track', url: '/mock.mp3' };

describe('AudioManager', () => {
  it('initializes the audio graph and emits initial state', () => {
    const { manager, context, events } = createManager();
    expect(manager.getAnalyser()).toBeTruthy();
    expect(context.nodes.gain).toHaveLength(1);
    expect(context.nodes.analyser).toHaveLength(1);
    expect(events.at(-1).playing).toBe(false);
    expect(events.at(-1).unlocked).toBe(false);
    expect(manager.getState().volume).toBeCloseTo(0.7);
    expect(context.nodes.gain[0].gain.value).toBeCloseTo(0.49);
  });

  it('loads bundled tracks, starts playback, and caches buffers', async () => {
    const fetchMock = jest.fn(() =>
      Promise.resolve({
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(8),
      }),
    );

    const { manager, context } = createManager({ fetch: fetchMock });
    const track = await manager.playTrack(DEFAULT_TRACK.id);

    expect(track.title).toBe(DEFAULT_TRACK.title);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(context.createdSources).toHaveLength(1);
    const [{ when, offset }] = context.createdSources[0].started;
    expect(offset).toBe(0);
    expect(when).toBeCloseTo(1); // 1 second default delay

    manager.stop();
    await manager.playTrack(DEFAULT_TRACK.id);
    expect(fetchMock).toHaveBeenCalledTimes(1); // uses cache on second play
    expect(manager.getState().playing).toBe(true);
  });

  it('delays fresh track playback to allow the model to load', async () => {
    const { manager, context } = createManager();
    await manager.playTrack(DEFAULT_TRACK.id);

    const [{ when }] = context.createdSources[0].started;
    expect(when).toBeCloseTo(manager.trackStartDelayMs / 1000);

    context.currentTime = when / 2;
    expect(manager.getState().progress).toBe(0);

    context.currentTime = when + 0.25;
    expect(manager.getState().progress).toBeCloseTo(0.25);
  });

  it('captures pause offsets and resumes from the stored position', async () => {
    const { manager, context } = createManager();
    await manager.playTrack(DEFAULT_TRACK.id);
    const startDelay = manager.trackStartDelayMs / 1000;

    context.currentTime = startDelay + 5;
    manager.pause();
    expect(manager.getState().playing).toBe(false);

    context.currentTime = startDelay + 8;
    await manager.play();

    expect(context.createdSources).toHaveLength(2);
    const resumed = context.createdSources.at(-1);
    expect(resumed.started[0].offset).toBeCloseTo(5);
  });

  it('emits offset metadata with TRACK_LOADED events, so resumes can be ignored', async () => {
    const { manager, context } = createManager();
    const payloads = [];
    manager.on(AudioManagerEvents.TRACK_LOADED, (payload) => payloads.push(payload));
    const startDelay = manager.trackStartDelayMs / 1000;

    await manager.playTrack(DEFAULT_TRACK.id);
    expect(payloads[0].offset).toBe(0);
    expect(payloads[0].isResume).toBe(false);

    context.currentTime = startDelay + 3.5;
    manager.pause();
    context.currentTime = startDelay + 7;
    await manager.play();

    expect(payloads[1].isResume).toBe(true);
    expect(payloads[1].offset).toBeCloseTo(3.5);
  });

  it('handles uploaded files and emits upload events', async () => {
    const { manager, context, uploads } = createManager();
    const fakeFile = {
      name: 'custom.mp3',
      type: 'audio/mpeg',
      arrayBuffer: async () => new ArrayBuffer(16),
    };

    const track = await manager.handleFile(fakeFile);
    expect(track.title).toBe('custom.mp3');
    expect(context.createdSources).toHaveLength(1);
    expect(uploads).toHaveLength(1);
    expect(uploads[0].track.title).toBe('custom.mp3');
  });

  it('keeps uploaded tracks in the catalog so they can be replayed via the selector', async () => {
    const { manager, context } = createManager();
    const fakeFile = {
      name: 'custom-two.mp3',
      type: 'audio/mpeg',
      arrayBuffer: async () => new ArrayBuffer(32),
    };

    const uploaded = await manager.handleFile(fakeFile);
    manager.stop();

    const storedUploads = manager.getUploadedTracks();
    expect(storedUploads).toHaveLength(1);
    expect(storedUploads[0]).toMatchObject({ id: uploaded.id, title: 'custom-two.mp3', source: 'upload' });

    const replayed = await manager.playTrack(uploaded.id);
    expect(replayed.id).toBe(uploaded.id);
    expect(context.createdSources).toHaveLength(2);
  });

  it('rejects non-audio uploads with an error event', async () => {
    const { manager, errors } = createManager();
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const fakeFile = {
      name: 'notes.txt',
      type: 'text/plain',
      arrayBuffer: async () => new ArrayBuffer(4),
    };

    const result = await manager.handleFile(fakeFile);
    expect(result).toBeNull();
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/audio file/i);
    consoleSpy.mockRestore();
  });

  it('auto-plays the next bundled track after a short delay when a song ends', async () => {
    const tracks = [
      { id: 'track-01', title: 'Track One', url: '/one.mp3' },
      { id: 'track-02', title: 'Track Two', url: '/two.mp3' },
    ];
    const { manager, context } = createManager({
      tracks,
      autoAdvanceDelayMs: 25,
    });

    await manager.playTrack(tracks[0].id);
    const [source] = context.createdSources;
    expect(source).toBeTruthy();
    const waitForSecondTrack = waitForTrack(manager, tracks[1].id);

    source.onended(); // schedules the delayed autoplay

    await waitForSecondTrack;
    expect(manager.getState().currentTrack.id).toBe(tracks[1].id);
    expect(manager.getState().playing).toBe(true);
  });

  it('wraps playback to the first bundled track when the final song ends', async () => {
    const tracks = [
      { id: 'track-01', title: 'Track One', url: '/one.mp3' },
      { id: 'track-02', title: 'Track Two', url: '/two.mp3' },
    ];
    const { manager, context } = createManager({
      tracks,
      autoAdvanceDelayMs: 25,
    });

    await manager.playTrack(tracks[1].id);
    const source = context.createdSources.at(-1);
    expect(source).toBeTruthy();

    const waitForFirstTrack = waitForTrack(manager, tracks[0].id);
    source.onended();
    await waitForFirstTrack;

    expect(manager.getState().currentTrack.id).toBe(tracks[0].id);
    expect(manager.getState().playing).toBe(true);
  });

  it('skips to the next track via playNextTrack while honoring the start delay', async () => {
    const tracks = [
      { id: 'track-01', title: 'Track One', url: '/one.mp3' },
      { id: 'track-02', title: 'Track Two', url: '/two.mp3' },
      { id: 'track-03', title: 'Track Three', url: '/three.mp3' },
    ];
    const { manager, context } = createManager({ tracks });

    await manager.playTrack(tracks[0].id);
    const waitForSecond = waitForTrack(manager, tracks[1].id);
    await manager.playNextTrack();
    await waitForSecond;

    expect(manager.getState().currentTrack.id).toBe(tracks[1].id);
    const latestSource = context.createdSources.at(-1);
    expect(latestSource.started.at(-1).when).toBeCloseTo(manager.trackStartDelayMs / 1000);

    const waitForThird = waitForTrack(manager, tracks[2].id);
    await manager.playNextTrack();
    await waitForThird;
    expect(manager.getState().currentTrack.id).toBe(tracks[2].id);
  });

  it('includes uploaded tracks when skipping forward by default', async () => {
    const { manager } = createManager({ tracks: [DEFAULT_TRACK] });
    await manager.playTrack(DEFAULT_TRACK.id);

    const fakeFile = {
      name: 'skip-me.mp3',
      type: 'audio/mpeg',
      arrayBuffer: async () => new ArrayBuffer(64),
    };

    const uploaded = await manager.handleFile(fakeFile);
    await manager.playTrack(DEFAULT_TRACK.id);

    const waitForUpload = waitForTrack(manager, uploaded.id);
    await manager.playNextTrack();
    await waitForUpload;

    expect(manager.getState().currentTrack.id).toBe(uploaded.id);
  });

  it('throws when attempting to skip without any tracks available', async () => {
    const { manager } = createManager({ tracks: [] });
    await expect(manager.playNextTrack()).rejects.toThrow(/no tracks/i);
  });

  it('toggles repeat mode via setRepeat and exposes it through state events', () => {
    const { manager, events } = createManager();
    expect(manager.getState().repeat).toBe(false);

    manager.setRepeat(true);
    expect(manager.getState().repeat).toBe(true);
    expect(events.at(-1).repeat).toBe(true);

    manager.setRepeat(false);
    expect(manager.getState().repeat).toBe(false);
    expect(events.at(-1).repeat).toBe(false);
  });

  it('restarts the current track instead of auto-advancing when repeat is enabled', async () => {
    const tracks = [
      { id: 'track-01', title: 'Track One', url: '/one.mp3' },
      { id: 'track-02', title: 'Track Two', url: '/two.mp3' },
    ];
    const { manager, context } = createManager({
      tracks,
      autoAdvanceDelayMs: 5,
    });

    await manager.playTrack(tracks[0].id);
    manager.setRepeat(true);

    const initialSource = context.createdSources.at(-1);
    expect(initialSource).toBeTruthy();

    const waitForLoop = waitForTrack(manager, tracks[0].id);
    initialSource.onended();
    await waitForLoop;

    expect(context.createdSources).toHaveLength(2);
    expect(manager.getState().currentTrack.id).toBe(tracks[0].id);

    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(manager.getState().currentTrack.id).toBe(tracks[0].id);
  });

  it('applies a perceived-linear volume curve and emits state updates', () => {
    const { manager, context, events } = createManager();
    const baselineEvents = events.length;
    manager.setVolume(0.5);
    const state = manager.getState();
    expect(state.volume).toBeCloseTo(0.5);
    expect(context.nodes.gain[0].gain.value).toBeCloseTo(0.25);
    expect(events.length).toBeGreaterThan(baselineEvents);
    expect(events.at(-1).volume).toBeCloseTo(0.5);
  });

  it('routes the analyser ahead of the gain node so analysis ignores volume scaling', async () => {
    const { manager, context } = createManager();
    await manager.playTrack(DEFAULT_TRACK.id);

    const analyser = context.nodes.analyser[0];
    const gain = context.nodes.gain[0];
    const destination = context.destination;
    const latestSource = context.createdSources.at(-1);

    expect(analyser.connections).toContain(gain);
    expect(gain.connections).toContain(destination);
    expect(latestSource.connections).toContain(analyser);
  });
});

/* Helpers */
function createManager(overrides = {}) {
  const context = new MockAudioContext();
  const fetchImpl =
    overrides.fetch ||
    jest.fn(() =>
      Promise.resolve({
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(8),
      }),
    );

  const dropTarget = overrides.dropTarget ?? createStubDropTarget();
  const fileInput = overrides.fileInput ?? createStubInput();

  const events = [];
  const errors = [];
  const uploads = [];

  const manager = new AudioManager({
    tracks: overrides.tracks || [DEFAULT_TRACK],
    autoAdvanceDelayMs: overrides.autoAdvanceDelayMs,
    repeat: overrides.repeat,
    contextFactory: () => context,
    fetch: fetchImpl,
    fileInput,
    dropTarget,
  });

  manager.on(AudioManagerEvents.STATE, (state) => events.push(state));
  manager.on(AudioManagerEvents.ERROR, ({ error }) => errors.push(error));
  manager.on(AudioManagerEvents.UPLOAD, (payload) => uploads.push(payload));

  manager.init();
  return { manager, context, events, errors, uploads };
}

function createStubDropTarget() {
  return {
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    classList: {
      add: jest.fn(),
      remove: jest.fn(),
    },
  };
}

function createStubInput() {
  return {
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    click: jest.fn(),
    files: [],
    value: '',
  };
}

class MockAudioContext {
  constructor() {
    this.destination = new MockAudioNode('destination');
    this.currentTime = 0;
    this.state = 'suspended';
    this.createdSources = [];
    this.nodes = {
      gain: [],
      analyser: [],
    };
  }

  resume = jest.fn().mockImplementation(() => {
    this.state = 'running';
    return Promise.resolve();
  });

  createGain() {
    const node = new MockGainNode();
    this.nodes.gain.push(node);
    return node;
  }

  createAnalyser() {
    const node = new MockAnalyserNode();
    this.nodes.analyser.push(node);
    return node;
  }

  createBufferSource() {
    const node = new MockBufferSourceNode();
    this.createdSources.push(node);
    return node;
  }

  decodeAudioData(arrayBuffer, success) {
    setTimeout(() => success(new MockAudioBuffer()), 0);
  }
}

class MockAudioNode {
  constructor(name = 'node') {
    this.name = name;
    this.connections = [];
  }

  connect(target) {
    this.connections.push(target);
    return target;
  }
  disconnect() {}
}

class MockGainNode extends MockAudioNode {
  constructor() {
    super();
    this.gain = { value: 1 };
  }
}

class MockAnalyserNode extends MockAudioNode {
  constructor() {
    super();
    this.fftSize = 0;
    this.smoothingTimeConstant = 0;
    this.minDecibels = 0;
    this.maxDecibels = 0;
  }
}

class MockBufferSourceNode extends MockAudioNode {
  constructor() {
    super();
    this.started = [];
    this.buffer = null;
    this.onended = null;
  }

  start(when = 0, offset = 0) {
    this.started.push({ when, offset });
  }

  stop() {
    if (typeof this.onended === 'function') {
      this.onended();
    }
  }
}

class MockAudioBuffer {
  constructor(duration = 60) {
    this.duration = duration;
  }
}

function waitForTrack(manager, targetId) {
  return new Promise((resolve) => {
    const dispose = manager.on(AudioManagerEvents.TRACK_LOADED, ({ track }) => {
      if (track.id === targetId) {
        dispose();
        resolve();
      }
    });
  });
}
