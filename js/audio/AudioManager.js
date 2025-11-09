import { BaseModule } from '../core/BaseModule.js';

export const AudioManagerEvents = {
  STATE: 'state',
  TRACK_LOADED: 'track_loaded',
  TRACK_ENDED: 'track_ended',
  ERROR: 'error',
  UPLOAD: 'upload',
  DRAG_STATE: 'drag_state',
};

const DEFAULT_ANALYSER_CONFIG = {
  fftSize: 2048,
  smoothingTimeConstant: 0.8,
  minDecibels: -90,
  maxDecibels: -12,
};

const DEFAULT_OPTIONS = {
  tracks: [],
  initialVolume: 0.7,
  analyser: DEFAULT_ANALYSER_CONFIG,
  dragActiveClass: 'audio-drop-active',
  autoAdvanceDelayMs: 1000,
  repeat: false,
};

const AUDIO_MIME_PATTERN = /^audio\//i;
const AUDIO_EXTENSION_PATTERN = /\.(mp3|wav|ogg|m4a|aac|flac)$/i;

const isAudioFile = (file) => {
  if (!file) return false;
  const type = file.type || '';
  const name = file.name || '';
  return AUDIO_MIME_PATTERN.test(type) || AUDIO_EXTENSION_PATTERN.test(name);
};

export class AudioManager extends BaseModule {
  constructor(options = {}) {
    super('AudioManager');

    const analyserOptions = { ...DEFAULT_ANALYSER_CONFIG, ...(options.analyser || {}) };
    const trackList = Array.isArray(options.tracks)
      ? options.tracks.map((track, index) => ({
          ...track,
          id: track.id ?? track.file ?? track.url ?? `track-${index}`,
          title: track.title ?? track.name ?? `Track ${index + 1}`,
          url: track.url ?? track.file ?? track.path,
        }))
      : [];

    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
      analyser: analyserOptions,
      tracks: trackList,
    };
    this.options.initialVolume = this._clampVolume(this.options.initialVolume);
    this.fetchImpl =
      this.options.fetch || (typeof fetch !== 'undefined' ? fetch.bind(globalThis) : null);

    this.context = this.options.context || null;
    this.contextFactory = this.options.contextFactory || null;
    this.analyser = null;
    this.gainNode = null;
    this.sourceNode = null;
    this.fileInput = this.options.fileInput || null;
    this.dropTarget = this.options.dropTarget || null;
    this.dragClass = this.options.dragActiveClass;

    this._trackCache = new Map();
    this._uploadedTracks = [];
    this._listeners = new Map();
    this._currentBuffer = null;
    this._currentTrackMeta = null;
    this._pauseOffset = 0;
    this._startTime = 0;
    this._autoAdvanceTimer = null;
    this.autoAdvanceDelayMs = Number.isFinite(this.options.autoAdvanceDelayMs)
      ? this.options.autoAdvanceDelayMs
      : DEFAULT_OPTIONS.autoAdvanceDelayMs;

    this.state = {
      playing: false,
      unlocked: false,
      isLoading: false,
      currentTrack: null,
      volume: this.options.initialVolume,
      repeat: Boolean(this.options.repeat),
    };
  }

  init() {
    if (this.initialized) {
      return this;
    }

    super.init();
    this._ensureContext();
    this._configureNodes();
    this._attachFileInput(this.fileInput || this._queryFileInput());
    this._attachDropTarget(this.dropTarget || this._defaultDropTarget());
    this._emitState();
    return this;
  }

  dispose() {
    this.stop();
    super.dispose();
  }

  on(event, handler) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    const bucket = this._listeners.get(event);
    bucket.add(handler);
    return () => this.off(event, handler);
  }

  off(event, handler) {
    const bucket = this._listeners.get(event);
    if (bucket) {
      bucket.delete(handler);
    }
  }

  emit(event, payload) {
    const bucket = this._listeners.get(event);
    if (!bucket) return;
    for (const handler of bucket) {
      try {
        handler(payload);
      } catch (error) {
        console.error(`[AudioManager] Listener for ${event} failed`, error);
      }
    }
  }

  getTracks() {
    return [...(this.options.tracks || [])];
  }

  getUploadedTracks() {
    return [...this._uploadedTracks];
  }

  getAnalyser() {
    return this.analyser;
  }

  getState() {
    const duration = this._currentBuffer?.duration ?? 0;
    const progress = this.state.playing
      ? Math.min(duration, this._getCurrentPosition())
      : this._pauseOffset;

    return {
      ...this.state,
      duration,
      progress,
    };
  }

  async resumeContext() {
    this._ensureContext();
    if (!this.context) return false;
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
    if (!this.state.unlocked) {
      this._updateState({ unlocked: true });
    }
    return true;
  }

  async play() {
    if (!this._currentBuffer) {
      const firstTrack = this.getTracks()[0];
      if (!firstTrack) return null;
      return this.playTrack(firstTrack.id ?? firstTrack.url ?? firstTrack.title);
    }
    await this.resumeContext();
    return this._startPlayback(this._currentBuffer, this._currentTrackMeta, {
      offset: this._pauseOffset,
    });
  }

  async playTrack(trackIdOrMeta) {
    const track = this._resolveTrack(trackIdOrMeta);
    if (!track) {
      throw new Error(`[AudioManager] Track "${trackIdOrMeta}" not found.`);
    }

    await this.resumeContext();
    this._updateState({ isLoading: true });
    try {
      const buffer = await this._getTrackBuffer(track);
      return await this._startPlayback(buffer, { ...track, source: track.source || 'bundled' });
    } catch (error) {
      this._emitError(error);
      throw error;
    } finally {
      this._updateState({ isLoading: false });
    }
  }

  pause() {
    if (!this.sourceNode || !this.state.playing) {
      return;
    }
    this._pauseOffset = this._getCurrentPosition();
    this._stopSource();
    this._updateState({ playing: false });
  }

  stop() {
    this._stopSource();
    this._pauseOffset = 0;
    this._updateState({ playing: false });
  }

  setVolume(value, { silent = false } = {}) {
    const clamped = this._clampVolume(value);
    if (this.gainNode) {
      this.gainNode.gain.value = this._perceivedToGain(clamped);
    }
    if (silent) {
      this.state = { ...this.state, volume: clamped };
    } else {
      this._updateState({ volume: clamped });
    }
    return clamped;
  }

  setRepeat(enabled) {
    const next = Boolean(enabled);
    if (this.state.repeat === next) {
      return this.state.repeat;
    }
    this._updateState({ repeat: next });
    return next;
  }

  async handleFile(file) {
    if (!isAudioFile(file)) {
      this._emitError(new Error('Unsupported file type. Please select an audio file.'));
      return null;
    }

    await this.resumeContext();
    this._updateState({ isLoading: true });

    try {
      const arrayBuffer = await file.arrayBuffer();
      const buffer = await this._decodeAudioData(arrayBuffer);
      const meta = {
        id: `upload-${Date.now()}`,
        title: file.name,
        source: 'upload',
      };
      const track = this._registerUploadedTrack(meta, buffer);
      this.emit(AudioManagerEvents.UPLOAD, { track });
      return await this._startPlayback(buffer, track);
    } catch (error) {
      this._emitError(error);
      return null;
    } finally {
      this._updateState({ isLoading: false });
    }
  }

  triggerFilePicker() {
    if (!this.fileInput) {
      this.fileInput = this._queryFileInput();
    }
    this.fileInput?.click?.();
  }

  /* Internal helpers */
  _ensureContext() {
    if (this.context) {
      return;
    }

    let ctx = null;
    if (typeof this.contextFactory === 'function') {
      ctx = this.contextFactory();
    } else if (typeof window !== 'undefined') {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      if (Ctor) {
        ctx = new Ctor();
      }
    }

    if (!ctx) {
      throw new Error('[AudioManager] Web Audio API is unavailable in this environment.');
    }

    this.context = ctx;
  }

  _configureNodes() {
    if (!this.context || this.analyser) return;

    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = this.options.analyser.fftSize;
    this.analyser.smoothingTimeConstant = this.options.analyser.smoothingTimeConstant;
    this.analyser.minDecibels = this.options.analyser.minDecibels;
    this.analyser.maxDecibels = this.options.analyser.maxDecibels;

    this.gainNode = this.context.createGain();
    this.setVolume(this.state.volume, { silent: true });

    this.analyser.connect(this.gainNode);
    this.gainNode.connect(this.context.destination);
  }

  _resolveTrack(idOrMeta) {
    if (!idOrMeta) return null;
    if (typeof idOrMeta === 'object') {
      return idOrMeta;
    }

    const bundled = this.getTracks().find(
      (track) =>
        track.id === idOrMeta ||
        track.title === idOrMeta ||
        track.url === idOrMeta ||
        track.file === idOrMeta,
    );
    if (bundled) {
      return bundled;
    }

    return this._uploadedTracks.find(
      (track) => track.id === idOrMeta || track.title === idOrMeta || track.url === idOrMeta,
    );
  }

  async _getTrackBuffer(track) {
    const cacheKey = track.id || track.url || track.file;
    if (cacheKey && this._trackCache.has(cacheKey)) {
      return this._trackCache.get(cacheKey);
    }

    if (!track.url && !track.file) {
      throw new Error('[AudioManager] Track is missing a url.');
    }

    if (!this.fetchImpl) {
      throw new Error('[AudioManager] fetch is not available (missing polyfill in this environment).');
    }

    const response = await this.fetchImpl(track.url || track.file);
    if (!response.ok) {
      throw new Error(`[AudioManager] Failed to load track (${response.status}).`);
    }
    const arrayBuffer = await response.arrayBuffer();
    const buffer = await this._decodeAudioData(arrayBuffer);
    if (cacheKey) {
      this._trackCache.set(cacheKey, buffer);
    }
    return buffer;
  }

  async _decodeAudioData(arrayBuffer) {
    return new Promise((resolve, reject) => {
      this.context.decodeAudioData(
        arrayBuffer,
        (buffer) => resolve(buffer),
        (error) => reject(error),
      );
    });
  }

  async _startPlayback(buffer, meta = {}, options = {}) {
    if (!buffer) return null;
    const offset = Math.min(Math.max(options.offset || 0, 0), buffer.duration);

    this._stopSource();
    this._clearAutoAdvanceTimer();
    this._currentBuffer = buffer;
    this._currentTrackMeta = meta;

    const sourceNode = this.context.createBufferSource();
    sourceNode.buffer = buffer;
    sourceNode.connect(this.analyser);
    sourceNode.onended = () => {
      const endedTrack = this.state.currentTrack;
      this._pauseOffset = 0;
      this._updateState({ playing: false });
      this.emit(AudioManagerEvents.TRACK_ENDED, { track: endedTrack });
      if (this.state.repeat && endedTrack?.id) {
        Promise.resolve()
          .then(() => this.playTrack(endedTrack.id))
          .catch((error) => this._emitError(error));
      } else {
        this._scheduleAutoAdvance(endedTrack);
      }
    };

    sourceNode.start(0, offset);
    this.sourceNode = sourceNode;
    this._startTime = this.context.currentTime - offset;
    this._pauseOffset = offset;

    const currentTrack = {
      id: meta.id ?? meta.url ?? `track-${Date.now()}`,
      title: meta.title ?? 'Untitled Track',
      source: meta.source ?? 'bundled',
    };

    const payload = {
      track: currentTrack,
      offset,
      isResume: offset > 0,
    };
    this._updateState({ playing: true, currentTrack });
    this.emit(AudioManagerEvents.TRACK_LOADED, payload);
    return currentTrack;
  }

  _stopSource() {
    if (this.sourceNode) {
      try {
        this.sourceNode.onended = null;
        this.sourceNode.stop(0);
      } catch (_) {
        /* noop */
      }
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    this._clearAutoAdvanceTimer();
  }

  _getCurrentPosition() {
    if (!this.context || !this._currentBuffer) {
      return 0;
    }
    return Math.min(this.context.currentTime - this._startTime, this._currentBuffer.duration);
  }

  _attachFileInput(input) {
    if (!input || !input.addEventListener) return;
    this.fileInput = input;

    const onChange = (event) => {
      const files = Array.from(event.target.files || []);
      this._handleFileList(files);
      event.target.value = ''; // allow same file twice
    };

    input.addEventListener('change', onChange);
    this.addDisposable(() => input.removeEventListener('change', onChange));
  }

  _attachDropTarget(target) {
    if (!target || !target.addEventListener) return;
    this.dropTarget = target;

    const prevent = (event) => {
      event.preventDefault();
      event.stopPropagation();
    };

    const onDragOver = (event) => {
      prevent(event);
      target.classList.add(this.dragClass);
      this.emit(AudioManagerEvents.DRAG_STATE, { active: true });
    };

    const onDragLeave = (event) => {
      prevent(event);
      target.classList.remove(this.dragClass);
      this.emit(AudioManagerEvents.DRAG_STATE, { active: false });
    };

    const onDrop = (event) => {
      prevent(event);
      target.classList.remove(this.dragClass);
      this.emit(AudioManagerEvents.DRAG_STATE, { active: false });
      const files = Array.from(event.dataTransfer?.files || []);
      this._handleFileList(files);
    };

    target.addEventListener('dragover', onDragOver);
    target.addEventListener('dragleave', onDragLeave);
    target.addEventListener('drop', onDrop);

    this.addDisposable(() => {
      target.removeEventListener('dragover', onDragOver);
      target.removeEventListener('dragleave', onDragLeave);
      target.removeEventListener('drop', onDrop);
      target.classList.remove(this.dragClass);
    });
  }

  _handleFileList(files) {
    if (!files || !files.length) {
      return;
    }
    const audioFile = files.find((file) => isAudioFile(file));
    if (!audioFile) {
      this._emitError(new Error('Please drop an audio file.'));
      return;
    }
    this.handleFile(audioFile);
  }

  _queryFileInput() {
    if (typeof document === 'undefined') return null;
    return document.querySelector('#file-input');
  }

  _defaultDropTarget() {
    if (typeof document === 'undefined') return null;
    return document.body;
  }

  _emitState() {
    this.emit(AudioManagerEvents.STATE, this.getState());
  }

  _updateState(patch) {
    this.state = { ...this.state, ...patch };
    this._emitState();
  }

  _emitError(error) {
    console.error('[AudioManager]', error);
    this.emit(AudioManagerEvents.ERROR, { error });
  }

  _perceivedToGain(value) {
    const normalized = this._clampVolume(value);
    return normalized ** 2;
  }

  _clampVolume(value) {
    if (!Number.isFinite(value)) {
      return DEFAULT_OPTIONS.initialVolume;
    }
    if (value <= 0) return 0;
    if (value >= 1) return 1;
    return value;
  }

  _clearAutoAdvanceTimer() {
    if (this._autoAdvanceTimer) {
      clearTimeout(this._autoAdvanceTimer);
      this._autoAdvanceTimer = null;
    }
  }

  _scheduleAutoAdvance(track) {
    if (this.state.repeat) {
      return;
    }
    if (!track || track.source !== 'bundled') {
      return;
    }
    const tracks = this.getTracks();
    if (!Array.isArray(tracks) || tracks.length < 2) {
      return;
    }
    const currentIndex = tracks.findIndex((item) => item.id === track.id);
    if (currentIndex < 0) {
      return;
    }
    const nextIndex = (currentIndex + 1) % tracks.length;
    if (nextIndex === currentIndex) {
      return;
    }
    const nextTrack = tracks[nextIndex];
    if (!nextTrack) {
      return;
    }
    this._clearAutoAdvanceTimer();
    this._autoAdvanceTimer = setTimeout(() => {
      this._autoAdvanceTimer = null;
      this.playTrack(nextTrack.id).catch((error) => this._emitError(error));
    }, this.autoAdvanceDelayMs);
  }

  _registerUploadedTrack(meta = {}, buffer = null) {
    const track = {
      id: meta.id || `upload-${Date.now()}`,
      title: meta.title || 'Custom Track',
      source: 'upload',
    };

    const existingIndex = this._uploadedTracks.findIndex((item) => item.id === track.id);
    if (existingIndex >= 0) {
      this._uploadedTracks[existingIndex] = track;
    } else {
      this._uploadedTracks.push(track);
    }

    if (track.id && buffer) {
      this._trackCache.set(track.id, buffer);
    }

    return track;
  }
}
