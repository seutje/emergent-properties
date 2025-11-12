import { BaseModule } from '../core/BaseModule.js';

export function isProbablyMobileDevice(navigatorRef = typeof navigator !== 'undefined' ? navigator : null) {
  if (!navigatorRef) {
    return false;
  }
  if (typeof navigatorRef.userAgentData === 'object' && navigatorRef.userAgentData) {
    if (typeof navigatorRef.userAgentData.mobile === 'boolean') {
      return navigatorRef.userAgentData.mobile;
    }
    if (Array.isArray(navigatorRef.userAgentData.brands)) {
      const hint = navigatorRef.userAgentData.brands
        .map((brand) => brand.brand || '')
        .join(' ')
        .toLowerCase();
      if (hint.includes('android') || hint.includes('iphone') || hint.includes('ipad')) {
        return true;
      }
    }
  }

  const ua = String(navigatorRef.userAgent || navigatorRef.appVersion || '').toLowerCase();
  if (!ua) {
    return typeof navigatorRef.maxTouchPoints === 'number' && navigatorRef.maxTouchPoints > 1;
  }
  if (/android|iphone|ipad|ipod|blackberry|iemobile|opera mini|mobile|silk/.test(ua)) {
    return true;
  }
  if (
    typeof navigatorRef.maxTouchPoints === 'number' &&
    navigatorRef.maxTouchPoints > 1 &&
    /touch|tablet/.test(ua)
  ) {
    return true;
  }
  return false;
}

const DEFAULT_PROFILES = {
  desktop: {
    targetFps: 60,
    tolerance: 6,
    reductionFactor: 0.8,
    increaseFactor: 1.12,
    minCount: 12000,
  },
  mobile: {
    targetFps: 30,
    tolerance: 4,
    reductionFactor: 0.7,
    increaseFactor: 1.08,
    minCount: 6000,
  },
};

export class ParticleBudgetController extends BaseModule {
  constructor({
    particleField,
    mlpController = null,
    stats = null,
    isMobile = isProbablyMobileDevice(),
    options = {},
  } = {}) {
    super('ParticleBudget');
    if (!particleField) {
      throw new Error('[ParticleBudget] particleField is required.');
    }
    this.particleField = particleField;
    this.mlpController = mlpController;
    this.stats = stats;

    const profile = { ...(isMobile ? DEFAULT_PROFILES.mobile : DEFAULT_PROFILES.desktop) };
    this.targetFps = Number.isFinite(options.targetFps) ? options.targetFps : profile.targetFps;
    this.tolerance = Number.isFinite(options.tolerance) ? options.tolerance : profile.tolerance;
    this.reductionFactor = Number.isFinite(options.reductionFactor)
      ? options.reductionFactor
      : profile.reductionFactor;
    this.increaseFactor = Number.isFinite(options.increaseFactor)
      ? options.increaseFactor
      : profile.increaseFactor;

    const initialCount = Number(this.particleField.getParticleCount?.() || this.particleField.count || 0) || 0;
    const configuredMin = Number.isFinite(options.minCount) ? options.minCount : profile.minCount;
    const configuredMax = Number.isFinite(options.maxCount) ? options.maxCount : initialCount;
    this.minCount = Math.max(1024, Math.min(configuredMin, initialCount || configuredMin));
    this.maxCount = Math.max(this.minCount, Math.max(initialCount, configuredMax));

    this.lowSamplesRequired = Number.isFinite(options.lowSamplesRequired) ? options.lowSamplesRequired : 3;
    this.highSamplesRequired = Number.isFinite(options.highSamplesRequired) ? options.highSamplesRequired : 6;
    this.sampleInterval = Number.isFinite(options.sampleInterval) ? options.sampleInterval : 0.5;
    this.cooldownSeconds = Number.isFinite(options.cooldownSeconds) ? options.cooldownSeconds : 1.5;

    this.sampleAccumulator = 0;
    this.cooldownTimer = 0;
    this.lowSampleCount = 0;
    this.highSampleCount = 0;

    if (this.stats) {
      if (!Number.isFinite(this.stats.particleCount)) {
        this.stats.particleCount = initialCount;
      }
      this.stats.targetFps = this.targetFps;
    }
  }

  update(deltaSeconds = 0) {
    if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) {
      return;
    }
    this.sampleAccumulator += deltaSeconds;
    if (this.cooldownTimer > 0) {
      this.cooldownTimer = Math.max(0, this.cooldownTimer - deltaSeconds);
    }
    if (this.sampleAccumulator < this.sampleInterval) {
      return;
    }
    this.sampleAccumulator = 0;

    const fps = this.stats?.fps;
    if (!Number.isFinite(fps) || fps <= 0) {
      return;
    }
    const currentCount = Number(this.particleField.getParticleCount?.() || 0);
    if (!Number.isFinite(currentCount) || currentCount <= 0) {
      return;
    }
    if (this.cooldownTimer > 0) {
      return;
    }

    if (fps < this.targetFps - this.tolerance && currentCount > this.minCount) {
      this.lowSampleCount += 1;
      this.highSampleCount = 0;
      if (this.lowSampleCount >= this.lowSamplesRequired) {
        const ratio = Math.max(0.4, fps / this.targetFps);
        const scaled = Math.floor(currentCount * ratio);
        const reduced = Math.floor(currentCount * this.reductionFactor);
        const next = Math.max(this.minCount, Math.min(reduced, scaled));
        this._applyCount(next < currentCount ? next : Math.max(this.minCount, reduced));
        this.lowSampleCount = 0;
      }
      return;
    }

    if (fps > this.targetFps + this.tolerance && currentCount < this.maxCount) {
      this.highSampleCount += 1;
      this.lowSampleCount = 0;
      if (this.highSampleCount >= this.highSamplesRequired) {
        const increased = Math.floor(currentCount * this.increaseFactor);
        const next = Math.max(currentCount + 128, increased);
        this._applyCount(Math.min(this.maxCount, next));
        this.highSampleCount = 0;
      }
      return;
    }

    this.lowSampleCount = 0;
    this.highSampleCount = 0;
  }

  getTargetFps() {
    return this.targetFps;
  }

  _applyCount(nextCount) {
    const current = Number(this.particleField.getParticleCount?.() || 0);
    const clamped = Math.max(this.minCount, Math.min(this.maxCount, Math.floor(nextCount)));
    if (!Number.isFinite(clamped) || clamped <= 0 || clamped === current) {
      return;
    }
    this.particleField.setParticleCount?.(clamped);
    this.mlpController?.refreshParticleState?.();
    if (this.stats) {
      this.stats.particleCount = this.particleField.getParticleCount?.() || clamped;
    }
    this.cooldownTimer = this.cooldownSeconds;
  }
}
