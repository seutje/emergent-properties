import { describe, expect, it, jest } from '@jest/globals';
import { ParticleBudgetController, isProbablyMobileDevice } from './ParticleBudgetController.js';

describe('ParticleBudgetController', () => {
  const createField = (initial = 32768) => {
    let count = initial;
    return {
      getParticleCount: () => count,
      setParticleCount: (value) => {
        count = value;
      },
    };
  };

  it('reduces particle count when FPS stays below target', () => {
    const particleField = createField(24000);
    const mlpController = { refreshParticleState: jest.fn() };
    const stats = { fps: 25, particleCount: particleField.getParticleCount() };
    const controller = new ParticleBudgetController({
      particleField,
      mlpController,
      stats,
      isMobile: false,
      options: {
        minCount: 8000,
        sampleInterval: 0.01,
        lowSamplesRequired: 1,
        cooldownSeconds: 0,
        tolerance: 2,
      },
    });
    controller.init();
    controller.update(0.02);
    expect(particleField.getParticleCount()).toBeLessThan(24000);
    expect(particleField.getParticleCount()).toBeGreaterThanOrEqual(8000);
    expect(mlpController.refreshParticleState).toHaveBeenCalled();
    expect(stats.particleCount).toBe(particleField.getParticleCount());
  });

  it('increases particle count when FPS recovers', () => {
    const particleField = createField(12000);
    particleField.setParticleCount(8000);
    const stats = { fps: 75, particleCount: particleField.getParticleCount() };
    const controller = new ParticleBudgetController({
      particleField,
      stats,
      isMobile: false,
      options: {
        minCount: 6000,
        maxCount: 12000,
        sampleInterval: 0.01,
        highSamplesRequired: 1,
        cooldownSeconds: 0,
        tolerance: 2,
      },
    });
    controller.init();
    controller.update(0.02);
    expect(particleField.getParticleCount()).toBeGreaterThan(8000);
    expect(particleField.getParticleCount()).toBeLessThanOrEqual(12000);
  });
});

describe('isProbablyMobileDevice', () => {
  it('detects mobile user agents', () => {
    const mockNavigator = {
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit',
      maxTouchPoints: 5,
    };
    expect(isProbablyMobileDevice(mockNavigator)).toBe(true);
  });

  it('returns false for desktop-like agents', () => {
    const mockNavigator = {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      maxTouchPoints: 0,
    };
    expect(isProbablyMobileDevice(mockNavigator)).toBe(false);
  });
});
