import { describe, expect, it } from '@jest/globals';
import {
  createSeededRandom,
  deriveGridFromCount,
  generateParticleAttributes,
} from './particleUtils.js';

describe('particleUtils', () => {
  it('derives near-cubic grid dimensions that satisfy the target count', () => {
    const result = deriveGridFromCount(1000);
    expect(result.x * result.y * result.z).toBeGreaterThanOrEqual(1000);
    expect(Math.abs(result.x - result.y)).toBeLessThanOrEqual(1);
    expect(Math.abs(result.y - result.z)).toBeLessThanOrEqual(1);
  });

  it('generates deterministic attributes for the same seed', () => {
    const config = {
      grid: { x: 4, y: 3, z: 2 },
      spacing: 0.15,
      jitter: 0,
      palette: ['#000000', '#ffffff'],
      sizeRange: [1, 1],
      flickerRateRange: [0.5, 0.5],
      flickerDepthRange: [0.2, 0.2],
      seed: 42,
    };
    const first = generateParticleAttributes(config);
    const second = generateParticleAttributes(config);
    expect(Array.from(second.positions)).toEqual(Array.from(first.positions));
    expect(Array.from(second.idHash)).toEqual(Array.from(first.idHash));
    expect(Array.from(second.baseColor)).toEqual(Array.from(first.baseColor));
  });

  it('keeps generated values within the requested ranges', () => {
    const result = generateParticleAttributes({
      grid: { x: 2, y: 2, z: 2 },
      palette: ['#ff0000', '#00ff00'],
      sizeRange: [1, 2],
      flickerRateRange: [0.4, 0.6],
      flickerDepthRange: [0.1, 0.2],
      jitter: 0.25,
      seed: 7,
    });

    expect(result.positions.length).toBe(24);
    expect(result.baseColor.every((value) => value >= 0 && value <= 1)).toBe(true);
    expect(result.baseSize.every((value) => value >= 1 && value <= 2)).toBe(true);
    expect(result.flickerRate.every((value) => value >= 0.4 && value <= 0.6)).toBe(true);
    expect(result.flickerDepth.every((value) => value >= 0.1 && value <= 0.2)).toBe(true);
    expect(result.idHash.every((value) => value >= 0 && value <= 1)).toBe(true);
  });

  it('centers the geometry by default', () => {
    const result = generateParticleAttributes({
      grid: { x: 4, y: 4, z: 4 },
      jitter: 0.2,
      seed: 11,
    });
    const totals = [0, 0, 0];
    for (let i = 0; i < result.positions.length; i += 3) {
      totals[0] += result.positions[i];
      totals[1] += result.positions[i + 1];
      totals[2] += result.positions[i + 2];
    }
    const count = result.positions.length / 3;
    expect(totals[0] / count).toBeCloseTo(0, 5);
    expect(totals[1] / count).toBeCloseTo(0, 5);
    expect(totals[2] / count).toBeCloseTo(0, 5);
  });

  it('seeded random produces repeatable sequences', () => {
    const randomA = createSeededRandom(99);
    const randomB = createSeededRandom(99);
    const samplesA = Array.from({ length: 5 }, () => randomA());
    const samplesB = Array.from({ length: 5 }, () => randomB());
    expect(samplesA).toEqual(samplesB);
  });
});
