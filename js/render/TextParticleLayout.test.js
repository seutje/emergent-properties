import { describe, expect, it } from '@jest/globals';
import { buildTextMask, generateTextLayoutPositions, sanitizeTrackLabel } from './TextParticleLayout.js';

describe('TextParticleLayout', () => {
  it('sanitizes whitespace-heavy labels', () => {
    expect(sanitizeTrackLabel('  Hello   World  ')).toBe('Hello World');
    expect(sanitizeTrackLabel('\nLine\tBreak')).toBe('Line Break');
  });

  it('builds a text mask with populated cells for supported characters', () => {
    const mask = buildTextMask('AB');
    expect(mask.width).toBeGreaterThan(5);
    expect(mask.height).toBeGreaterThan(0);
    expect(mask.cells.length).toBeGreaterThan(10);
  });

  it('generates deterministic positions that respect the requested bounds', () => {
    const config = { text: 'AI', count: 10, seed: 42, maxWidth: 4, maxHeight: 2, depthRange: 0.2 };
    const first = generateTextLayoutPositions(config);
    const second = generateTextLayoutPositions(config);
    expect(first.positions.length).toBe(30);
    expect(Array.from(first.positions)).toEqual(Array.from(second.positions));
    const xs = [];
    const ys = [];
    for (let i = 0; i < first.positions.length; i += 3) {
      xs.push(first.positions[i]);
      ys.push(first.positions[i + 1]);
    }
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    expect(spanX).toBeLessThanOrEqual(first.width + 0.5);
    expect(spanY).toBeLessThanOrEqual(first.height + 0.5);
  });

  it('returns null when text resolves to no glyphs', () => {
    expect(generateTextLayoutPositions({ text: '' })).toBeNull();
  });
});
