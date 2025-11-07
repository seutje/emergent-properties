import { DEFAULT_REACTIVITY, deriveReactivity } from './MLPOrchestrator.js';

describe('deriveReactivity', () => {
  it('boosts gain and reduces blend when audio energy spikes', () => {
    const features = {
      rms: 0.82,
      bandLow: 0.9,
      bandMid: 0.65,
      bandHigh: 0.55,
      peak: 0.97,
      tempoProxy: 0.7,
    };
    const prevState = { envelope: 0.2, prevPeak: 0.25 };
    const baseBlend = 0.85;
    const result = deriveReactivity(features, prevState, DEFAULT_REACTIVITY, baseBlend);

    expect(result.envelope).toBeGreaterThan(prevState.envelope);
    expect(result.gain).toBeGreaterThan(DEFAULT_REACTIVITY.floor);
    expect(result.blend).toBeLessThan(baseBlend);
    expect(result.flickerBoost).toBeGreaterThan(1);
  });

  it('decays envelope and restores blend when the signal calms down', () => {
    const baseBlend = 0.8;
    const prevState = { envelope: 0.6, prevPeak: 0.8 };
    const hotFeatures = {
      rms: 0.75,
      bandLow: 0.8,
      bandMid: 0.6,
      bandHigh: 0.55,
      peak: 0.9,
      tempoProxy: 0.5,
    };
    const spike = deriveReactivity(hotFeatures, prevState, DEFAULT_REACTIVITY, baseBlend);

    const calmFeatures = {
      rms: 0.05,
      bandLow: 0.05,
      bandMid: 0.04,
      bandHigh: 0.03,
      peak: 0.04,
      tempoProxy: 0.1,
    };
    const calm = deriveReactivity(
      calmFeatures,
      { envelope: spike.envelope, prevPeak: spike.prevPeak },
      DEFAULT_REACTIVITY,
      baseBlend,
    );

    expect(calm.envelope).toBeLessThan(spike.envelope);
    expect(calm.blend).toBeGreaterThan(spike.blend);
    expect(calm.gain).toBeLessThan(spike.gain);
  });

  it('honors custom reactivity bounds', () => {
    const options = {
      ...DEFAULT_REACTIVITY,
      floor: 0.5,
      ceiling: 1.2,
      blendDrop: 0.5,
      minBlend: 0.4,
      boost: 2,
    };
    const features = {
      rms: 0.9,
      bandLow: 0.85,
      bandMid: 0.6,
      bandHigh: 0.5,
      peak: 0.95,
      tempoProxy: 0.65,
    };
    const result = deriveReactivity(features, { envelope: 0.1, prevPeak: 0.1 }, options, 0.9);

    expect(result.gain).toBeLessThanOrEqual(options.ceiling);
    expect(result.blend).toBeGreaterThanOrEqual(options.minBlend);
  });
});
