import { describe, expect, it } from '@jest/globals';
import { FeatureExtractor } from './FeatureExtractor.js';

const TEST_BANDS = {
  bandSub: [0, 2000],
  bandBass: [2000, 4000],
  bandLowMid: [4000, 6000],
  bandMid: [6000, 9000],
  bandHigh: [9000, 16000],
};

describe('FeatureExtractor', () => {
  it('computes normalized feature vectors from analyser data', () => {
    const analyser = createAnalyser({
      fftSize: 32,
      sampleRate: 32000,
      frequency: [
        0, 100, 0, 150, 200, 0, 0, 80, 0, 0, 0, 50, 0, 0, 0, 0,
      ],
      timeDomain: createWaveSamples(32),
    });

    const extractor = new FeatureExtractor(analyser, {
      sampleRate: 60,
      decimation: { enabled: false },
      smoothing: { enabled: false },
      bands: TEST_BANDS,
    });

    const features = extractor.sample(1 / 60);

    expect(features.rms).toBeCloseTo(computeExpectedRms(createWaveSamples(32)), 5);
    expect(features.specCentroid).toBeCloseTo(0.265, 3);
    expect(features.specRolloff).toBeCloseTo(0.438, 2);
    expect(features.bandSub).toBeCloseTo(0.196, 3);
    expect(features.bandBass).toBeCloseTo(0.294, 3);
    expect(features.bandLowMid).toBeCloseTo(0.392, 3);
    expect(features.bandMid).toBeCloseTo(0.105, 3);
    expect(features.bandHigh).toBeCloseTo(0.028, 3);
    expect(features.peak).toBeCloseTo(200 / 255, 5);
    expect(features.zeroCrossRate).toBeGreaterThan(0.15);
    expect(features.tempoProxy).toBe(0);
  });

  it('applies EMA smoothing when enabled', () => {
    const analyser = createAnalyser({
      fftSize: 32,
      sampleRate: 32000,
    });

    analyser.setFrequencyData(new Array(16).fill(50));
    analyser.setTimeDomainData(createWaveSamples(32));

    const smoothed = new FeatureExtractor(analyser, {
      sampleRate: 60,
      decimation: { enabled: false },
      smoothing: { enabled: true, alpha: 0.5 },
      bands: TEST_BANDS,
    });
    smoothed.sample(1 / 60);
    const previousBass = smoothed.getFeatures().bandBass;

    analyser.setFrequencyData(
      Array.from({ length: 16 }, (_, index) => (index < 4 ? 200 : 10)),
    );
    analyser.setTimeDomainData(createWaveSamples(32, { amplitude: 255 }));

    const raw = new FeatureExtractor(analyser, {
      sampleRate: 60,
      decimation: { enabled: false },
      smoothing: { enabled: false },
      bands: TEST_BANDS,
    });
    const rawFeatures = raw.sample(1 / 60);
    smoothed.sample(1 / 60);

    const expected = previousBass + 0.5 * (rawFeatures.bandBass - previousBass);
    expect(smoothed.getFeatures().bandBass).toBeCloseTo(expected, 5);
  });

  it('respects decimation toggles', () => {
    const analyser = createAnalyser({
      fftSize: 32,
      sampleRate: 32000,
    });
    analyser.setFrequencyData(new Array(16).fill(100));
    analyser.setTimeDomainData(createWaveSamples(32));

    const extractor = new FeatureExtractor(analyser, {
      sampleRate: 30,
      smoothing: { enabled: false },
      decimation: { enabled: true },
    });

    extractor.sample(1 / 30);
    const firstCallCount = analyser.calls.frequency;

    extractor.sample(1 / 120);
    expect(analyser.calls.frequency).toBe(firstCallCount);

    extractor.setDecimationEnabled(false);
    extractor.sample(1 / 120);
    expect(analyser.calls.frequency).toBeGreaterThan(firstCallCount);
  });
});

function createAnalyser({ fftSize = 32, sampleRate = 32000, frequency, timeDomain } = {}) {
  const analyser = new MockAnalyser({ fftSize, sampleRate });
  if (frequency) {
    analyser.setFrequencyData(frequency);
  }
  if (timeDomain) {
    analyser.setTimeDomainData(timeDomain);
  }
  return analyser;
}

function createWaveSamples(length, { amplitude = 255 } = {}) {
  const base = [128, amplitude, 128, 0];
  const samples = [];
  while (samples.length < length) {
    samples.push(...base);
  }
  return samples.slice(0, length);
}

function computeExpectedRms(samples) {
  const normalized = samples.map((value) => (value - 128) / 128);
  const meanSquare =
    normalized.reduce((sum, value) => sum + value * value, 0) / samples.length;
  return Math.sqrt(meanSquare);
}

class MockAnalyser {
  constructor({ fftSize = 32, sampleRate = 32000 } = {}) {
    this.fftSize = fftSize;
    this.frequencyBinCount = fftSize / 2;
    this.context = { sampleRate };
    this.calls = {
      frequency: 0,
      time: 0,
    };
    this._frequencyData = new Uint8Array(this.frequencyBinCount);
    this._timeDomainData = new Uint8Array(this.fftSize);
  }

  getByteFrequencyData(array) {
    this.calls.frequency += 1;
    array.set(this._frequencyData);
  }

  getByteTimeDomainData(array) {
    this.calls.time += 1;
    array.set(this._timeDomainData);
  }

  setFrequencyData(values) {
    this._frequencyData.set(values);
  }

  setTimeDomainData(values) {
    this._timeDomainData.set(values);
  }
}
