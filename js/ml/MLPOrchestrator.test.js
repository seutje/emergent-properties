import { describe, expect, it, jest } from '@jest/globals';
import { MLPModel } from './MLPModel.js';
import { MLPOrchestrator } from './MLPOrchestrator.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';

const createFeatures = (value = 0.5) =>
  FEATURE_KEYS.reduce((acc, key) => {
    acc[key] = value;
    return acc;
  }, {});

describe('MLPOrchestrator', () => {
  it('runs inference and mutates particle attribute buffers', async () => {
    const handles = createAttributeHandles(4);
    const particleField = createParticleField(handles);
    const featureExtractor = {
      getFeatures: () => createFeatures(0.25),
    };

    const model = new MLPModel({
      inputSize: 17,
      outputSize: 9,
      hiddenLayers: [4],
      backend: 'cpu',
    });
    await model.init();

    const orchestrator = new MLPOrchestrator({
      model,
      particleField,
      featureExtractor,
      options: { blend: 1, rateHz: 10 },
    });

    await orchestrator.init();
    await orchestrator.runOnce();

    expect(Array.from(handles.deltaPos.array).some((value) => value !== 0)).toBe(true);
    expect(Array.from(handles.colorDelta.array).some((value) => value !== 0)).toBe(true);
    expect(handles.deltaPos.markNeedsUpdate).toHaveBeenCalled();
    expect(handles.colorDelta.markNeedsUpdate).toHaveBeenCalled();
    expect(handles.sizeDelta.markNeedsUpdate).toHaveBeenCalled();

    orchestrator.dispose();
    model.dispose();
  });
});

function createAttributeHandles(count) {
  const makeHandle = (array) => ({
    array,
    markNeedsUpdate: jest.fn(),
  });
  return {
    deltaPos: makeHandle(new Float32Array(count * 3)),
    colorDelta: makeHandle(new Float32Array(count * 3)),
    sizeDelta: makeHandle(new Float32Array(count)),
    flickerRate: makeHandle(new Float32Array(count).fill(1)),
    flickerDepth: makeHandle(new Float32Array(count).fill(0.2)),
  };
}

function createParticleField(handles) {
  const count = handles.sizeDelta.array.length;
  const positions = new Float32Array(count * 3);
  const distOrigin = new Float32Array(count).fill(0.5);
  const idHash = new Float32Array(count).fill(0.25);
  const phase = new Float32Array(count).fill(Math.PI * 0.5);

  return {
    getParticleState() {
      return { positions, distOrigin, idHash, phase };
    },
    getAttributeHandles() {
      return handles;
    },
  };
}
