export const CORRELATION_FEATURE_SOURCES = Object.freeze({
  AUDIO: 'audio',
  PARTICLE: 'particle',
});

export const PARTICLE_POSITIONAL_FEATURES = Object.freeze([
  {
    id: 'particlePosX',
    label: 'Particle Position X',
    baseIndex: 0,
  },
  {
    id: 'particlePosY',
    label: 'Particle Position Y',
    baseIndex: 1,
  },
  {
    id: 'particlePosZ',
    label: 'Particle Position Z',
    baseIndex: 2,
  },
  {
    id: 'particleDistOrigin',
    label: 'Particle Distance to Origin',
    baseIndex: 3,
  },
]);

export const PARTICLE_POSITIONAL_FEATURE_MAP = PARTICLE_POSITIONAL_FEATURES.reduce(
  (acc, feature) => {
    acc[feature.id] = feature;
    return acc;
  },
  {},
);

export const getParticlePositionalFeature = (id) =>
  PARTICLE_POSITIONAL_FEATURE_MAP[id] || null;
