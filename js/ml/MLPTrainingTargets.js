export const PARTICLE_PARAMETER_TARGETS = [
  { id: 'deltaPosX', label: 'Δ Position X', outputIndex: 0, axis: 'x', group: 'deltaPos' },
  { id: 'deltaPosY', label: 'Δ Position Y', outputIndex: 1, axis: 'y', group: 'deltaPos' },
  { id: 'deltaPosZ', label: 'Δ Position Z', outputIndex: 2, axis: 'z', group: 'deltaPos' },
  { id: 'sizeDelta', label: 'Size Delta', outputIndex: 3, group: 'size' },
  { id: 'colorR', label: 'Color ΔR', outputIndex: 4, axis: 'r', group: 'color' },
  { id: 'colorG', label: 'Color ΔG', outputIndex: 5, axis: 'g', group: 'color' },
  { id: 'colorB', label: 'Color ΔB', outputIndex: 6, axis: 'b', group: 'color' },
  { id: 'flickerRate', label: 'Flicker Rate', outputIndex: 7, group: 'flickerRate' },
  { id: 'flickerDepth', label: 'Flicker Depth', outputIndex: 8, group: 'flickerDepth' },
];

export const getParticleParameterTarget = (id) =>
  PARTICLE_PARAMETER_TARGETS.find((target) => target.id === id) || null;
