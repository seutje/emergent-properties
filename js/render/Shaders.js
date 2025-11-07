export const particleVertexShader = /* glsl */ `
  attribute float aBaseSize;
  attribute float aSizeDelta;
  attribute vec3 aBaseColor;
  attribute vec3 aColorDelta;
  attribute vec3 aDeltaPos;
  attribute float aFlickerRate;
  attribute float aFlickerDepth;
  attribute float aPhase;

  varying vec3 vColor;
  varying float vAlpha;

  uniform float uTime;
  uniform float uPointScale;
  uniform float uColorMix;
  uniform float uAlphaScale;

  vec3 applyColorDelta(vec3 baseColor, vec3 delta, float mixFactor) {
    vec3 offset = clamp(baseColor + delta, 0.0, 1.0);
    return mix(baseColor, offset, mixFactor);
  }

  void main() {
    float flicker = 0.5 + aFlickerDepth * sin(uTime * aFlickerRate + aPhase);
    vec3 displaced = position + aDeltaPos * flicker;
    vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
    float size = (aBaseSize + aSizeDelta * flicker) * uPointScale;
    gl_PointSize = max(size, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    vColor = applyColorDelta(aBaseColor, aColorDelta, uColorMix);
    vAlpha = clamp(0.2 + flicker, 0.2, 1.0) * uAlphaScale;
  }
`;

export const particleFragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = dot(uv, uv);
    if (d > 1.0) discard;
    float glow = pow(1.0 - d, 1.5);
    gl_FragColor = vec4(vColor, glow * vAlpha);
  }
`;

export function createParticleUniforms() {
  return {
    uTime: { value: 0 },
    uPointScale: { value: 1 },
    uColorMix: { value: 0.65 },
    uAlphaScale: { value: 1 },
  };
}
