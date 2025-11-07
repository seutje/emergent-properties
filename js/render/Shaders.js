export const vertexShader = /* glsl */ `
  varying float vIntensity;
  void main() {
    vIntensity = 1.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 2.5;
  }
`;

export const fragmentShader = /* glsl */ `
  varying float vIntensity;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    float alpha = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vec3(0.45, 0.8, 1.0) * vIntensity, alpha);
  }
`;
