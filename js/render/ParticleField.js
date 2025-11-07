import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { BaseModule } from '../core/BaseModule.js';

const DEFAULT_OPTIONS = {
  count: 5000,
  spread: 3.5,
  size: 0.028,
  color: '#7de1ff',
  rotationSpeed: 0.08,
  pulseSpeed: 0.65,
};

export class ParticleField extends BaseModule {
  constructor(scene, options = {}) {
    super('ParticleField');
    this.scene = scene;
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.points = null;
    this.elapsed = 0;
  }

  init() {
    if (!this.scene) {
      throw new Error('[ParticleField] Scene is required before calling init().');
    }

    super.init();

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.options.count * 3);

    for (let i = 0; i < this.options.count; i++) {
      const radius = this.options.spread * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const sinPhi = Math.sin(phi);

      const x = radius * sinPhi * Math.cos(theta);
      const y = radius * Math.cos(phi) * 0.6; // flatten slightly for better parallax
      const z = radius * sinPhi * Math.sin(theta);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();

    const material = new THREE.PointsMaterial({
      color: this.options.color,
      size: this.options.size,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.elapsed = 0;
    return this;
  }

  update(delta = 0) {
    if (!this.points) {
      return;
    }

    this.elapsed += delta;
    this.points.rotation.y += delta * this.options.rotationSpeed;

    const pulse = 0.85 + 0.15 * Math.sin(this.elapsed * this.options.pulseSpeed);
    this.points.material.size = this.options.size * pulse;
  }

  dispose() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry?.dispose();
      this.points.material?.dispose();
      this.points = null;
    }
    super.dispose();
  }
}
