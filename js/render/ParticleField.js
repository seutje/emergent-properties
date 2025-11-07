import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { BaseModule } from '../core/BaseModule.js';

export class ParticleField extends BaseModule {
  constructor(scene, count = 1000) {
    super('ParticleField');
    this.scene = scene;
    this.count = count;
    this.points = null;
  }

  init() {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.count * 3);
    for (let i = 0; i < this.count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 4;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 4;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: '#66c7ff',
      size: 0.02,
    });

    this.points = new THREE.Points(geometry, material);
    this.scene.add(this.points);
  }
}
