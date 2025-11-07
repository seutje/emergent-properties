import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { BaseModule } from '../core/BaseModule.js';
import { particleVertexShader, particleFragmentShader, createParticleUniforms } from './Shaders.js';
import { generateParticleAttributes } from './particleUtils.js';

const DEFAULT_OPTIONS = {
  count: 32768,
  spacing: 0.22,
  jitter: 0.35,
  palette: ['#7de1ff', '#587dff', '#ffc4ff', '#e5fff5'],
  colorVariance: 0.08,
  sizeRange: [2.5, 5.5],
  flickerRateRange: [0.35, 2.75],
  flickerDepthRange: [0.1, 0.45],
  seed: 1337,
  rotationSpeed: 0.08,
  wobbleStrength: 0.12,
  wobbleFrequency: 0.35,
  colorMix: 0.65,
  alphaScale: 1,
  pointScale: 1,
};

export class ParticleField extends BaseModule {
  constructor(scene, options = {}) {
    super('ParticleField');
    this.scene = scene;
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.points = null;
    this.geometry = null;
    this.material = null;
    this.uniforms = null;
    this.elapsed = 0;
    this.attributeHandles = {};
    this.state = null;
    this.defaults = {};
    this._center = new THREE.Vector3();
    this._pixelRatio = 1;
  }

  init() {
    if (!this.scene) {
      throw new Error('[ParticleField] Scene is required before calling init().');
    }
    if (this.points) {
      return this;
    }

    super.init();

    const layout = generateParticleAttributes(this.options);
    this.count = layout.count;
    this.state = {
      positions: layout.positions,
      idHash: layout.idHash,
      phase: layout.phase,
      distOrigin: layout.distOrigin,
      grid: layout.grid,
      seed: layout.seed,
    };

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(layout.positions, 3));
    this.geometry.setAttribute('aBaseColor', new THREE.BufferAttribute(layout.baseColor, 3));
    this.geometry.setAttribute('aBaseSize', new THREE.BufferAttribute(layout.baseSize, 1));
    this.geometry.setAttribute('aPhase', new THREE.BufferAttribute(layout.phase, 1));
    this.geometry.setAttribute('aIdHash', new THREE.BufferAttribute(layout.idHash, 1));
    this.geometry.setAttribute('aFlickerRate', this._createDynamicAttribute(layout.flickerRate, 1));
    this.geometry.setAttribute('aFlickerDepth', this._createDynamicAttribute(layout.flickerDepth, 1));

    const deltaPos = new Float32Array(layout.count * 3);
    const colorDelta = new Float32Array(layout.count * 3);
    const sizeDelta = new Float32Array(layout.count);

    this.geometry.setAttribute('aDeltaPos', this._createDynamicAttribute(deltaPos, 3));
    this.geometry.setAttribute('aColorDelta', this._createDynamicAttribute(colorDelta, 3));
    this.geometry.setAttribute('aSizeDelta', this._createDynamicAttribute(sizeDelta, 1));
    this.geometry.computeBoundingSphere();

    this.uniforms = createParticleUniforms();
    this.uniforms.uColorMix.value = this.options.colorMix;
    this.uniforms.uAlphaScale.value = this.options.alphaScale;
    this.uniforms.uPointScale.value = this.options.pointScale;

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: particleVertexShader,
      fragmentShader: particleFragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    this.attributeHandles = this._buildAttributeHandles();
    this.defaults.flickerRate = new Float32Array(this.attributeHandles.flickerRate.array);
    this.defaults.flickerDepth = new Float32Array(this.attributeHandles.flickerDepth.array);
    this.elapsed = 0;
    return this;
  }

  update(delta = 0) {
    if (!this.points || !this.uniforms) {
      return;
    }
    this.elapsed += delta;
    this.uniforms.uTime.value = this.elapsed;
    this.points.rotation.y += delta * this.options.rotationSpeed;
    this.points.rotation.x =
      Math.sin(this.elapsed * this.options.wobbleFrequency) * this.options.wobbleStrength;
  }

  setPixelRatio(pixelRatio = 1) {
    if (!this.uniforms) return;
    const clamped = Math.max(0.5, Math.min(pixelRatio, 3));
    this._pixelRatio = clamped;
    this._syncPointScale();
  }

  setRotationSpeed(value = this.options.rotationSpeed) {
    this.options.rotationSpeed = Math.max(0, value);
  }

  setWobbleStrength(value = this.options.wobbleStrength) {
    this.options.wobbleStrength = Math.max(0, value);
  }

  setWobbleFrequency(value = this.options.wobbleFrequency) {
    this.options.wobbleFrequency = Math.max(0, value);
  }

  setColorMix(value = this.options.colorMix) {
    this.options.colorMix = Math.max(0, Math.min(1.5, value));
    if (this.uniforms?.uColorMix) {
      this.uniforms.uColorMix.value = this.options.colorMix;
    }
  }

  setAlphaScale(value = this.options.alphaScale) {
    this.options.alphaScale = Math.max(0, Math.min(2, value));
    if (this.uniforms?.uAlphaScale) {
      this.uniforms.uAlphaScale.value = this.options.alphaScale;
    }
  }

  setPointScale(value = this.options.pointScale) {
    this.options.pointScale = Math.max(0.25, Math.min(4, value));
    this._syncPointScale();
  }

  getParticleCount() {
    return this.count || 0;
  }

  getAttributeHandles() {
    return this.attributeHandles;
  }

  getParticleState() {
    return this.state;
  }

  getBoundsCenter(target = this._center) {
    if (!target) {
      target = new THREE.Vector3();
    }
    if (!this.geometry) {
      return target.set(0, 0, 0);
    }
    if (!this.geometry.boundingSphere) {
      this.geometry.computeBoundingSphere();
    }
    if (this.geometry.boundingSphere) {
      target.copy(this.geometry.boundingSphere.center);
    } else {
      target.set(0, 0, 0);
    }
    return target;
  }

  resetDynamicAttributes() {
    ['deltaPos', 'colorDelta'].forEach((key) => {
      const handle = this.attributeHandles[key];
      if (!handle) return;
      handle.array.fill(0);
      handle.markNeedsUpdate();
    });
    const sizeHandle = this.attributeHandles.sizeDelta;
    if (sizeHandle) {
      sizeHandle.array.fill(0);
      sizeHandle.markNeedsUpdate();
    }
    this.restoreFlickerDefaults();
  }

  restoreFlickerDefaults() {
    const rateHandle = this.attributeHandles.flickerRate;
    const depthHandle = this.attributeHandles.flickerDepth;
    if (rateHandle && this.defaults.flickerRate) {
      rateHandle.array.set(this.defaults.flickerRate);
      rateHandle.markNeedsUpdate();
    }
    if (depthHandle && this.defaults.flickerDepth) {
      depthHandle.array.set(this.defaults.flickerDepth);
      depthHandle.markNeedsUpdate();
    }
  }

  dispose() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry?.dispose();
      this.points.material?.dispose();
    }
    this.points = null;
    this.geometry = null;
    this.material = null;
    this.uniforms = null;
    super.dispose();
  }

  _createDynamicAttribute(array, itemSize) {
    const attribute = new THREE.BufferAttribute(array, itemSize);
    if (attribute.setUsage) {
      attribute.setUsage(THREE.DynamicDrawUsage);
    }
    return attribute;
  }

  _buildAttributeHandles() {
    if (!this.geometry) return {};
    const handles = {};
    const map = {
      deltaPos: 'aDeltaPos',
      colorDelta: 'aColorDelta',
      sizeDelta: 'aSizeDelta',
      flickerRate: 'aFlickerRate',
      flickerDepth: 'aFlickerDepth',
    };
    Object.entries(map).forEach(([key, attrName]) => {
      const attribute = this.geometry.getAttribute(attrName);
      if (attribute) {
        handles[key] = {
          name: key,
          attribute,
          array: attribute.array,
          markNeedsUpdate: () => {
            attribute.needsUpdate = true;
          },
        };
      }
    });
    return handles;
  }

  _syncPointScale() {
    if (!this.uniforms?.uPointScale) return;
    const ratio = this._pixelRatio || 1;
    this.uniforms.uPointScale.value = ratio * this.options.pointScale;
  }
}
