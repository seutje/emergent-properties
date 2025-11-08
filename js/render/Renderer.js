import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { BaseModule } from '../core/BaseModule.js';
import { ParticleField } from './ParticleField.js';

const DEFAULTS = {
  clearColor: '#050505',
  placeholderCount: 32768,
  pixelRatioLimit: 1.8,
};

export class Renderer extends BaseModule {
  constructor(container, options = {}) {
    super('Renderer');
    this.container = container;
    this.options = { ...DEFAULTS, ...options };
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.resizeObserver = null;
    this.particleField = null;
    this.elapsed = 0;
    this.onResize = this.onResize.bind(this);
    this._sceneCenter = new THREE.Vector3();
    this._cameraZoom = 0;
  }

  init() {
    if (!this.container) {
      throw new Error('[Renderer] Container element is required.');
    }

    super.init();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.options.clearColor);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(this._getPixelRatio());
    this._setOutputColorSpace();
    this.container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
    this.camera.position.set(0, 1.5, 7);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.enablePan = true;
    this.controls.rotateSpeed = 0.65;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.55;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 20;
    this.controls.target.set(0, 0, 0);
    this._cameraZoom = this.camera.position.distanceTo(this.controls.target);

    this.particleField = new ParticleField(this.scene, {
      count: this.options.placeholderCount,
    });
    this.particleField.init();
    this.particleField.setPixelRatio(this._getPixelRatio());
    this._syncCameraTarget();

    this._observeContainer();
    this.onResize();
    return this;
  }

  update(delta = 0) {
    if (!this.renderer || !this.scene || !this.camera) {
      return;
    }

    this.elapsed += delta;
    this.particleField?.update(delta);
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    if (!this.renderer || !this.camera) {
      return;
    }

    const rect = this.container.getBoundingClientRect();
    const width = rect.width || window.innerWidth || 1;
    const height = rect.height || window.innerHeight || 1;

    this.renderer.setPixelRatio(this._getPixelRatio());
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.particleField?.setPixelRatio(this._getPixelRatio());
    this._syncCameraTarget();
  }

  dispose() {
    this.particleField?.dispose();
    this.controls?.dispose?.();
    this.resizeObserver?.disconnect();

    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement.parentElement === this.container) {
        this.container.removeChild(this.renderer.domElement);
      }
    }

    super.dispose();
  }

  _observeContainer() {
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', this.onResize);
      this.addDisposable(() => window.removeEventListener('resize', this.onResize));
    }

    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.onResize());
      this.resizeObserver.observe(this.container);
      this.addDisposable(() => this.resizeObserver?.disconnect());
    }
  }

  _getPixelRatio() {
    if (typeof window === 'undefined') {
      return 1;
    }
    return Math.min(window.devicePixelRatio || 1, this.options.pixelRatioLimit);
  }

  _setOutputColorSpace() {
    if (!this.renderer) return;
    if ('outputColorSpace' in this.renderer && THREE.SRGBColorSpace) {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else if ('outputEncoding' in this.renderer) {
      this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
  }

  getParticleField() {
    return this.particleField;
  }

  focusOnParticles() {
    this._syncCameraTarget();
  }

  setCameraZoom(distance = this._cameraZoom || 6) {
    if (!this.camera || !this.controls) {
      return;
    }
    const min = this.controls.minDistance ?? 1;
    const max = this.controls.maxDistance ?? 50;
    const clamped = Math.max(min, Math.min(max, distance));
    const target = this.controls.target.clone();
    const direction = this.camera.position.clone().sub(target);
    if (!direction.lengthSq()) {
      direction.set(0, 0, 1);
    } else {
      direction.normalize();
    }
    const nextPosition = target.clone().add(direction.multiplyScalar(clamped));
    this.camera.position.copy(nextPosition);
    this.camera.updateProjectionMatrix();
    this._cameraZoom = clamped;
    this.controls.update();
  }

  getCameraZoom() {
    if (Number.isFinite(this._cameraZoom) && this._cameraZoom > 0) {
      return this._cameraZoom;
    }
    if (this.camera && this.controls) {
      return this.camera.position.distanceTo(this.controls.target);
    }
    return 0;
  }

  _syncCameraTarget() {
    if (!this.controls || !this.particleField) {
      return;
    }
    const center = this.particleField.getBoundsCenter(this._sceneCenter);
    this.controls.target.copy(center);
    this.controls.update();
    if (this.camera) {
      this.camera.lookAt(center);
      this._cameraZoom = this.camera.position.distanceTo(center);
    }
  }
}
