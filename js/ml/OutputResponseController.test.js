import { OutputResponseController } from './OutputResponseController.js';

describe('OutputResponseController', () => {
  it('applies asymmetric smoothing and slew limiting', () => {
    const controller = new OutputResponseController();
    controller.registerChannel('rotationSpeed', {
      attack: 0.5,
      release: 0.1,
      maxStep: 0.5,
      restValue: 0,
      initialValue: 0,
    });
    const rise = controller.step('rotationSpeed', 2, { dt: 1 / 30, energy: 1, blend: 1 });
    expect(rise).toBeGreaterThan(0);
    expect(rise).toBeLessThan(2);

    const fall = controller.step('rotationSpeed', -2, { dt: 1 / 30, energy: 1, blend: 1 });
    const deltaRise = rise;
    const deltaFall = rise - fall;

    expect(deltaFall).toBeLessThan(deltaRise);
    expect(Math.abs(fall - rise)).toBeLessThanOrEqual(0.5);
  });

  it('ignores changes inside the dead-zone and hysteresis band', () => {
    const controller = new OutputResponseController();
    controller.registerChannel('colorMix', {
      deadZone: 0.1,
      hysteresis: 0.2,
      attack: 0.4,
      release: 0.4,
      restValue: 0,
      initialValue: 0,
    });

    const idle = controller.step('colorMix', 0.05, { dt: 1 / 60, energy: 1, blend: 1 });
    expect(idle).toBeCloseTo(0, 5);

    const nudge = controller.step('colorMix', -0.05, { dt: 1 / 60, energy: 1, blend: 1 });
    expect(nudge).toBeCloseTo(0, 5);

    const move = controller.step('colorMix', 0.5, { dt: 1 / 60, energy: 1, blend: 1 });
    expect(move).toBeGreaterThan(0.0001);
  });

  it('gates toward rest when energy is quiet', () => {
    const controller = new OutputResponseController();
    controller.registerChannel('cameraZoom', {
      gateThreshold: 0.4,
      gateFrames: 3,
      gateDrift: 0.2,
      restValue: 6,
      initialValue: 9,
    });

    const active = controller.step('cameraZoom', 9, { dt: 1 / 24, energy: 1, blend: 1 });
    expect(active).toBeGreaterThan(8.5);

    for (let i = 0; i < 4; i += 1) {
      controller.step('cameraZoom', 9, { dt: 1 / 24, energy: 0.05, blend: 1 });
    }
    const gated = controller.step('cameraZoom', 9, { dt: 1 / 24, energy: 0.05, blend: 1 });
    expect(gated).toBeLessThan(active);
    expect(gated).toBeGreaterThanOrEqual(6);
    expect(gated).toBeLessThan(9);
  });
});
