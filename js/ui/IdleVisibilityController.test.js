import { jest } from '@jest/globals';
import { attachIdleVisibilityController } from './IdleVisibilityController.js';

describe('attachIdleVisibilityController', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  const createFakeDocument = () => {
    const listeners = new Map();
    return {
      addEventListener(type, handler) {
        const handlers = listeners.get(type) || [];
        handlers.push(handler);
        listeners.set(type, handlers);
      },
      removeEventListener(type, handler) {
        const handlers = listeners.get(type) || [];
        listeners.set(
          type,
          handlers.filter((fn) => fn !== handler),
        );
      },
      dispatch(type) {
        const handlers = listeners.get(type) || [];
        handlers.forEach((handler) => handler({ type }));
      },
    };
  };

  const createFakeElement = () => {
    const listeners = new Map();
    const classSet = new Set();
    const classList = {
      add(cls) {
        classSet.add(cls);
      },
      remove(cls) {
        classSet.delete(cls);
      },
      contains(cls) {
        return classSet.has(cls);
      },
    };
    return {
      classList,
      addEventListener(type, handler) {
        const handlers = listeners.get(type) || [];
        handlers.push(handler);
        listeners.set(type, handlers);
      },
      removeEventListener(type, handler) {
        const handlers = listeners.get(type) || [];
        listeners.set(
          type,
          handlers.filter((fn) => fn !== handler),
        );
      },
      dispatch(type) {
        const handlers = listeners.get(type) || [];
        handlers.forEach((handler) => handler({ type }));
      },
    };
  };

  it('hides the target after inactivity when not hovered', () => {
    const doc = createFakeDocument();
    const el = createFakeElement();

    attachIdleVisibilityController(el, { idleMs: 1000, document: doc });

    doc.dispatch('pointermove');
    jest.advanceTimersByTime(999);
    expect(el.classList.contains('is-auto-hidden')).toBe(false);

    jest.advanceTimersByTime(1);
    expect(el.classList.contains('is-auto-hidden')).toBe(true);
  });

  it('keeps the element visible while hovered and hides once the pointer leaves', () => {
    const doc = createFakeDocument();
    const el = createFakeElement();

    attachIdleVisibilityController(el, { idleMs: 1500, document: doc });

    doc.dispatch('pointermove');
    el.dispatch('pointerenter');
    jest.advanceTimersByTime(5000);
    expect(el.classList.contains('is-auto-hidden')).toBe(false);

    el.dispatch('pointerleave');
    jest.advanceTimersByTime(1500);
    expect(el.classList.contains('is-auto-hidden')).toBe(true);
  });

  it('reveals the element as soon as activity resumes', () => {
    const doc = createFakeDocument();
    const el = createFakeElement();

    attachIdleVisibilityController(el, { idleMs: 1000, document: doc });

    doc.dispatch('pointermove');
    jest.advanceTimersByTime(1000);
    expect(el.classList.contains('is-auto-hidden')).toBe(true);

    doc.dispatch('pointermove');
    expect(el.classList.contains('is-auto-hidden')).toBe(false);
  });
});
