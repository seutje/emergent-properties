const DEFAULT_IDLE_MS = 3000;
const DEFAULT_HIDDEN_CLASS = 'is-auto-hidden';

const noop = () => {};

/**
 * Automatically toggles a CSS class on an element when pointer activity stops.
 * The controller hides the element after the configured idle window as long
 * as the user is not hovering the target.
 *
 * @param {HTMLElement} element
 * @param {Object} options
 * @param {number} [options.idleMs=3000] Delay before hiding in milliseconds.
 * @param {string} [options.hiddenClass='is-auto-hidden'] Class to toggle.
 * @param {Document} [options.document=global document] Document to attach listeners to.
 * @returns {{ dispose: () => void } | null}
 */
export function attachIdleVisibilityController(element, options = {}) {
  if (!element) {
    return null;
  }
  const doc = options.document || (typeof document !== 'undefined' ? document : null);
  if (!doc) {
    return null;
  }

  const idleMs =
    Number.isFinite(options.idleMs) && options.idleMs >= 0 ? options.idleMs : DEFAULT_IDLE_MS;
  const hiddenClass = options.hiddenClass || DEFAULT_HIDDEN_CLASS;
  let timer = null;
  let hovering = false;
  let disposed = false;
  let hasPointerActivity = false;

  const show = () => {
    element.classList.remove(hiddenClass);
  };

  const hide = () => {
    if (!hovering) {
      element.classList.add(hiddenClass);
    }
  };

  const clearTimer = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const scheduleHide = () => {
    clearTimer();
    if (!hasPointerActivity) {
      return;
    }
    if (idleMs === 0) {
      hide();
      return;
    }
    timer = setTimeout(() => {
      hide();
    }, idleMs);
  };

  const handlePointerMove = () => {
    if (disposed) return;
    hasPointerActivity = true;
    show();
    scheduleHide();
  };

  const handlePointerDown = () => {
    handlePointerMove();
  };

  const handlePointerEnter = () => {
    hovering = true;
    show();
    clearTimer();
  };

  const handlePointerLeave = () => {
    hovering = false;
    scheduleHide();
  };

  doc.addEventListener('pointermove', handlePointerMove);
  doc.addEventListener('pointerdown', handlePointerDown);
  element.addEventListener('pointerenter', handlePointerEnter);
  element.addEventListener('pointerleave', handlePointerLeave);

  const dispose = () => {
    if (disposed) {
      return;
    }
    disposed = true;
    clearTimer();
    doc.removeEventListener('pointermove', handlePointerMove);
    doc.removeEventListener('pointerdown', handlePointerDown);
    element.removeEventListener('pointerenter', handlePointerEnter);
    element.removeEventListener('pointerleave', handlePointerLeave);
  };

  return { dispose };
}

export const __testUtils = {
  DEFAULT_IDLE_MS,
  DEFAULT_HIDDEN_CLASS,
  noop,
};
