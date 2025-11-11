const clamp = (value, min = -Infinity, max = Infinity) => {
  if (!Number.isFinite(value)) return min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
};

const clamp01 = (value) => clamp(value, 0, 1);

const BASE_POLICY = {
  attack: 0.2,
  release: 0.08,
  maxStep: 0.35,
  deadZone: 0.002,
  hysteresis: 0.004,
  spring: 22,
  damping: null,
  gateThreshold: 0.08,
  gateFrames: 24,
  gateDrift: 0.015,
  restValue: 0,
  min: -Infinity,
  max: Infinity,
};

export const DEFAULT_CHANNEL_POLICIES = {
  rotationSpeed: {
    attack: 0.12,
    release: 0.04,
    maxStep: 0.025,
    spring: 18,
    restValue: 0.08,
  },
  wobbleStrength: {
    attack: 0.2,
    release: 0.05,
    maxStep: 0.03,
    deadZone: 0.001,
    spring: 26,
    restValue: 0.015,
  },
  wobbleFrequency: {
    attack: 0.18,
    release: 0.07,
    maxStep: 0.05,
    deadZone: 0.0025,
    restValue: 0.35,
    spring: 20,
  },
  colorMix: {
    attack: 0.45,
    release: 0.15,
    maxStep: 0.25,
    deadZone: 0.01,
    spring: 38,
    restValue: 1.2,
    gateThreshold: 0.04,
    gateDrift: 0.05,
  },
  alphaScale: {
    attack: 0.3,
    release: 0.12,
    maxStep: 0.2,
    deadZone: 0.01,
    spring: 28,
    restValue: 1,
  },
  pointScale: {
    attack: 0.22,
    release: 0.1,
    maxStep: 0.15,
    deadZone: 0.01,
    spring: 24,
    restValue: 2,
  },
  cameraZoom: {
    attack: 0.08,
    release: 0.03,
    maxStep: 0.12,
    deadZone: 0.02,
    hysteresis: 0.05,
    spring: 14,
    gateThreshold: 0.12,
    gateFrames: 36,
    gateDrift: 0.02,
    restValue: 6.5,
    min: 3,
    max: 12,
  },
};

const resolvePolicyValue = (policy, key, fallback) => {
  if (Object.prototype.hasOwnProperty.call(policy, key)) {
    return policy[key];
  }
  if (Object.prototype.hasOwnProperty.call(BASE_POLICY, key)) {
    return BASE_POLICY[key];
  }
  return fallback;
};

export class OutputResponseController {
  constructor(channelPolicies = {}) {
    this.channelPolicies = channelPolicies || {};
    this.channels = new Map();
  }

  registerChannel(id, options = {}) {
    if (!id) {
      return null;
    }
    const existing = this.channels.get(id);
    const overrides = {
      ...options,
      ...(options.policy || {}),
    };
    delete overrides.initialValue;
    delete overrides.policy;
    const mergedPolicy = {
      ...BASE_POLICY,
      ...(DEFAULT_CHANNEL_POLICIES[id] || {}),
      ...(this.channelPolicies[id] || {}),
      ...overrides,
    };
    const initialValue = Number.isFinite(options.initialValue)
      ? options.initialValue
      : existing?.value ?? mergedPolicy.restValue ?? 0;
    const channel = {
      id,
      policy: mergedPolicy,
      value: initialValue,
      smooth: initialValue,
      velocity: 0,
      restValue: Number.isFinite(mergedPolicy.restValue) ? mergedPolicy.restValue : initialValue,
      quietFrames: 0,
      lastSign: 0,
    };
    this.channels.set(id, channel);
    return channel.value;
  }

  step(id, targetValue, context = {}) {
    const channel = this.channels.get(id);
    if (!channel) {
      return Number.isFinite(targetValue) ? targetValue : 0;
    }
    const current = channel.value;
    const target = Number.isFinite(targetValue) ? targetValue : current;
    const dt = clamp(context.dt ?? 1 / 24, 1 / 480, 1 / 5);
    const blend = clamp(context.blend ?? 1, 0, 1);
    const policy = channel.policy;

    const blendedTarget = current + (target - current) * blend;
    const alphaAttack = clamp(policy.attack ?? BASE_POLICY.attack, 0.001, 1);
    const alphaRelease = clamp(
      Math.min(policy.release ?? BASE_POLICY.release, alphaAttack),
      0.001,
      1,
    );
    const alpha = blendedTarget >= channel.smooth ? alphaAttack : alphaRelease;
    channel.smooth += (blendedTarget - channel.smooth) * alpha;
    let desired = channel.smooth;

    const energy = clamp01(context.energy ?? 1);
    const gateThreshold = clamp01(resolvePolicyValue(policy, 'gateThreshold', 0));
    const gateFrames = Math.max(0, Math.floor(resolvePolicyValue(policy, 'gateFrames', 0)));
    if (gateFrames > 0 && energy < gateThreshold) {
      channel.quietFrames = Math.min(channel.quietFrames + 1, gateFrames * 2);
    } else if (channel.quietFrames > 0) {
      channel.quietFrames = Math.max(0, channel.quietFrames - 2);
    }
    if (gateFrames > 0 && channel.quietFrames >= gateFrames) {
      const drift = resolvePolicyValue(policy, 'gateDrift', 0);
      const rest = Number.isFinite(policy.restValue) ? policy.restValue : channel.restValue;
      if (drift <= 0) {
        desired = rest;
      } else {
        const toRest = rest - current;
        desired = current + clamp(toRest, -drift, drift);
      }
    }

    let error = desired - current;
    const absError = Math.abs(error);
    const sign = Math.sign(error);
    const deadZone = Math.max(0, policy.deadZone ?? BASE_POLICY.deadZone);
    const hysteresis = Math.max(
      deadZone,
      policy.hysteresis ?? BASE_POLICY.hysteresis ?? deadZone,
    );
    if (absError < deadZone) {
      error = 0;
    } else if (sign && channel.lastSign && sign !== channel.lastSign && absError < hysteresis) {
      error = 0;
    }
    channel.lastSign = error === 0 ? 0 : sign;

    const spring = Math.max(0, policy.spring ?? BASE_POLICY.spring);
    const damping = Number.isFinite(policy.damping)
      ? policy.damping
      : spring > 0
        ? 2 * Math.sqrt(spring)
        : 0;
    channel.velocity += (spring * error - damping * channel.velocity) * dt;
    let next = current + channel.velocity * dt;

    const maxStep = policy.maxStep ?? BASE_POLICY.maxStep;
    if (Number.isFinite(maxStep) && maxStep > 0) {
      const delta = clamp(next - current, -maxStep, maxStep);
      next = current + delta;
      if (dt > 0) {
        channel.velocity = delta / dt;
      }
    }

    const minBound = policy.min ?? BASE_POLICY.min;
    const maxBound = policy.max ?? BASE_POLICY.max;
    next = clamp(next, minBound, maxBound);

    channel.value = next;
    return next;
  }

  getValue(id) {
    return this.channels.get(id)?.value ?? null;
  }

  reset(id, value = null) {
    const channel = this.channels.get(id);
    if (!channel) {
      return;
    }
    const next = Number.isFinite(value) ? value : channel.restValue;
    channel.value = next;
    channel.smooth = next;
    channel.velocity = 0;
    channel.quietFrames = 0;
    channel.lastSign = 0;
  }
}
