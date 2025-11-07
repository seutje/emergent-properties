const DEFAULT_CONFIG = {
  count: 32768,
  spacing: 0.22,
  jitter: 0.35,
  palette: ['#7de1ff', '#587dff', '#ffc4ff', '#e5fff5'],
  colorVariance: 0.08,
  sizeRange: [2.5, 5.5],
  flickerRateRange: [0.35, 2.75],
  flickerDepthRange: [0.1, 0.45],
  seed: 1337,
  center: true,
};

export function createSeededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function deriveGridFromCount(count = DEFAULT_CONFIG.count) {
  const target = Math.max(1, Math.floor(count));
  const cubeRoot = Math.cbrt(target);
  const base = Math.max(1, Math.floor(cubeRoot));
  const dims = [base, base, base];

  while (dims[0] * dims[1] * dims[2] < target) {
    dims[dims.indexOf(Math.min(...dims))] += 1;
  }

  return { x: dims[0], y: dims[1], z: dims[2] };
}

export function generateParticleAttributes(options = {}) {
  const config = {
    ...DEFAULT_CONFIG,
    ...options,
    palette: normalizePalette(options.palette ?? DEFAULT_CONFIG.palette),
    grid: options.grid || deriveGridFromCount(options.count ?? DEFAULT_CONFIG.count),
  };

  const grid = config.grid;
  const total = grid.x * grid.y * grid.z;
  const rng = createSeededRandom(config.seed ?? DEFAULT_CONFIG.seed);
  const spacing = config.spacing ?? DEFAULT_CONFIG.spacing;
  const jitterScale = spacing * (config.jitter ?? DEFAULT_CONFIG.jitter);
  const sizeRange = config.sizeRange ?? DEFAULT_CONFIG.sizeRange;
  const flickerRateRange = config.flickerRateRange ?? DEFAULT_CONFIG.flickerRateRange;
  const flickerDepthRange = config.flickerDepthRange ?? DEFAULT_CONFIG.flickerDepthRange;
  const colorVariance = config.colorVariance ?? DEFAULT_CONFIG.colorVariance;

  const positions = new Float32Array(total * 3);
  const idHash = new Float32Array(total);
  const phase = new Float32Array(total);
  const baseColor = new Float32Array(total * 3);
  const baseSize = new Float32Array(total);
  const flickerRate = new Float32Array(total);
  const flickerDepth = new Float32Array(total);
  const distOrigin = new Float32Array(total);

  const offset = {
    x: (grid.x - 1) * spacing * 0.5,
    y: (grid.y - 1) * spacing * 0.5,
    z: (grid.z - 1) * spacing * 0.5,
  };

  let index = 0;
  for (let ix = 0; ix < grid.x; ix += 1) {
    for (let iy = 0; iy < grid.y; iy += 1) {
      for (let iz = 0; iz < grid.z; iz += 1) {
        const px = ix * spacing - offset.x + (rng() - 0.5) * jitterScale;
        const py = iy * spacing - offset.y + (rng() - 0.5) * jitterScale * 0.75;
        const pz = iz * spacing - offset.z + (rng() - 0.5) * jitterScale;
        const positionIndex = index * 3;

        positions[positionIndex] = px;
        positions[positionIndex + 1] = py;
        positions[positionIndex + 2] = pz;

        const color = sampleColor(config.palette, rng, colorVariance);
        baseColor[positionIndex] = color[0];
        baseColor[positionIndex + 1] = color[1];
        baseColor[positionIndex + 2] = color[2];

        baseSize[index] = lerp(sizeRange[0], sizeRange[1], skew(rng()));
        flickerRate[index] = lerp(flickerRateRange[0], flickerRateRange[1], rng());
        flickerDepth[index] = lerp(flickerDepthRange[0], flickerDepthRange[1], rng());
        phase[index] = rng() * Math.PI * 2;
        idHash[index] = rng();
        index += 1;
      }
    }
  }

  if (config.center !== false) {
    subtractCenter(positions);
  }

  for (let i = 0; i < total; i += 1) {
    const idx = i * 3;
    const px = positions[idx];
    const py = positions[idx + 1];
    const pz = positions[idx + 2];
    distOrigin[i] = Math.hypot(px, py, pz);
  }

  return {
    count: total,
    positions,
    idHash,
    phase,
    baseColor,
    baseSize,
    flickerRate,
    flickerDepth,
    distOrigin,
    grid,
    seed: config.seed,
  };
}

function normalizePalette(palette) {
  if (!Array.isArray(palette) || palette.length === 0) {
    return DEFAULT_CONFIG.palette.map(parseColor);
  }
  return palette.map((color) => parseColor(color)).filter(Boolean);
}

function parseColor(value) {
  if (!value && value !== 0) {
    return null;
  }
  if (Array.isArray(value) && value.length === 3) {
    return value.map((component) => clamp01(component));
  }
  if (typeof value === 'number') {
    return [clamp01(value), clamp01(value), clamp01(value)];
  }
  if (typeof value === 'string') {
    const clean = value.trim().replace('#', '');
    if (clean.length === 3) {
      const expanded = clean
        .split('')
        .map((char) => char + char)
        .join('');
      return hexToRgb(expanded);
    }
    if (clean.length === 6) {
      return hexToRgb(clean);
    }
  }
  return null;
}

function hexToRgb(hex) {
  const num = Number.parseInt(hex, 16);
  const r = ((num >> 16) & 0xff) / 255;
  const g = ((num >> 8) & 0xff) / 255;
  const b = (num & 0xff) / 255;
  return [r, g, b];
}

function sampleColor(palette, rng, variance = 0.05) {
  if (!palette.length) {
    return [1, 1, 1];
  }
  const a = palette[Math.floor(rng() * palette.length)];
  const b = palette[Math.floor(rng() * palette.length)];
  const t = rng();
  return [
    clamp01(lerp(a[0], b[0], t) + (rng() - 0.5) * variance),
    clamp01(lerp(a[1], b[1], t) + (rng() - 0.5) * variance),
    clamp01(lerp(a[2], b[2], t) + (rng() - 0.5) * variance),
  ];
}

function clamp01(value) {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function lerp(min, max, t) {
  return min + (max - min) * t;
}

function skew(value) {
  return Math.pow(value, 1.2);
}

function subtractCenter(array) {
  const center = [0, 0, 0];
  const count = array.length / 3;
  if (!count) return;

  for (let i = 0; i < array.length; i += 3) {
    center[0] += array[i];
    center[1] += array[i + 1];
    center[2] += array[i + 2];
  }
  center[0] /= count;
  center[1] /= count;
  center[2] /= count;

  for (let i = 0; i < array.length; i += 3) {
    array[i] -= center[0];
    array[i + 1] -= center[1];
    array[i + 2] -= center[2];
  }
}
