import { createSeededRandom } from './particleUtils.js';

const GLYPH_HEIGHT = 7;
const DEFAULT_CHAR_WIDTH = 5;
const CHAR_SPACING = 1;
const LINE_SPACING = 2;

const DEFAULT_TEXT_OPTIONS = {
  maxWidth: 10,
  maxHeight: 3.5,
  depthRange: 0.9,
  jitter: 0.35,
};

const RAW_GLYPHS = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  B: ['11110', '10001', '11110', '10001', '10001', '10001', '11110'],
  C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '11110', '10000', '10000', '10000', '11111'],
  F: ['11111', '10000', '11110', '10000', '10000', '10000', '10000'],
  G: ['01111', '10000', '10000', '10111', '10001', '10001', '01111'],
  H: ['10001', '10001', '11111', '10001', '10001', '10001', '10001'],
  I: ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
  J: ['11111', '00010', '00010', '00010', '00010', '10010', '01100'],
  K: ['10001', '10010', '11100', '10010', '10001', '10001', '10001'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  M: ['10001', '11011', '10101', '10001', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
  V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
  W: ['10001', '10001', '10001', '10101', '10101', '10101', '01010'],
  X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
  Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
  Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
  0: ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  1: ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  2: ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  3: ['11110', '00001', '00001', '00110', '00001', '00001', '11110'],
  4: ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  5: ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  6: ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  7: ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  8: ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  9: ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
  '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
  "'": ['00100', '00100', '00100', '00000', '00000', '00000', '00000'],
  '.': ['00000', '00000', '00000', '00000', '00000', '00110', '00110'],
  ',': ['00000', '00000', '00000', '00000', '00110', '00100', '01000'],
  '!': ['00100', '00100', '00100', '00100', '00100', '00000', '00100'],
  '?': ['01110', '10001', '00010', '00100', '00100', '00000', '00100'],
  '&': ['01100', '10010', '10100', '01000', '10101', '10010', '01101'],
  '/': ['00001', '00010', '00100', '01000', '10000', '00000', '00000'],
  ':': ['00000', '00110', '00110', '00000', '00110', '00110', '00000'],
};

const DEFAULT_GLYPH = ['11111', '10001', '00110', '00110', '00110', '10001', '11111'];

const SPECIAL_PRESERVE_CASE = new Set(["'", '-', '.', ',', '!', '?', '&', '/', ':']);

export function sanitizeTrackLabel(text) {
  if (typeof text !== 'string') {
    if (text === undefined || text === null) {
      return '';
    }
    text = String(text);
  }
  const collapsed = text.replace(/\s+/g, ' ').trim();
  return collapsed;
}

export function buildTextMask(text) {
  const clean = sanitizeTrackLabel(text);
  if (!clean) {
    return null;
  }
  const words = clean
    .split(' ')
    .map((word) => word.trim())
    .filter(Boolean);
  if (!words.length) {
    return null;
  }

  const wordMasks = words.map((word) => buildWordMask(word));
  const maxWidth = Math.max(...wordMasks.map((mask) => Math.max(mask.width, 1)));
  const totalHeight =
    words.length * GLYPH_HEIGHT + (words.length > 1 ? (words.length - 1) * LINE_SPACING : 0);

  const cells = [];
  let cursorY = 0;
  wordMasks.forEach((mask, index) => {
    const clampedWidth = Math.max(mask.width, 1);
    const offsetX = Math.floor((maxWidth - clampedWidth) / 2);
    mask.cells.forEach(({ x, y }) => {
      cells.push({ x: x + offsetX, y: cursorY + y });
    });
    if (index < wordMasks.length - 1) {
      cursorY += GLYPH_HEIGHT + LINE_SPACING;
    }
  });

  return {
    cells,
    width: maxWidth,
    height: totalHeight,
  };
}

function buildWordMask(word) {
  const cells = [];
  if (!word) {
    return { cells, width: 1 };
  }
  let cursorX = 0;
  let lastSpacing = 0;

  for (const rawChar of word) {
    const key = SPECIAL_PRESERVE_CASE.has(rawChar) ? rawChar : rawChar.toUpperCase();
    const glyph = RAW_GLYPHS[key] || DEFAULT_GLYPH;
    const width = glyph[0]?.length || DEFAULT_CHAR_WIDTH;
    const spacing = CHAR_SPACING;

    for (let row = 0; row < GLYPH_HEIGHT; row += 1) {
      const line = glyph[row] || '';
      for (let col = 0; col < width; col += 1) {
        if (line[col] === '1') {
          cells.push({ x: cursorX + col, y: row });
        }
      }
    }

    cursorX += width + spacing;
    lastSpacing = spacing;
  }

  const widthCells = Math.max(cursorX - lastSpacing, 1);
  return {
    cells,
    width: widthCells,
  };
}

export function generateTextLayoutPositions(options = {}) {
  const config = {
    ...DEFAULT_TEXT_OPTIONS,
    ...options,
  };
  const mask = buildTextMask(config.text);
  if (!mask || !mask.cells.length) {
    return null;
  }

  const count = Math.max(1, Math.floor(config.count ?? mask.cells.length));
  const widthCells = Math.max(1, mask.width);
  const heightCells = Math.max(1, mask.height);
  const widthScale =
    config.maxWidth && config.maxWidth > 0 ? config.maxWidth / widthCells : Number.POSITIVE_INFINITY;
  const heightScale =
    config.maxHeight && config.maxHeight > 0
      ? config.maxHeight / heightCells
      : Number.POSITIVE_INFINITY;
  let scale = Math.min(widthScale, heightScale);
  if (!Number.isFinite(scale) || scale <= 0) {
    scale = config.maxWidth ? config.maxWidth / widthCells : 0.2;
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    scale = 0.2;
  }

  const depthRange = Number.isFinite(config.depthRange) ? config.depthRange : DEFAULT_TEXT_OPTIONS.depthRange;
  const jitter = Number.isFinite(config.jitter) ? config.jitter : DEFAULT_TEXT_OPTIONS.jitter;
  const boundsWidth = widthCells * scale;
  const boundsHeight = heightCells * scale;

  const textSeed = hashStringToSeed(config.text || '');
  const rng = createSeededRandom((config.seed ?? 0) ^ textSeed);
  const positions = new Float32Array(count * 3);
  const jitterScale = Math.max(0, jitter) * scale;

  for (let i = 0; i < count; i += 1) {
    const cell = mask.cells[i % mask.cells.length];
    const baseX = ((cell.x + rng()) * scale) - boundsWidth / 2;
    const baseY = boundsHeight / 2 - (cell.y + rng()) * scale;
    const x = baseX + (rng() - 0.5) * jitterScale;
    const y = baseY + (rng() - 0.5) * jitterScale;
    const z = (rng() - 0.5) * depthRange;
    const idx = i * 3;
    positions[idx] = x;
    positions[idx + 1] = y;
    positions[idx + 2] = z;
  }

  return {
    positions,
    width: boundsWidth,
    height: boundsHeight,
  };
}

function hashStringToSeed(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export { GLYPH_HEIGHT };
