const normalizeList = (urls = []) => {
  if (!Array.isArray(urls)) {
    return [];
  }
  return urls
    .map((url) => (typeof url === 'string' ? url.trim() : ''))
    .filter((url) => Boolean(url));
};

export class ModelCycler {
  constructor(urls = []) {
    this._urls = normalizeList(urls);
    this._index = 0;
  }

  setUrls(urls = []) {
    this._urls = normalizeList(urls);
    this.reset();
  }

  reset() {
    this._index = 0;
  }

  peek() {
    if (!this._urls.length) {
      return null;
    }
    return this._urls[this._index];
  }

  next() {
    if (!this._urls.length) {
      return null;
    }
    const current = this._urls[this._index];
    this._index = (this._index + 1) % this._urls.length;
    return current;
  }

  syncTo(url) {
    if (!url || !this._urls.length) {
      return false;
    }
    const index = this._urls.indexOf(url);
    if (index === -1) {
      return false;
    }
    this._index = (index + 1) % this._urls.length;
    return true;
  }

  hasNext() {
    return this._urls.length > 0;
  }

  size() {
    return this._urls.length;
  }
}

export default ModelCycler;
