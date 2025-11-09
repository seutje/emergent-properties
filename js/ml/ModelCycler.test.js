import { ModelCycler } from './ModelCycler.js';

describe('ModelCycler', () => {
  it('returns entries sequentially and wraps to the start', () => {
    const cycler = new ModelCycler(['a.json', 'b.json', 'c.json']);

    expect(cycler.next()).toBe('a.json');
    expect(cycler.next()).toBe('b.json');
    expect(cycler.next()).toBe('c.json');
    expect(cycler.next()).toBe('a.json');
  });

  it('syncs to a known URL and returns the following entry on next()', () => {
    const cycler = new ModelCycler(['foo', 'bar', 'baz']);

    expect(cycler.syncTo('bar')).toBe(true);
    expect(cycler.next()).toBe('baz');
    expect(cycler.next()).toBe('foo');
  });

  it('ignores empty values and reports when nothing is available', () => {
    const cycler = new ModelCycler(['', '   ', null, undefined]);

    expect(cycler.hasNext()).toBe(false);
    expect(cycler.peek()).toBeNull();
    expect(cycler.next()).toBeNull();
    expect(cycler.syncTo('./foo.json')).toBe(false);
    expect(cycler.size()).toBe(0);
  });
});
