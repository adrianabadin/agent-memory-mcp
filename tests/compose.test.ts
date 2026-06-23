import { describe, it, expect } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { buildMemoryComponents, LazyEmbedder } from '../src/compose.js';
import type { Embedder } from '../src/types.js';

function makeStubEmbedder(): Embedder & { initCount: number } {
  const stub = {
    initCount: 0,
    async initialize() {
      stub.initCount++;
    },
    async embed(text: string) {
      return [text.length];
    },
    async embedBatch(texts: string[]) {
      return texts.map((t) => [t.length]);
    },
    dimensions() {
      return 1;
    },
  };
  return stub;
}

describe('compose (unified-server memory components)', () => {
  it('buildMemoryComponents initializes the store schema without downloading the model', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'pmc-compose-'));
    const dbPath = join(dir, 'mem');

    const { store, embedder } = await buildMemoryComponents({
      dbPath,
      embedder: makeStubEmbedder(),
    });

    // The store schema is ready: a read-only call works without the embedder.
    const stats = await store.stats();
    expect(stats).toBeDefined();

    // The embedder is lazy: the model has NOT been initialized at boot.
    expect((embedder as LazyEmbedder).isInitialized()).toBe(false);
  });

  it('buildMemoryComponents requires a dbPath', async () => {
    await expect(buildMemoryComponents({ dbPath: '' })).rejects.toThrow(/dbPath/);
  });

  it('buildMemoryComponents exposes the global store (possibly null) and embedder', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'pmc-compose-'));
    const dbPath = join(dir, 'mem');

    const components = await buildMemoryComponents({
      dbPath,
      embedder: makeStubEmbedder(),
    });

    expect(components.store).toBeDefined();
    expect(components.embedder).toBeDefined();
    // globalStore may be null in restricted environments; the contract is that
    // the field exists so callers can do `globalStore ?? undefined`.
    expect('globalStore' in components).toBe(true);
  });

  it('LazyEmbedder defers initialization until the first embed call', async () => {
    const inner = makeStubEmbedder();
    const lazy = new LazyEmbedder(inner);

    expect(lazy.isInitialized()).toBe(false);
    expect(inner.initCount).toBe(0);

    const vec = await lazy.embed('hello');
    expect(vec).toEqual([5]);
    expect(lazy.isInitialized()).toBe(true);
    expect(inner.initCount).toBe(1);

    // A second embed must not re-initialize (idempotent).
    await lazy.embed('world');
    expect(inner.initCount).toBe(1);
  });

  it('LazyEmbedder reports dimensions without initializing', async () => {
    const inner = makeStubEmbedder();
    const lazy = new LazyEmbedder(inner);

    // dimensions() is synchronous and must not trigger model init.
    expect(lazy.dimensions()).toBe(1);
    expect(lazy.isInitialized()).toBe(false);
  });
});
