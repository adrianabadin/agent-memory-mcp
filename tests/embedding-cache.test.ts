import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { EmbeddingCache } from '../src/embedding-cache.js';

describe('EmbeddingCache', () => {
  let dir: string;
  let cache: EmbeddingCache;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'emb-cache-'));
    cache = new EmbeddingCache(dir, 'test-model');
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('returns null on cache miss', async () => {
    expect(await cache.get('unknown text')).toBeNull();
  });

  it('round-trips a vector', async () => {
    const vec = [0.1, 0.2, 0.3, 0.4];
    await cache.set('hello world', vec);
    const retrieved = await cache.get('hello world');
    expect(retrieved).not.toBeNull();
    expect(retrieved!.length).toBe(vec.length);
    for (let i = 0; i < vec.length; i++) {
      expect(retrieved![i]).toBeCloseTo(vec[i], 5);
    }
  });

  it('different texts produce different cache entries', async () => {
    await cache.set('text A', [1, 0, 0]);
    await cache.set('text B', [0, 1, 0]);
    const a = await cache.get('text A');
    const b = await cache.get('text B');
    expect(a![0]).toBeCloseTo(1);
    expect(b![1]).toBeCloseTo(1);
  });

  it('same text always returns same vector', async () => {
    const vec = [0.5, 0.5];
    await cache.set('repeated', vec);
    const first = await cache.get('repeated');
    const second = await cache.get('repeated');
    expect(first).toEqual(second);
  });

  it('different models produce different cache keys (no cross-contamination)', async () => {
    const cacheA = new EmbeddingCache(dir, 'model-A');
    const cacheB = new EmbeddingCache(dir, 'model-B');
    await cacheA.set('same text', [1, 0]);
    await cacheB.set('same text', [0, 1]);
    const fromA = await cacheA.get('same text');
    const fromB = await cacheB.get('same text');
    expect(fromA![0]).toBeCloseTo(1);
    expect(fromB![1]).toBeCloseTo(1);
  });

  it('creates cache directory automatically when nested', async () => {
    const nestedDir = join(dir, 'deep', 'nested', 'cache');
    const nestedCache = new EmbeddingCache(nestedDir, 'model');
    await nestedCache.set('x', [1, 2, 3]);
    const result = await nestedCache.get('x');
    expect(result).not.toBeNull();
  });

  it('leaves no .tmp files after write', async () => {
    await cache.set('test', [1, 2, 3]);
    const { readdir } = await import('fs/promises');
    const files = await readdir(dir);
    const tmpFiles = files.filter(f => f.endsWith('.tmp'));
    expect(tmpFiles).toHaveLength(0);
  });
});
