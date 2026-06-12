import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { ConcurrentUpdateError, MemoryNotFoundError } from '../src/errors.js';
import { MockEmbedder } from './mocks.js';
import { blobToEmbedding, embeddingToBlob } from '../src/shared/vector-utils.js';

describe('optimistic concurrency control', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let embedder: MockEmbedder;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'concurrent-test-'));
    dbPath = join(tmpDir, 'test.db');
    embedder = new MockEmbedder();
    store = new SqliteMemoryStore(dbPath, embedder);
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    const dir = join(dbPath, '..');
    await rm(dir, { recursive: true, force: true });
  });

  it('new memory has version 1', async () => {
    const mem = await store.store({ content: 'hello', category: 'other', tags: [] });
    expect(mem.version).toBe(1);
  });

  it('update increments version', async () => {
    const mem = await store.store({ content: 'original', category: 'other', tags: [] });
    const updated = await store.update(mem.id, { content: 'updated' });
    expect(updated.version).toBe(2);
  });

  it('sequential updates increment version correctly', async () => {
    const mem = await store.store({ content: 'v1', category: 'other', tags: [] });
    const v2 = await store.update(mem.id, { content: 'v2' });
    const v3 = await store.update(v2.id, { content: 'v3' });
    expect(v3.version).toBe(3);
  });

  it('two SqliteMemoryStore instances on same .db file can both read the same memory', async () => {
    const mem = await store.store({ content: 'shared content', category: 'other', tags: [] });

    // Open a second store on the same .db file
    const store2 = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store2.initialize();

    try {
      // Both stores can read the same memory
      const results = await store2.search('shared content', 'keyword', {});
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.id).toBe(mem.id);
    } finally {
      store2.close();
    }
  });

  it('ConcurrentUpdateError is thrown when version has been bumped externally', async () => {
    const mem = await store.store({ content: 'base', category: 'other', tags: [] });

    // Open a second store on the same .db file to simulate concurrent writer
    const store2 = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store2.initialize();

    try {
      // store2 updates first — bumps to version 2
      await store2.update(mem.id, { content: 'store2 wins' });

      // store1 tries to update with the now-stale version (1) via internal OCC method
      await expect(
        (store as any)._updateWithVersion(mem.id, { content: 'store1 stale update' }, 1),
      ).rejects.toThrow(ConcurrentUpdateError);
    } finally {
      store2.close();
    }
  });

  it('ConcurrentUpdateError carries correct version info', async () => {
    const err = new ConcurrentUpdateError('test-id', 1, 2);
    expect(err.id).toBe('test-id');
    expect(err.expectedVersion).toBe(1);
    expect(err.actualVersion).toBe(2);
    expect(err.name).toBe('ConcurrentUpdateError');
    expect(err.message).toMatch(/version 1/);
    expect(err.message).toMatch(/version 2/);
  });

  it('update on deleted memory throws MemoryNotFoundError', async () => {
    const mem = await store.store({ content: 'to be deleted', category: 'other', tags: [] });
    await store.delete(mem.id);

    await expect(
      store.update(mem.id, { content: 'should fail' })
    ).rejects.toThrow(MemoryNotFoundError);
  });

  it('update on non-existent id throws MemoryNotFoundError', async () => {
    await expect(
      store.update('00000000-0000-0000-0000-000000000000', { content: 'ghost' })
    ).rejects.toThrow(MemoryNotFoundError);
  });

  it('row count stays at 1 after update with optimistic locking', async () => {
    const mem = await store.store({ content: 'initial', category: 'other', tags: [] });
    await store.update(mem.id, { content: 'changed' });
    const stats = await store.stats();
    expect(stats.totalMemories).toBe(1);
  });

  it('embedder failure leaves the row untouched', async () => {
    class FailOnSecondEmbedder extends MockEmbedder {
      private calls = 0;

      override async embed(text: string): Promise<number[]> {
        this.calls += 1;
        if (this.calls >= 2) {
          throw new Error('embed failed');
        }
        return super.embed(text);
      }
    }

    const failingStore = new SqliteMemoryStore(dbPath, new FailOnSecondEmbedder());
    await failingStore.initialize();

    try {
      const mem = await failingStore.store({ content: 'original', category: 'other', tags: [] });

      await expect(
        failingStore.update(mem.id, { content: 'new' }),
      ).rejects.toThrow('embed failed');

      const row = (failingStore as any).db.prepare(
        'SELECT content, version FROM memories WHERE id = ?',
      ).get(mem.id) as { content: string; version: number };

      expect(row.content).toBe('original');
      expect(row.version).toBe(1);
    } finally {
      failingStore.close();
    }
  });

  it('content and embedding move together on update', async () => {
    const mem = await store.store({ content: 'original', category: 'other', tags: [] });
    const updated = await store.update(mem.id, { content: 'updated' });

    const row = (store as any).db.prepare(
      'SELECT embedding FROM memory_embeddings WHERE id = ?',
    ).get(mem.id) as { embedding: Buffer };

    const expectedEmbedder = new MockEmbedder();
    const expected = blobToEmbedding(
      embeddingToBlob(await expectedEmbedder.embed('updated')),
    );

    expect(updated.content).toBe('updated');
    expect(updated.version).toBe(2);
    expect(blobToEmbedding(row.embedding)).toEqual(expected);
  });
});
