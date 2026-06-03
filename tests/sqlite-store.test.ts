import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import { ConcurrentUpdateError, MemoryNotFoundError } from '../src/errors.js';

describe('SqliteMemoryStore', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let embedder: MockEmbedder;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sqlite-memory-test-'));
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

  // ── store() and storeBatch() ──────────────────────────────────────

  describe('store() and storeBatch()', () => {
    it('stores a single memory and it appears in search results', async () => {
      await store.store({
        content: 'Unique keyword xyzzy for test retrieval',
        category: 'learning',
        tags: ['test'],
      });

      const results = await store.search('xyzzy', 'keyword', {});
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.content).toContain('xyzzy');
    });

    it('storeBatch stores multiple memories', async () => {
      await store.storeBatch([
        { content: 'Batch memory alpha', category: 'learning', tags: [] },
        { content: 'Batch memory beta', category: 'other', tags: [] },
        { content: 'Batch memory gamma', category: 'architecture', tags: [] },
      ]);

      const stats = await store.stats();
      expect(stats.totalMemories).toBe(3);
    });

    it('duplicate id on store() upserts without error', async () => {
      const memory = await store.store({
        content: 'Original content',
        category: 'learning',
        tags: [],
      });

      // Re-store same id — should not throw
      await expect(
        store.store({ content: 'Updated via upsert', category: 'other', tags: [], id: memory.id } as any),
      ).resolves.toBeDefined();

      const stats = await store.stats();
      expect(stats.totalMemories).toBe(1);
    });
  });

  // ── search() modes ────────────────────────────────────────────────

  describe('search() modes', () => {
    beforeEach(async () => {
      await store.store({
        content: 'The quick brown fox jumps over the lazy dog',
        category: 'learning',
        tags: ['animals'],
      });
    });

    it('keyword mode returns memory containing the searched word', async () => {
      const results = await store.search('fox', 'keyword', {});
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.content).toContain('fox');
    });

    it('semantic mode returns results via embedding similarity', async () => {
      const results = await store.search('fox', 'semantic', {});
      expect(results.length).toBeGreaterThan(0);
    });

    it('hybrid mode returns results', async () => {
      const results = await store.search('fox', 'hybrid', {});
      expect(results.length).toBeGreaterThan(0);
    });

    it('empty query returns results or empty array without error', async () => {
      const results = await store.search('', 'keyword', {});
      expect(Array.isArray(results)).toBe(true);
    });

    it('respects limit filter', async () => {
      await store.storeBatch([
        { content: 'fox one', category: 'other', tags: [] },
        { content: 'fox two', category: 'other', tags: [] },
        { content: 'fox three', category: 'other', tags: [] },
      ]);
      const results = await store.search('fox', 'keyword', { limit: 2 });
      expect(results.length).toBeLessThanOrEqual(2);
    });
  });

  // ── update() OCC ──────────────────────────────────────────────────

  describe('update() OCC', () => {
    it('updates a stored memory and increments version', async () => {
      const memory = await store.store({
        content: 'Original content',
        category: 'learning',
        tags: [],
      });
      expect(memory.version).toBe(1);

      const updated = await store.update(memory.id, { content: 'Updated content' });
      expect(updated.version).toBe(2);
      expect(updated.content).toBe('Updated content');
    });

    it('throws MemoryNotFoundError when updating missing id', async () => {
      await expect(
        store.update('nonexistent-id', { content: 'new' }),
      ).rejects.toThrow(MemoryNotFoundError);
    });

    it('throws ConcurrentUpdateError when version mismatch via updateWithVersion', async () => {
      const memory = await store.store({
        content: 'Concurrent content',
        category: 'learning',
        tags: [],
      });

      // Update memory to version 2
      await store.update(memory.id, { content: 'First update' });

      // Now attempt update with stale version (1) using the version-aware method
      await expect(
        store.updateWithVersion(memory.id, { content: 'Stale update' }, 1),
      ).rejects.toThrow(ConcurrentUpdateError);
    });
  });

  // ── delete() ─────────────────────────────────────────────────────

  describe('delete()', () => {
    it('deletes existing memory and it no longer appears in search', async () => {
      const memory = await store.store({
        content: 'Memory to be deleted uniqueword12345',
        category: 'other',
        tags: [],
      });

      await store.delete(memory.id);

      const results = await store.search('uniqueword12345', 'keyword', {});
      expect(results.find(r => r.memory.id === memory.id)).toBeUndefined();
    });

    it('throws MemoryNotFoundError when deleting non-existing id', async () => {
      await expect(store.delete('nonexistent-id')).rejects.toThrow(MemoryNotFoundError);
    });
  });

  // ── listRecent() ─────────────────────────────────────────────────

  describe('listRecent()', () => {
    it('returns memories ordered by created_at descending', async () => {
      await store.store({ content: 'First memory', category: 'learning', tags: [] });
      // Small delay to ensure different timestamps
      await new Promise(r => setTimeout(r, 5));
      await store.store({ content: 'Second memory', category: 'learning', tags: [] });
      await new Promise(r => setTimeout(r, 5));
      await store.store({ content: 'Third memory', category: 'learning', tags: [] });

      const recent = await store.listRecent(3);
      expect(recent[0].content).toBe('Third memory');
      expect(recent[1].content).toBe('Second memory');
      expect(recent[2].content).toBe('First memory');
    });

    it('respects limit parameter', async () => {
      await store.storeBatch([
        { content: 'Memory A', category: 'other', tags: [] },
        { content: 'Memory B', category: 'other', tags: [] },
        { content: 'Memory C', category: 'other', tags: [] },
        { content: 'Memory D', category: 'other', tags: [] },
      ]);

      const recent = await store.listRecent(2);
      expect(recent.length).toBe(2);
    });
  });

  // ── stats() ───────────────────────────────────────────────────────

  describe('stats()', () => {
    it('returns correct total and byCategory counts', async () => {
      await store.storeBatch([
        { content: 'Code memory', category: 'code-solution', tags: [] },
        { content: 'Bug fix', category: 'bug-fix', tags: [] },
        { content: 'Another code', category: 'code-solution', tags: [] },
      ]);

      const stats = await store.stats();
      expect(stats.totalMemories).toBe(3);
      expect(stats.byCategory['code-solution']).toBe(2);
      expect(stats.byCategory['bug-fix']).toBe(1);
    });

    it('returns zero stats when empty', async () => {
      const stats = await store.stats();
      expect(stats.totalMemories).toBe(0);
      expect(stats.byCategory).toEqual({});
      expect(stats.oldestMemory).toBeNull();
      expect(stats.newestMemory).toBeNull();
    });
  });

  // ── prune() ───────────────────────────────────────────────────────

  describe('prune()', () => {
    it('dry run returns candidates without deleting', async () => {
      await store.store({ content: 'Potentially prunable memory', category: 'other', tags: [] });

      const result = await store.prune({ dryRun: true, minStrength: 0.99 });
      expect(result.dryRun).toBe(true);
      expect(result.inspected).toBeGreaterThan(0);
    });

    it('prune with low threshold removes old memories when not dry run', async () => {
      // Store a memory
      await store.store({ content: 'Old memory to prune', category: 'other', tags: [] });

      // Prune with impossibly high minStrength so everything gets pruned
      const result = await store.prune({ dryRun: false, minStrength: 1.0 });
      expect(result.pruned).toBeGreaterThan(0);

      const stats = await store.stats();
      expect(stats.totalMemories).toBe(0);
    });
  });

  // ── findRelated() ─────────────────────────────────────────────────

  describe('findRelated()', () => {
    it('returns related memories not including self', async () => {
      await store.storeBatch([
        { content: 'Memory about cats and animals', category: 'learning', tags: [] },
        { content: 'Memory about dogs and animals', category: 'learning', tags: [] },
        { content: 'Memory about fish and animals', category: 'learning', tags: [] },
      ]);

      const all = await store.listRecent(10);
      const targetId = all[0].id;

      const related = await store.findRelated(targetId, 5);
      expect(related.length).toBeGreaterThan(0);
      expect(related.every(r => r.memory.id !== targetId)).toBe(true);
    });

    it('throws MemoryNotFoundError when id not found', async () => {
      await expect(store.findRelated('nonexistent-id', 5)).rejects.toThrow(MemoryNotFoundError);
    });
  });

  // ── Concurrency ───────────────────────────────────────────────────

  describe('Concurrency', () => {
    it('two instances writing to same db do not corrupt data', async () => {
      const store2 = new SqliteMemoryStore(dbPath, new MockEmbedder());
      await store2.initialize();

      try {
        await Promise.all([
          store.store({ id: 'c1', content: 'from store1', category: 'other', tags: [] } as any),
          store2.store({ id: 'c2', content: 'from store2', category: 'other', tags: [] } as any),
        ]);

        const stats = await store.stats();
        expect(stats.totalMemories).toBeGreaterThanOrEqual(2);
      } finally {
        store2.close();
      }
    });
  });
});
