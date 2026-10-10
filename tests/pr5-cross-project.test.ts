import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import type { SearchFilters } from '../src/types.js';

// ── T-003: Cross-project search flag ────────────────────────────────

describe('T-003: cross-project search flag (all_projects)', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr5-crossproj-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('SearchFilters type accepts all_projects boolean', () => {
    // This test verifies the type compiles with all_projects
    const filters: SearchFilters = {
      limit: 10,
      all_projects: true,
    };
    expect(filters.all_projects).toBe(true);
  });

  it('default search (all_projects=false/undefined) returns results from same project', async () => {
    await store.store({
      id: 'proj-a-mem',
      content: 'project alpha knowledge',
      category: 'learning',
      tags: [],
    });

    const results = await store.search('project alpha', 'hybrid', { limit: 10 });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results.some(r => r.memory.id === 'proj-a-mem')).toBe(true);
  });

  it('search with all_projects=true still returns results from current DB', async () => {
    await store.store({
      id: 'proj-b-mem',
      content: 'cross project search test',
      category: 'architecture',
      tags: [],
    });

    const results = await store.search('cross project', 'hybrid', {
      limit: 10,
      all_projects: true,
    });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results.some(r => r.memory.id === 'proj-b-mem')).toBe(true);
  });

  it('filters (category, tags) still apply when all_projects=true', async () => {
    await store.store({
      id: 'cat-filter-1',
      content: 'architecture decision',
      category: 'architecture',
      tags: ['design'],
    });
    await store.store({
      id: 'cat-filter-2',
      content: 'architecture learning',
      category: 'learning',
      tags: ['study'],
    });

    const results = await store.search('architecture', 'hybrid', {
      limit: 10,
      all_projects: true,
      category: 'architecture',
    });
    expect(results.every(r => r.memory.category === 'architecture')).toBe(true);
    expect(results.some(r => r.memory.id === 'cat-filter-1')).toBe(true);
    expect(results.some(r => r.memory.id === 'cat-filter-2')).toBe(false);
  });

  it('limit applies to result set when all_projects=true', async () => {
    for (let i = 0; i < 5; i++) {
      await store.store({
        id: `limit-mem-${i}`,
        content: `limit test memory ${i}`,
        category: 'learning',
        tags: [],
      });
    }

    const results = await store.search('limit test', 'hybrid', {
      limit: 2,
      all_projects: true,
    });
    expect(results.length).toBeLessThanOrEqual(2);
  });

  it('pinned memories sort first even with all_projects=true', async () => {
    await store.store({
      id: 'unpinned-1',
      content: 'pinned sort test unpinned',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'pinned-1',
      content: 'pinned sort test pinned',
      category: 'learning',
      tags: [],
    });
    await store.pin('pinned-1');

    const results = await store.search('pinned sort test', 'hybrid', {
      limit: 10,
      all_projects: true,
    });

    expect(results.length).toBeGreaterThanOrEqual(2);
    const pinnedIdx = results.findIndex(r => r.memory.id === 'pinned-1');
    const unpinnedIdx = results.findIndex(r => r.memory.id === 'unpinned-1');
    expect(pinnedIdx).toBeLessThan(unpinnedIdx);
  });

  it('ORDER BY includes is_pinned DESC (pinned rows have higher raw sort priority)', async () => {
    // Verify the SQL ordering works: insert two memories with same content,
    // pin one, and confirm it appears first.
    await store.store({ id: 'sort-a', content: 'order by test alpha', category: 'learning', tags: [] });
    await store.store({ id: 'sort-b', content: 'order by test beta', category: 'learning', tags: [] });
    await store.pin('sort-a');

    const results = await store.search('order by test', 'hybrid', { limit: 10 });
    expect(results.length).toBe(2);
    expect(results[0].memory.id).toBe('sort-a');
    expect(results[1].memory.id).toBe('sort-b');
  });
});

// ── T-013: Multi-DB enumeration for all_projects=true ──────────────

describe('T-013: multi-DB enumeration (all_projects=true)', () => {
  let storeA: SqliteMemoryStore;
  let storeB: SqliteMemoryStore;
  let dbPathA: string;
  let dbPathB: string;
  let tmpDirA: string;
  let tmpDirB: string;

  beforeEach(async () => {
    tmpDirA = await mkdtemp(join(tmpdir(), 'amm-pr5-multidb-a-'));
    tmpDirB = await mkdtemp(join(tmpdir(), 'amm-pr5-multidb-b-'));
    dbPathA = join(tmpDirA, 'test.db');
    dbPathB = join(tmpDirB, 'test.db');

    storeA = new SqliteMemoryStore(dbPathA, new MockEmbedder());
    storeB = new SqliteMemoryStore(dbPathB, new MockEmbedder());
    await storeA.initialize();
    await storeB.initialize();

    // Wire sibling stores
    storeA.setSiblingStores([storeB]);
    storeB.setSiblingStores([storeA]);
  });

  afterEach(async () => {
    storeA.close();
    storeB.close();
    await rm(tmpDirA, { recursive: true, force: true });
    await rm(tmpDirB, { recursive: true, force: true });
  });

  it('search with all_projects=true returns results from sibling DBs', async () => {
    await storeA.store({
      id: 'db-a-mem',
      content: 'knowledge from project alpha database',
      category: 'learning',
      tags: [],
    });
    await storeB.store({
      id: 'db-b-mem',
      content: 'knowledge from project beta database',
      category: 'architecture',
      tags: [],
    });

    const results = await storeA.search('knowledge from project', 'keyword', {
      limit: 10,
      all_projects: true,
    });

    // Should find memories from both DBs
    expect(results.length).toBe(2);
    const ids = results.map(r => r.memory.id);
    expect(ids).toContain('db-a-mem');
    expect(ids).toContain('db-b-mem');
  });

  it('search without all_projects returns only local results', async () => {
    await storeA.store({
      id: 'local-only',
      content: 'local knowledge alpha',
      category: 'learning',
      tags: [],
    });
    await storeB.store({
      id: 'remote-only',
      content: 'remote knowledge beta',
      category: 'learning',
      tags: [],
    });

    const results = await storeA.search('knowledge', 'keyword', {
      limit: 10,
      // all_projects not set — default false
    });

    // Should only find local memory
    expect(results.length).toBe(1);
    expect(results[0].memory.id).toBe('local-only');
  });

  it('limit applies to merged result set across DBs', async () => {
    // Store 3 memories in each DB
    for (let i = 0; i < 3; i++) {
      await storeA.store({
        id: `db-a-${i}`,
        content: `merged limit test memory ${i} from alpha`,
        category: 'learning',
        tags: [],
      });
      await storeB.store({
        id: `db-b-${i}`,
        content: `merged limit test memory ${i} from beta`,
        category: 'learning',
        tags: [],
      });
    }

    const results = await storeA.search('merged limit test', 'keyword', {
      limit: 3,
      all_projects: true,
    });

    // Limit should apply to the merged set
    expect(results.length).toBeLessThanOrEqual(3);
  });

  it('filters apply to all DBs when all_projects=true', async () => {
    await storeA.store({
      id: 'filter-a',
      content: 'filter test architecture alpha',
      category: 'architecture',
      tags: [],
    });
    await storeA.store({
      id: 'filter-a-learn',
      content: 'filter test learning alpha',
      category: 'learning',
      tags: [],
    });
    await storeB.store({
      id: 'filter-b',
      content: 'filter test architecture beta',
      category: 'architecture',
      tags: [],
    });

    const results = await storeA.search('filter test', 'keyword', {
      limit: 10,
      all_projects: true,
      category: 'architecture',
    });

    // Only architecture memories from both DBs
    expect(results.every(r => r.memory.category === 'architecture')).toBe(true);
    expect(results.length).toBe(2);
    const ids = results.map(r => r.memory.id);
    expect(ids).toContain('filter-a');
    expect(ids).toContain('filter-b');
  });

  it('results are sorted by score across all DBs', async () => {
    await storeA.store({
      id: 'score-a',
      content: 'unique scoring test alpha',
      category: 'learning',
      tags: [],
    });
    await storeB.store({
      id: 'score-b',
      content: 'unique scoring test beta',
      category: 'learning',
      tags: [],
    });

    const results = await storeA.search('unique scoring test', 'keyword', {
      limit: 10,
      all_projects: true,
    });

    expect(results.length).toBe(2);
    // Results should be sorted by score descending
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });
});
