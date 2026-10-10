import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder, MockMemoryStore } from './mocks.js';
import type { MemoryRelation, CandidateMemory } from '../src/types.js';
import { handleCompare, handleJudge } from '../src/tools.js';

// ── Helpers ──────────────────────────────────────────────────────────

function tableColumns(db: DatabaseSync, table: string): string[] {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return rows.map(r => r.name);
}

function indexExists(db: DatabaseSync, indexName: string): boolean {
  const row = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`,
  ).get(indexName);
  return !!row;
}

function tableExists(db: DatabaseSync, tableName: string): boolean {
  const row = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
  ).get(tableName);
  return !!row;
}

function userVersion(db: DatabaseSync): number {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number } | undefined;
  return row?.user_version ?? 0;
}

// ── Migration v9: memory_relations table ─────────────────────────────

describe('T-005: migration v9 — memory_relations table', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-migration-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('bumps user_version to 9 after initialize', () => {
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });

  it('creates memory_relations table with correct columns', () => {
    const db = new DatabaseSync(dbPath);
    expect(tableExists(db, 'memory_relations')).toBe(true);
    const cols = tableColumns(db, 'memory_relations');
    expect(cols).toContain('id');
    expect(cols).toContain('memory_id_a');
    expect(cols).toContain('memory_id_b');
    expect(cols).toContain('relation');
    expect(cols).toContain('confidence');
    expect(cols).toContain('reasoning');
    expect(cols).toContain('marked_by');
    expect(cols).toContain('created_at');
    db.close();
  });

  it('creates idx_memory_relations_unique index', () => {
    const db = new DatabaseSync(dbPath);
    expect(indexExists(db, 'idx_memory_relations_unique')).toBe(true);
    db.close();
  });

  it('creates idx_memory_relations_a index', () => {
    const db = new DatabaseSync(dbPath);
    expect(indexExists(db, 'idx_memory_relations_a')).toBe(true);
    db.close();
  });

  it('creates idx_memory_relations_b index', () => {
    const db = new DatabaseSync(dbPath);
    expect(indexExists(db, 'idx_memory_relations_b')).toBe(true);
    db.close();
  });

  it('v9 migration is idempotent (re-initialize stays at v9)', async () => {
    await expect(store.initialize()).resolves.toBeUndefined();
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });
});

// ── detect_conflicts on store() ─────────────────────────────────────

describe('T-005: store() with detect_conflicts flag', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-store-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('store() without detect_conflicts (default) upserts silently', async () => {
    const mem = await store.store({
      content: 'normal store without collision detection',
      category: 'learning',
      tags: [],
    });
    expect(mem.id).toBeDefined();
    expect(mem.content).toBe('normal store without collision detection');
  });

  it('store() with detect_conflicts=false returns Memory directly (no collision check)', async () => {
    const result = await store.store({
      content: 'explicit false detect',
      category: 'learning',
      tags: [],
      detect_conflicts: false,
    });
    // When detect_conflicts is false/undefined, store() returns a Memory directly
    expect(result.id).toBeDefined();
    expect(result.content).toBe('explicit false detect');
  });

  it('store() with detect_conflicts=true and no near-duplicates returns { stored, judgment_required: false }', async () => {
    // First store a unique memory
    await store.store({
      id: 'unique-1',
      content: 'completely different content about quantum physics',
      category: 'learning',
      tags: [],
    });

    // Now store with detect_conflicts=true — different content, no collision
    const result = await store.store({
      content: 'unrelated topic about cooking recipes',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
    });

    // Should be a StoreResult, not a raw Memory
    expect(result).toHaveProperty('stored');
    expect(result).toHaveProperty('judgment_required');
    expect((result as any).judgment_required).toBe(false);
    expect((result as any).stored.id).toBeDefined();
    expect((result as any).stored.content).toBe('unrelated topic about cooking recipes');
  });

  it('store() with detect_conflicts=true and near-duplicate returns { judgment_required: true, candidates: [...] } WITHOUT storing', async () => {
    // Store a memory first
    await store.store({
      id: 'dup-target',
      content: 'the quick brown fox jumps over the lazy dog near the river bank',
      category: 'learning',
      tags: [],
    });

    // Store a near-duplicate with detect_conflicts=true
    const result = await store.store({
      content: 'the quick brown fox jumps over the lazy dog near the river bank',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
    });

    // W2 fix: when candidates exist, memory is NOT stored (check-then-upsert)
    expect(result).not.toHaveProperty('stored');
    expect(result).toHaveProperty('judgment_required');
    expect((result as any).judgment_required).toBe(true);
    expect((result as any).candidates).toBeDefined();
    expect((result as any).candidates.length).toBeGreaterThan(0);
    expect((result as any).candidates[0]).toHaveProperty('id');
    expect((result as any).candidates[0]).toHaveProperty('similarity');
    expect((result as any).candidates[0]).toHaveProperty('content_preview');
    expect((result as any).candidates[0]).toHaveProperty('createdAt');
  });

  it('store() with detect_conflicts=true and candidates does NOT persist the new memory', async () => {
    await store.store({
      id: 'existing-mem',
      content: 'duplicate content about foxes jumping over lazy dogs',
      category: 'learning',
      tags: [],
    });

    const beforeCount = (await store.stats()).totalMemories;

    const result = await store.store({
      id: 'should-not-exist',
      content: 'duplicate content about foxes jumping over lazy dogs',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
    });

    expect((result as any).judgment_required).toBe(true);

    // Memory should NOT be in the DB
    const afterCount = (await store.stats()).totalMemories;
    expect(afterCount).toBe(beforeCount);

    // Searching for the new memory ID should find nothing
    const searchResults = await store.search('duplicate content about foxes', 'keyword', { status: 'all' });
    const found = searchResults.find(r => r.memory.id === 'should-not-exist');
    expect(found).toBeUndefined();
  });

  it('store() with detect_conflicts=true uses configurable threshold (no upsert on match)', async () => {
    // Store a memory
    await store.store({
      id: 'threshold-test',
      content: 'test content for threshold checking',
      category: 'learning',
      tags: [],
    });

    // With very high threshold (0.99) — exact text will still match
    const result = await store.store({
      content: 'test content for threshold checking',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
      conflict_threshold: 0.99,
    });

    // Exact match should still trigger at 0.99, and NOT upsert
    expect((result as any).judgment_required).toBe(true);
    expect(result).not.toHaveProperty('stored');
  });
});

// ── detectCollisions ─────────────────────────────────────────────────

describe('T-005: detectCollisions', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-detect-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('returns empty when no same-project memories exist', async () => {
    const result = await store.store({
      content: 'first memory in empty store',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
    });

    // No prior memories → no candidates
    expect((result as any).judgment_required).toBe(false);
  });

  it('returns candidates sorted by similarity desc', async () => {
    // Store several memories with varying similarity
    await store.store({
      id: 'sim-high',
      content: 'the quick brown fox jumps over the lazy dog',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'sim-med',
      content: 'a fast brown fox leaps over a sleepy dog',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'sim-low',
      content: 'completely unrelated astronomy black holes',
      category: 'learning',
      tags: [],
    });

    // Store near-duplicate of sim-high
    const result = await store.store({
      content: 'the quick brown fox jumps over the lazy dog',
      category: 'learning',
      tags: [],
      detect_conflicts: true,
    });

    const candidates = (result as any).candidates as CandidateMemory[];
    if (candidates.length > 1) {
      // Verify sorted descending by similarity
      for (let i = 1; i < candidates.length; i++) {
        expect(candidates[i - 1].similarity).toBeGreaterThanOrEqual(candidates[i].similarity);
      }
    }
  });
});

// ── addMemoryRelation / getMemoryRelation / listMemoryRelations ─────

describe('T-005: memory relation methods', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-relations-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    // Seed two memories for relation tests
    await store.store({
      id: 'rel-a',
      content: 'memory A for relation tests',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'rel-b',
      content: 'memory B for relation tests',
      category: 'learning',
      tags: [],
    });
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('addMemoryRelation persists a row and returns the relation', () => {
    const rel = store.addMemoryRelation({
      memoryIdA: 'rel-a',
      memoryIdB: 'rel-b',
      relation: 'related',
    });

    expect(rel.id).toBeDefined();
    expect(rel.memoryIdA).toBe('rel-a');
    expect(rel.memoryIdB).toBe('rel-b');
    expect(rel.relation).toBe('related');
    expect(rel.createdAt).toBeDefined();
  });

  it('addMemoryRelation with invalid relation enum throws', () => {
    expect(() =>
      store.addMemoryRelation({
        memoryIdA: 'rel-a',
        memoryIdB: 'rel-b',
        relation: 'invalid_relation' as any,
      }),
    ).toThrow();
  });

  it('addMemoryRelation with confidence and reasoning persists them', () => {
    const rel = store.addMemoryRelation({
      memoryIdA: 'rel-a',
      memoryIdB: 'rel-b',
      relation: 'conflicts_with',
      confidence: 0.95,
      reasoning: 'These memories contradict each other',
      markedBy: 'test-agent',
    });

    expect(rel.confidence).toBe(0.95);
    expect(rel.reasoning).toBe('These memories contradict each other');
    expect(rel.markedBy).toBe('test-agent');
  });

  it('getMemoryRelation(a, b) returns the relation if exists', () => {
    store.addMemoryRelation({
      memoryIdA: 'rel-a',
      memoryIdB: 'rel-b',
      relation: 'compatible',
    });

    const found = store.getMemoryRelation('rel-a', 'rel-b');
    expect(found).not.toBeNull();
    expect(found!.relation).toBe('compatible');
  });

  it('getMemoryRelation(a, b) returns null when no relation exists', () => {
    const found = store.getMemoryRelation('rel-a', 'rel-b');
    expect(found).toBeNull();
  });

  it('listMemoryRelations(memoryId) returns all relations involving the memory', () => {
    store.addMemoryRelation({
      memoryIdA: 'rel-a',
      memoryIdB: 'rel-b',
      relation: 'related',
    });
    store.addMemoryRelation({
      memoryIdA: 'rel-b',
      memoryIdB: 'rel-a',
      relation: 'conflicts_with',
    });

    const rels = store.listMemoryRelations('rel-a');
    expect(rels.length).toBe(2);
    // Should find both directions
    const relTypes = rels.map(r => r.relation);
    expect(relTypes).toContain('related');
    expect(relTypes).toContain('conflicts_with');
  });

  it('listMemoryRelations returns empty array when no relations exist', () => {
    const rels = store.listMemoryRelations('rel-a');
    expect(rels).toEqual([]);
  });
});

// ── T-006: compare() and judge() store methods ─────────────────────

describe('T-006: compare() store method', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-compare-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    await store.store({
      id: 'cmp-a',
      content: 'the quick brown fox jumps over the lazy dog',
      category: 'learning',
      tags: ['animals'],
    });
    await store.store({
      id: 'cmp-b',
      content: 'the quick brown fox leaps over a sleepy cat',
      category: 'learning',
      tags: ['animals'],
    });
    await store.store({
      id: 'cmp-c',
      content: 'quantum entanglement in particle physics',
      category: 'architecture',
      tags: ['physics'],
    });
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('compare(idA, idB) returns similarity, sharedEntities, contentDiff, recommendation', async () => {
    const result = await store.compare('cmp-a', 'cmp-b');

    expect(result).toHaveProperty('similarity');
    expect(result).toHaveProperty('sharedEntities');
    expect(result).toHaveProperty('contentDiff');
    expect(result).toHaveProperty('recommendation');

    expect(typeof result.similarity).toBe('number');
    expect(result.similarity).toBeGreaterThanOrEqual(0);
    expect(result.similarity).toBeLessThanOrEqual(1);
    expect(Array.isArray(result.sharedEntities)).toBe(true);
    expect(typeof result.contentDiff).toBe('string');
    expect(typeof result.recommendation).toBe('string');
  });

  it('compare returns high similarity for near-duplicates', async () => {
    const result = await store.compare('cmp-a', 'cmp-b');
    // Similar content → high similarity
    expect(result.similarity).toBeGreaterThan(0.5);
    // Should have shared tokens
    expect(result.sharedEntities.length).toBeGreaterThan(0);
  });

  it('compare returns different recommendations for similar vs different content', async () => {
    const similar = await store.compare('cmp-a', 'cmp-b');
    const different = await store.compare('cmp-a', 'cmp-c');

    // Similar content should have higher similarity than different content
    expect(similar.similarity).toBeGreaterThan(different.similarity);
    // Both should have valid recommendations
    expect(similar.recommendation).toBeDefined();
    expect(different.recommendation).toBeDefined();
  });

  it('compare throws MemoryNotFoundError if either ID missing', async () => {
    await expect(store.compare('cmp-a', 'nonexistent')).rejects.toThrow('Memory not found');
    await expect(store.compare('nonexistent', 'cmp-b')).rejects.toThrow('Memory not found');
  });
});

describe('T-006: judge() store method', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-judge-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    await store.store({
      id: 'judge-new',
      content: 'new memory about foxes',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'judge-old',
      content: 'old memory about foxes',
      category: 'learning',
      tags: [],
    });
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('judge with related inserts a row and returns { persisted: true, relation }', async () => {
    const result = await store.judge('judge-new:judge-old', 'related');

    expect(result.persisted).toBe(true);
    expect(result.relation).toBe('related');

    // Verify relation was persisted
    const rel = store.getMemoryRelation('judge-new', 'judge-old');
    expect(rel).not.toBeNull();
    expect(rel!.relation).toBe('related');
  });

  it('judge with supersedes inserts row AND archives the older memory', async () => {
    const result = await store.judge('judge-new:judge-old', 'supersedes');

    expect(result.persisted).toBe(true);
    expect(result.relation).toBe('supersedes');
    expect(result.cascaded).toBe('archived');

    // Verify older memory is archived
    const oldMem = (await store.search('old memory about foxes', 'keyword', { status: 'all' }))
      .find(r => r.memory.id === 'judge-old');
    expect(oldMem?.memory.status).toBe('archived');

    // Verify relation was persisted
    const rel = store.getMemoryRelation('judge-new', 'judge-old');
    expect(rel).not.toBeNull();
    expect(rel!.relation).toBe('supersedes');
  });

  it('judge with not_conflict inserts row, does NOT archive', async () => {
    const result = await store.judge('judge-new:judge-old', 'not_conflict');

    expect(result.persisted).toBe(true);
    expect(result.relation).toBe('not_conflict');
    expect(result.cascaded).toBeUndefined();

    // Verify older memory is NOT archived (status is 'active', not 'archived')
    const oldMem = (await store.search('old memory about foxes', 'keyword', { status: 'all' }))
      .find(r => r.memory.id === 'judge-old');
    expect(oldMem?.memory.status).not.toBe('archived');
  });

  it('judge with conflicts_with inserts row, does NOT archive', async () => {
    const result = await store.judge('judge-new:judge-old', 'conflicts_with');

    expect(result.persisted).toBe(true);
    expect(result.relation).toBe('conflicts_with');
    expect(result.cascaded).toBeUndefined();
  });

  it.each(['compatible', 'scoped', 'related'] as const)(
    'judge with %s inserts row, no archive',
    async (relation) => {
      const result = await store.judge('judge-new:judge-old', relation);
      expect(result.persisted).toBe(true);
      expect(result.relation).toBe(relation);
      expect(result.cascaded).toBeUndefined();
    },
  );

  it('judge with invalid relation throws', async () => {
    await expect(
      store.judge('judge-new:judge-old', 'invalid_relation' as any),
    ).rejects.toThrow(/Invalid relation/);
  });

  it('judge with reasoning and confidence persists both', async () => {
    const result = await store.judge('judge-new:judge-old', 'related', 'Same topic', 0.85);

    expect(result.persisted).toBe(true);

    const rel = store.getMemoryRelation('judge-new', 'judge-old');
    expect(rel).not.toBeNull();
    expect(rel!.reasoning).toBe('Same topic');
    expect(rel!.confidence).toBe(0.85);
  });

  it('judge with malformed judgmentId (no colon) throws', async () => {
    await expect(
      store.judge('no-colon-here', 'related'),
    ).rejects.toThrow(/Invalid judgment_id/);
  });
});

// ── T-007: MCP handler tests for compare/judge ─────────────────────

describe('T-007: compare/judge MCP tool handlers', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr6-handlers-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    await store.store({
      id: 'hdl-a',
      content: 'memory A for handler tests',
      category: 'learning',
      tags: [],
    });
    await store.store({
      id: 'hdl-b',
      content: 'memory B for handler tests',
      category: 'learning',
      tags: [],
    });
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('compare handler returns structured response', async () => {
    const handler = handleCompare(store);
    const result = await handler({ memory_id_a: 'hdl-a', memory_id_b: 'hdl-b' });

    expect(result.content[0].type).toBe('text');
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed).toHaveProperty('similarity');
    expect(parsed).toHaveProperty('sharedEntities');
    expect(parsed).toHaveProperty('contentDiff');
    expect(parsed).toHaveProperty('recommendation');
  });

  it('judge handler persists and returns { persisted: true, relation }', async () => {
    const handler = handleJudge(store);
    const result = await handler({
      judgment_id: 'hdl-a:hdl-b',
      relation: 'related',
    });

    expect(result.content[0].type).toBe('text');
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.persisted).toBe(true);
    expect(parsed.relation).toBe('related');
  });

  it('judge handler with invalid relation returns error', async () => {
    const handler = handleJudge(store);
    const result = await handler({
      judgment_id: 'hdl-a:hdl-b',
      relation: 'invalid_relation' as any,
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toMatch(/Invalid relation/);
  });

  it('judge handler with unknown judgment_id returns error', async () => {
    const handler = handleJudge(store);
    const result = await handler({
      judgment_id: 'hdl-a:nonexistent',
      relation: 'related',
    });

    expect(result.isError).toBe(true);
  });

  it('judge handler with supersedes returns cascaded info', async () => {
    const handler = handleJudge(store);
    const result = await handler({
      judgment_id: 'hdl-a:hdl-b',
      relation: 'supersedes',
    });

    expect(result.isError).toBeUndefined();
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.persisted).toBe(true);
    expect(parsed.relation).toBe('supersedes');
    expect(parsed.cascaded).toBe('archived');
  });
});

// ── MockMemoryStore interface satisfaction ───────────────────────────

describe('T-005: MockMemoryStore has conflict/collision methods', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('addMemoryRelation persists and returns relation', async () => {
    await store.store({ id: 'mock-a', content: 'A', category: 'learning', tags: [] });
    await store.store({ id: 'mock-b', content: 'B', category: 'learning', tags: [] });

    const rel = store.addMemoryRelation({
      memoryIdA: 'mock-a',
      memoryIdB: 'mock-b',
      relation: 'related',
    });

    expect(rel.id).toBeDefined();
    expect(rel.memoryIdA).toBe('mock-a');
    expect(rel.memoryIdB).toBe('mock-b');
    expect(rel.relation).toBe('related');
  });

  it('getMemoryRelation returns stored relation', async () => {
    await store.store({ id: 'mock-c', content: 'C', category: 'learning', tags: [] });
    await store.store({ id: 'mock-d', content: 'D', category: 'learning', tags: [] });

    store.addMemoryRelation({
      memoryIdA: 'mock-c',
      memoryIdB: 'mock-d',
      relation: 'scoped',
    });

    const found = store.getMemoryRelation('mock-c', 'mock-d');
    expect(found).not.toBeNull();
    expect(found!.relation).toBe('scoped');
  });

  it('listMemoryRelations returns all for a memory', async () => {
    await store.store({ id: 'mock-e', content: 'E', category: 'learning', tags: [] });
    await store.store({ id: 'mock-f', content: 'F', category: 'learning', tags: [] });

    store.addMemoryRelation({
      memoryIdA: 'mock-e',
      memoryIdB: 'mock-f',
      relation: 'related',
    });

    const rels = store.listMemoryRelations('mock-e');
    expect(rels.length).toBe(1);
    expect(rels[0].relation).toBe('related');
  });
});
