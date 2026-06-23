import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';

// ── Helpers ──────────────────────────────────────────────────────────

function tableColumns(db: DatabaseSync, table: string): string[] {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return rows.map(r => r.name);
}

function indexExists(db: DatabaseSync, indexName: string): boolean {
  const row = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
  ).get(indexName);
  // indexes live in sqlite_master too
  const idxRow = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`,
  ).get(indexName);
  return !!row || !!idxRow;
}

function userVersion(db: DatabaseSync): number {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number } | undefined;
  return row?.user_version ?? 0;
}

// ── T-001: Migration v2 and status schema ────────────────────────────

describe('T-001: migration v2 and status schema', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr1-migration-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('bumps user_version to 7 after initialize', () => {
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(7);
    db.close();
  });

  it('adds status, review_reason, reviewed_at columns to memories', () => {
    const db = new DatabaseSync(dbPath);
    const cols = tableColumns(db, 'memories');
    expect(cols).toEqual(expect.arrayContaining(['status', 'review_reason', 'reviewed_at']));
    db.close();
  });

  it('defaults status to "active" for a newly stored memory', async () => {
    await store.store({ content: 'status default check', category: 'learning', tags: [] });
    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT status FROM memories WHERE content = ?`).get('status default check') as { status: string };
    expect(row.status).toBe('active');
    db.close();
  });

  const newTables: Array<[string, string[]]> = [
    ['session_prompts', ['id', 'session_id', 'raw_prompt', 'created_at', 'embedding']],
    ['session_responses', ['id', 'session_id', 'prompt_id', 'full_response', 'created_at', 'embedding']],
    ['session_tool_calls', ['id', 'session_id', 'prompt_id', 'response_id', 'tool_name', 'args_safe', 'result_summary', 'importance', 'created_at', 'embedding']],
    ['session_summaries', ['id', 'session_id', 'prompt_id', 'response_id', 'summary', 'decisions', 'key_titles', 'memory_ids', 'artifact_refs', 'created_at', 'embedding']],
    ['session_memory_links', ['prompt_id', 'response_id', 'memory_id', 'relation', 'created_at']],
    ['memory_topics', ['topic_key', 'memory_id', 'created_at', 'updated_at']],
  ];

  for (const [table, expectedCols] of newTables) {
    it(`creates table ${table} with expected columns`, () => {
      const db = new DatabaseSync(dbPath);
      const cols = tableColumns(db, table);
      for (const c of expectedCols) {
        expect(cols).toContain(c);
      }
      db.close();
    });
  }

  const newIndexes = [
    'idx_session_prompts_session_id',
    'idx_session_responses_session_id',
    'idx_session_responses_prompt_id',
    'idx_session_tool_calls_session_id',
    'idx_session_summaries_session_id',
    'idx_memory_topics_memory_id',
  ];

  for (const idx of newIndexes) {
    it(`creates index ${idx}`, () => {
      const db = new DatabaseSync(dbPath);
      expect(indexExists(db, idx)).toBe(true);
      db.close();
    });
  }

  it('re-initialization is idempotent (user_version stays 7, no error)', async () => {
    await expect(store.initialize()).resolves.toBeUndefined();
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(7);
    db.close();
  });

  it('upgrades an existing v1 database to v2', async () => {
    // Close the shared beforeEach store and clear its tmpdir so we can
    // build a pristine v1 database from scratch.
    store.close();
    await rm(tmpDir, { recursive: true, force: true });

    const v1Dir = await mkdtemp(join(tmpdir(), 'amm-pr1-v1upgrade-'));
    const v1Path = join(v1Dir, 'test.db');

    const v1db = new DatabaseSync(v1Path);
    v1db.exec(`
      CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        tags TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed_at TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        origin TEXT NOT NULL DEFAULT 'unknown',
        source_tool TEXT,
        session_id TEXT,
        content_hash TEXT
      );
      CREATE VIRTUAL TABLE memories_fts USING fts5(
        id UNINDEXED, content, tags,
        tokenize='unicode61', content='memories', content_rowid='rowid'
      );
      CREATE TABLE memory_embeddings (
        id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
        embedding BLOB NOT NULL
      );
      CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
      END;
      CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
      END;
      CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
        INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
      END;
      PRAGMA user_version = 1;
    `);
    v1db.close();

    // Initialize against the v1 db — should migrate all the way to v7.
    const s = new SqliteMemoryStore(v1Path, new MockEmbedder());
    await s.initialize();
    s.close();

    // Inspect after the store handle is released (Windows-friendly).
    const check = new DatabaseSync(v1Path);
    expect(userVersion(check)).toBe(7);
    const cols = tableColumns(check, 'memories');
    expect(cols).toEqual(expect.arrayContaining(['status', 'review_reason', 'reviewed_at']));
    expect(tableColumns(check, 'session_prompts').length).toBeGreaterThan(0);
    expect(tableColumns(check, 'memory_topics').length).toBeGreaterThan(0);
    check.close();

    await rm(v1Dir, { recursive: true, force: true });
  });
});

// ── T-002: Upsert fix and types ──────────────────────────────────────

describe('T-002: store() upsert fix', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr1-upsert-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('preserves created_at on re-store with same id', async () => {
    const first = await store.store({
      id: 'upsert-1',
      content: 'Original content',
      category: 'learning',
      tags: ['a'],
    } as any);

    // Capture original created_at
    const originalCreatedAt = first.createdAt;

    // Wait to ensure "now" would differ if it were overwritten
    await new Promise(r => setTimeout(r, 20));

    await store.store({
      id: 'upsert-1',
      content: 'Updated content',
      category: 'other',
      tags: ['b'],
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT created_at, updated_at FROM memories WHERE id = ?`).get('upsert-1') as { created_at: string; updated_at: string };
    db.close();

    expect(row.created_at).toBe(originalCreatedAt);
    // updated_at should have moved forward
    expect(row.updated_at).not.toBe(originalCreatedAt);
  });

  it('increments version on re-store (does not reset to 1)', async () => {
    const first = await store.store({
      id: 'upsert-2',
      content: 'v1 content',
      category: 'learning',
      tags: [],
    } as any);
    expect(first.version).toBe(1);

    await store.store({
      id: 'upsert-2',
      content: 'v2 content',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT version FROM memories WHERE id = ?`).get('upsert-2') as { version: number };
    db.close();

    expect(row.version).toBe(2);
  });

  it('merges provenance via COALESCE (keeps old provenance when new is absent)', async () => {
    // First store with full provenance
    await store.store({
      id: 'upsert-3',
      content: 'With provenance',
      category: 'learning',
      tags: [],
      provenance: {
        origin: 'inferred',
        sourceTool: 'pmc-enrichment/ollama',
        sessionId: 'run-orig',
        contentHash: 'hash-orig',
      },
    } as any);

    // Re-store WITHOUT provenance — old provenance must survive
    await store.store({
      id: 'upsert-3',
      content: 'Updated content',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(
      `SELECT origin, source_tool, session_id, content_hash FROM memories WHERE id = ?`,
    ).get('upsert-3') as { origin: string; source_tool: string; session_id: string; content_hash: string };
    db.close();

    expect(row.origin).toBe('inferred');
    expect(row.source_tool).toBe('pmc-enrichment/ollama');
    expect(row.session_id).toBe('run-orig');
    expect(row.content_hash).toBe('hash-orig');
  });

  it('merges provenance (new explicit values override old)', async () => {
    await store.store({
      id: 'upsert-4',
      content: 'orig',
      category: 'learning',
      tags: [],
      provenance: { origin: 'manual', sessionId: 'ses-old' },
    } as any);

    // Re-store with new sessionId and contentHash; origin stays from old, sourceTool newly set
    await store.store({
      id: 'upsert-4',
      content: 'updated',
      category: 'learning',
      tags: [],
      provenance: { origin: 'inferred', sessionId: 'ses-new', sourceTool: 'tool-x' },
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(
      `SELECT origin, source_tool, session_id, content_hash FROM memories WHERE id = ?`,
    ).get('upsert-4') as { origin: string; source_tool: string | null; session_id: string; content_hash: string | null };
    db.close();

    expect(row.origin).toBe('inferred');
    expect(row.source_tool).toBe('tool-x');
    expect(row.session_id).toBe('ses-new');
  });

  it('does NOT downgrade status when re-store omits status', async () => {
    // First store and then mark needs_review via raw SQL (updateMemoryStatus is PR-2)
    await store.store({
      id: 'upsert-5',
      content: 'needs review',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    db.prepare(`UPDATE memories SET status = 'needs_review' WHERE id = ?`).run('upsert-5');
    db.close();

    // Legacy caller re-stores without status — must NOT downgrade to active
    await store.store({
      id: 'upsert-5',
      content: 're-stored by legacy caller',
      category: 'learning',
      tags: [],
    } as any);

    const db2 = new DatabaseSync(dbPath);
    const row = db2.prepare(`SELECT status FROM memories WHERE id = ?`).get('upsert-5') as { status: string };
    db2.close();

    expect(row.status).toBe('needs_review');
  });

  it('allows explicit status change on re-store', async () => {
    await store.store({
      id: 'upsert-6',
      content: 'will be archived',
      category: 'learning',
      tags: [],
      status: 'archived',
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT status FROM memories WHERE id = ?`).get('upsert-6') as { status: string };
    db.close();

    expect(row.status).toBe('archived');
  });

  it('preserves status when re-store omits status (default active)', async () => {
    // Store active memory
    await store.store({
      id: 'upsert-7',
      content: 'active one',
      category: 'learning',
      tags: [],
    } as any);

    // Re-store without status — should remain active (not error)
    await store.store({
      id: 'upsert-7',
      content: 'still active',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT status FROM memories WHERE id = ?`).get('upsert-7') as { status: string };
    db.close();

    expect(row.status).toBe('active');
  });
});

// ── T-002: topic alias wiring in store() ─────────────────────────────

describe('T-002: store() topicKey wiring', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr1-topic-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('creates a topic alias when store() is called with topicKey', async () => {
    const mem = await store.store({
      id: 'topic-1',
      content: 'topic aliased memory',
      category: 'architecture',
      tags: ['pmc'],
      topicKey: 'architecture/auth-model',
    } as any);

    // resolveTopic should return the same memory
    const resolved = await store.resolveTopic('architecture/auth-model');
    expect(resolved).not.toBeNull();
    expect(resolved!.id).toBe(mem.id);
  });

  it('upserts topic alias on re-store (updates the same topic_key)', async () => {
    await store.store({
      id: 'topic-old',
      content: 'first version',
      category: 'learning',
      tags: [],
      topicKey: 'topic/evolving',
    } as any);

    await store.store({
      id: 'topic-new',
      content: 'second version',
      category: 'learning',
      tags: [],
      topicKey: 'topic/evolving',
    } as any);

    const resolved = await store.resolveTopic('topic/evolving');
    expect(resolved).not.toBeNull();
    expect(resolved!.id).toBe('topic-new');
  });

  it('resolveTopic returns null for unknown topic', async () => {
    const resolved = await store.resolveTopic('does/not/exist');
    expect(resolved).toBeNull();
  });
});

// ── T-001/T-002: status filtering in search ──────────────────────────

describe('T-001/T-002: status filter in search (archived excluded by default)', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr1-status-filter-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    // Store an active memory
    await store.store({
      id: 'mem-active',
      content: 'shared keyword zzz-active',
      category: 'learning',
      tags: [],
    } as any);

    // Store a memory then archive it via raw SQL (updateMemoryStatus is PR-2)
    await store.store({
      id: 'mem-archived',
      content: 'shared keyword zzz-archived',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    db.prepare(`UPDATE memories SET status = 'archived' WHERE id = ?`).run('mem-archived');
    db.close();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('excludes archived memories by default', async () => {
    const results = await store.search('zzz', 'keyword', {});
    const ids = results.map(r => r.memory.id);
    expect(ids).toContain('mem-active');
    expect(ids).not.toContain('mem-archived');
  });

  it('includes archived memories when status: "all"', async () => {
    const results = await store.search('zzz', 'keyword', { status: 'all' } as any);
    const ids = results.map(r => r.memory.id);
    expect(ids).toContain('mem-active');
    expect(ids).toContain('mem-archived');
  });

  it('filters to only archived when status: "archived"', async () => {
    const results = await store.search('zzz', 'keyword', { status: 'archived' } as any);
    const ids = results.map(r => r.memory.id);
    expect(ids).not.toContain('mem-active');
    expect(ids).toContain('mem-archived');
  });
});

// ── T-002: Memory interface surfaces new lifecycle fields ────────────

describe('T-002: Memory lifecycle fields round-trip', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr1-lifecycle-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('search result memory includes status field', async () => {
    await store.store({
      id: 'lf-1',
      content: 'lifecycle roundtrip keyword qqq',
      category: 'learning',
      tags: [],
    } as any);

    const db = new DatabaseSync(dbPath);
    db.prepare(`UPDATE memories SET status = 'needs_review', review_reason = 'stale' WHERE id = ?`).run('lf-1');
    db.close();

    const results = await store.search('qqq', 'keyword', { status: 'all' } as any);
    const mem = results.find(r => r.memory.id === 'lf-1')?.memory;
    expect(mem).toBeDefined();
    expect(mem!.status).toBe('needs_review');
    expect(mem!.reviewReason).toBe('stale');
  });
});
