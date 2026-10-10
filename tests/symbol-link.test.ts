import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore, parseSearchQuery } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import type { UnifiedSearchQuery } from '../src/types.js';

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

function userVersion(db: DatabaseSync): number {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number } | undefined;
  return row?.user_version ?? 0;
}

async function makeStore(): Promise<{
  store: SqliteMemoryStore;
  dbPath: string;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-symbol-link-'));
  const dbPath = join(tmpDir, 'test.db');
  const store = new SqliteMemoryStore(dbPath, new MockEmbedder());
  await store.initialize();
  return {
    store,
    dbPath,
    cleanup: async () => {
      store.close();
      await rm(tmpDir, { recursive: true, force: true });
    },
  };
}

const SESSION_TABLES = [
  'session_prompts',
  'session_responses',
  'session_tool_calls',
  'session_summaries',
] as const;

// ── T-v7: Schema migration — symbol_key + project_id columns ─────────

describe('v7 migration: symbol_key and project_id columns', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, dbPath, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('bumps user_version to 8 after initialize (v7 → v8 pin migration)', () => {
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });

  it('adds symbol_key column to memories', () => {
    const db = new DatabaseSync(dbPath);
    expect(tableColumns(db, 'memories')).toContain('symbol_key');
    db.close();
  });

  it('adds project_id and symbol_key columns to all 4 session tables', () => {
    const db = new DatabaseSync(dbPath);
    for (const table of SESSION_TABLES) {
      const cols = tableColumns(db, table);
      expect(cols).toContain('project_id');
      expect(cols).toContain('symbol_key');
    }
    db.close();
  });

  it('creates index on memories.symbol_key', () => {
    const db = new DatabaseSync(dbPath);
    expect(indexExists(db, 'idx_memories_symbol_key')).toBe(true);
    db.close();
  });

  it('creates project_id and symbol_key indexes on all 4 session tables', () => {
    const db = new DatabaseSync(dbPath);
    for (const table of SESSION_TABLES) {
      expect(indexExists(db, `idx_${table}_project_id`)).toBe(true);
      expect(indexExists(db, `idx_${table}_symbol_key`)).toBe(true);
    }
    db.close();
  });

  it('re-initialization is idempotent (user_version stays 7, no error)', async () => {
    await expect(store.initialize()).resolves.toBeUndefined();
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });

  it('symbol_key and project_id columns are nullable (no NOT NULL constraint)', () => {
    const db = new DatabaseSync(dbPath);
    // Insert a memories row with a NULL symbol_key directly — must succeed.
    db.prepare(
      `INSERT INTO memories (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version, origin, status)
       VALUES ('m-null', 'nullable check', 'learning', '[]', '2026-01-01', '2026-01-01', 0, '2026-01-01', 1, 'manual', 'active')`,
    ).run();
    const row = db.prepare(`SELECT symbol_key FROM memories WHERE id = 'm-null'`).get() as { symbol_key: string | null };
    expect(row.symbol_key).toBeNull();
    db.close();
  });
});

// ── T-v7: setSessionContext project_id backfill ──────────────────────

describe('setSessionContext: project_id backfill', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, dbPath, cleanup } = await makeStore());
    // Seed rows in all 4 session tables for one session BEFORE setting context,
    // so the rows start with NULL project_id (legacy state).
    const prompt = await store.storeSessionPrompt('sess-backfill', 'backfill prompt about widgets');
    const response = await store.storeSessionResponse('sess-backfill', prompt.id, 'backfill response about the widget fix');
    await store.storeSessionToolCall({
      sessionId: 'sess-backfill',
      promptId: prompt.id,
      responseId: response.id,
      toolName: 'widget_tool',
      resultSummary: 'widget tool result',
    });
    await store.storeSessionSummary({
      sessionId: 'sess-backfill',
      promptId: prompt.id,
      summary: 'widget session summary',
    });
  });

  afterEach(async () => {
    await cleanup();
  });

  it('backfills project_id on all 4 session tables when setSessionContext is called', () => {
    // Before: all rows have NULL project_id.
    const dbBefore = new DatabaseSync(dbPath);
    for (const table of SESSION_TABLES) {
      const row = dbBefore.prepare(
        `SELECT project_id FROM ${table} WHERE session_id = ? LIMIT 1`,
      ).get('sess-backfill') as { project_id: string | null };
      expect(row.project_id).toBeNull();
    }
    dbBefore.close();

    // Act: set session context — triggers backfill.
    store.setSessionContext('sess-backfill', 'proj-backfill-1');

    // After: all rows for that session now carry project_id.
    const dbAfter = new DatabaseSync(dbPath);
    for (const table of SESSION_TABLES) {
      const row = dbAfter.prepare(
        `SELECT project_id FROM ${table} WHERE session_id = ? LIMIT 1`,
      ).get('sess-backfill') as { project_id: string | null };
      expect(row.project_id).toBe('proj-backfill-1');
    }
    dbAfter.close();
  });

  it('does not overwrite a non-null project_id on backfill (idempotent)', async () => {
    // First backfill sets project_id to proj-A.
    store.setSessionContext('sess-backfill', 'proj-A');

    // A second call with a different project must NOT clobber the existing value
    // (backfill only touches NULL project_id rows).
    store.setSessionContext('sess-backfill', 'proj-B');

    const db = new DatabaseSync(dbPath);
    const row = db.prepare(
      `SELECT project_id FROM session_prompts WHERE session_id = ? LIMIT 1`,
    ).get('sess-backfill') as { project_id: string | null };
    expect(row.project_id).toBe('proj-A');
    db.close();
  });

  it('only backfills rows for the given session_id, not other sessions', async () => {
    // Seed a second session.
    await store.storeSessionPrompt('sess-other', 'other session prompt');

    store.setSessionContext('sess-backfill', 'proj-backfill-1');

    const db = new DatabaseSync(dbPath);
    const other = db.prepare(
      `SELECT project_id FROM session_prompts WHERE session_id = ? LIMIT 1`,
    ).get('sess-other') as { project_id: string | null };
    expect(other.project_id).toBeNull();
    db.close();
  });
});

// ── T-v7: parseSearchQuery enables projectId ─────────────────────────

describe('parseSearchQuery: projectId enablement', () => {
  it('accepts projectId and flows it through to the resolved query', () => {
    const resolved = parseSearchQuery({ query: 'alpha', projectId: 'proj-1' });
    expect(resolved.projectId).toBe('proj-1');
  });

  it('defaults includeLegacy to false when omitted', () => {
    const resolved = parseSearchQuery({ query: 'alpha' });
    expect(resolved.includeLegacy).toBe(false);
  });

  it('honors an explicit includeLegacy: true', () => {
    const resolved = parseSearchQuery({ query: 'alpha', includeLegacy: true });
    expect(resolved.includeLegacy).toBe(true);
  });

  it('still accepts queries without projectId', () => {
    expect(() => parseSearchQuery({ query: 'alpha', sessionId: 'sess-1' })).not.toThrow();
  });
});

// ── T-v7: project-scoped session search (default-deny) ───────────────

describe('searchSessions: project scoping (default-deny)', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
    // Seed two sessions in two different projects.
    // sess-proj-A → project 'proj-A'
    const promptA = await store.storeSessionPrompt('sess-proj-A', 'shared keyword alpha in project A');
    await store.storeSessionSummary({
      sessionId: 'sess-proj-A',
      promptId: promptA.id,
      summary: 'alpha summary for project A',
    });
    store.setSessionContext('sess-proj-A', 'proj-A');

    // sess-proj-B → project 'proj-B'
    const promptB = await store.storeSessionPrompt('sess-proj-B', 'shared keyword alpha in project B');
    await store.storeSessionSummary({
      sessionId: 'sess-proj-B',
      promptId: promptB.id,
      summary: 'alpha summary for project B',
    });
    store.setSessionContext('sess-proj-B', 'proj-B');

    // A legacy session with NULL project_id (never backfilled).
    const promptLegacy = await store.storeSessionPrompt('sess-legacy', 'shared keyword alpha in legacy session');
    await store.storeSessionSummary({
      sessionId: 'sess-legacy',
      promptId: promptLegacy.id,
      summary: 'alpha summary for legacy session with no project',
    });
    // NOTE: no setSessionContext call → project_id stays NULL.
  });

  afterEach(async () => {
    await cleanup();
  });

  it('returns only rows from the requested project (excludes other projects)', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      projectId: 'proj-A',
      includeRawSessionItems: true,
    } as UnifiedSearchQuery);
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.sessionId).toBe('sess-proj-A');
    }
  });

  it('excludes legacy (null project_id) rows by default', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      projectId: 'proj-A',
      includeRawSessionItems: true,
    } as UnifiedSearchQuery);
    for (const r of results) {
      expect(r.sessionId).not.toBe('sess-legacy');
    }
  });

  it('includes legacy (null project_id) rows when includeLegacy is true', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      projectId: 'proj-A',
      includeLegacy: true,
      includeRawSessionItems: true,
    } as UnifiedSearchQuery);
    const sessionIds = new Set(results.map(r => r.sessionId));
    expect(sessionIds.has('sess-proj-A')).toBe(true);
    expect(sessionIds.has('sess-legacy')).toBe(true);
    // But still excludes the other project.
    expect(sessionIds.has('sess-proj-B')).toBe(false);
  });

  it('returns all rows (including all projects) when no projectId is given', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
    });
    const sessionIds = new Set(results.map(r => r.sessionId));
    expect(sessionIds.has('sess-proj-A')).toBe(true);
    expect(sessionIds.has('sess-proj-B')).toBe(true);
    expect(sessionIds.has('sess-legacy')).toBe(true);
  });
});

// ── T-v7: symbol_key surfaces through row converters ─────────────────

describe('row converters: symbol_key and project_id mapping', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, dbPath, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('rowToMemory surfaces symbol_key as symbolKey when the column is populated', async () => {
    // Store a memory through the normal path, then set symbol_key directly
    // (PR #1 only adds the column; attach_symbol write path is PR #2).
    const mem = await store.store({ content: 'symbol link check', category: 'learning', tags: [] });
    const db = new DatabaseSync(dbPath);
    db.prepare(`UPDATE memories SET symbol_key = ? WHERE id = ?`).run(
      'ts|src/foo.ts|function|global|bar|0', mem.id,
    );
    db.close();

    // Read back via a public read path (listRecent) and verify symbolKey.
    const recent = await store.listRecent(10);
    const found = recent.find(m => m.id === mem.id);
    expect(found).toBeDefined();
    expect(found!.symbolKey).toBe('ts|src/foo.ts|function|global|bar|0');
  });

  it('rowToMemory omits symbolKey when the column is NULL', async () => {
    const mem = await store.store({ content: 'no symbol link', category: 'learning', tags: [] });
    const recent = await store.listRecent(10);
    const found = recent.find(m => m.id === mem.id);
    expect(found).toBeDefined();
    expect(found!.symbolKey).toBeUndefined();
  });

  it('session summary row converter surfaces projectId and symbolKey', async () => {
    const prompt = await store.storeSessionPrompt('sess-conv', 'converter prompt alpha');
    await store.storeSessionSummary({
      sessionId: 'sess-conv',
      promptId: prompt.id,
      summary: 'converter summary alpha',
    });
    // Set project via backfill and symbol_key directly.
    store.setSessionContext('sess-conv', 'proj-conv');
    const db = new DatabaseSync(dbPath);
    db.prepare(`UPDATE session_summaries SET symbol_key = ? WHERE session_id = ?`).run(
      'ts|src/bar.ts|class|module|Baz|0', 'sess-conv',
    );
    db.close();

    const ledger = await store.getSession('sess-conv');
    expect(ledger.summaries.length).toBe(1);
    expect(ledger.summaries[0].projectId).toBe('proj-conv');
    expect(ledger.summaries[0].symbolKey).toBe('ts|src/bar.ts|class|module|Baz|0');
  });
});

// ── T-v7: attachSymbol (PR #2) ─────────────────────────────────────

describe('attachSymbol', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, dbPath, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('attaches a symbol key to an existing memory', async () => {
    const mem = await store.store({ content: 'attach test', category: 'learning', tags: [] });
    const result = await store.attachSymbol({
      symbolKey: 'ts|src/foo.ts|function|global|bar|0',
      memoryId: mem.id,
    });

    expect(result.id).toBe(mem.id);
    expect(result.symbolKey).toBe('ts|src/foo.ts|function|global|bar|0');

    // Verify persisted.
    const recent = await store.listRecent(10);
    const found = recent.find(m => m.id === mem.id);
    expect(found!.symbolKey).toBe('ts|src/foo.ts|function|global|bar|0');
  });

  it('upserts: overwrites previous symbol_key on same memory', async () => {
    const mem = await store.store({ content: 'upsert test', category: 'learning', tags: [] });
    await store.attachSymbol({ symbolKey: 'old-key', memoryId: mem.id });
    await store.attachSymbol({ symbolKey: 'new-key', memoryId: mem.id });

    const recent = await store.listRecent(10);
    const found = recent.find(m => m.id === mem.id);
    expect(found!.symbolKey).toBe('new-key');
  });

  it('throws MemoryNotFoundError for non-existent memory', async () => {
    await expect(
      store.attachSymbol({ symbolKey: 'key', memoryId: 'nonexistent' }),
    ).rejects.toThrow('not found');
  });

  it('attaches a symbol key to a session summary', async () => {
    const prompt = await store.storeSessionPrompt('sess-attach', 'prompt');
    const summary = await store.storeSessionSummary({
      sessionId: 'sess-attach',
      promptId: prompt.id,
      summary: 'summary to attach',
    });

    const result = await store.attachSymbol({
      symbolKey: 'ts|src/mod.ts|class|module|MyClass|0',
      sessionItemId: summary.id,
    });

    expect(result.id).toBe(summary.id);
    expect(result.symbolKey).toBe('ts|src/mod.ts|class|module|MyClass|0');
  });

  it('throws for non-existent session item', async () => {
    await expect(
      store.attachSymbol({ symbolKey: 'key', sessionItemId: 'nonexistent' }),
    ).rejects.toThrow('NOT_FOUND');
  });
});

// ── T-v7: getBySymbol (PR #2) ──────────────────────────────────────

describe('getBySymbol', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('returns memories linked to a symbol key', async () => {
    const mem = await store.store({ content: 'linked memory', category: 'architecture', tags: [] });
    await store.attachSymbol({
      symbolKey: 'ts|src/auth.ts|function|global|jwt|1',
      memoryId: mem.id,
    });

    const results = await store.getBySymbol('ts|src/auth.ts|function|global|jwt|1');
    expect(results.length).toBe(1);
    expect(results[0].source).toBe('memory');
    expect(results[0].content).toBe('linked memory');
  });

  it('returns empty array for unknown symbol key (not an error)', async () => {
    const results = await store.getBySymbol('ts|src/missing.ts|function|global|nope|0');
    expect(results).toEqual([]);
  });

  it('returns session summaries when source is included', async () => {
    const prompt = await store.storeSessionPrompt('sess-get', 'prompt');
    const summary = await store.storeSessionSummary({
      sessionId: 'sess-get',
      promptId: prompt.id,
      summary: 'summary content',
    });
    await store.attachSymbol({
      symbolKey: 'ts|src/s.ts|function|global|fn|0',
      sessionItemId: summary.id,
    });

    const results = await store.getBySymbol('ts|src/s.ts|function|global|fn|0', {
      sources: ['session_summary'],
    });

    expect(results.length).toBe(1);
    expect(results[0].source).toBe('session_summary');
    expect(results[0].content).toBe('summary content');
  });

  it('respects limit parameter', async () => {
    const mem1 = await store.store({ content: 'first', category: 'learning', tags: [] });
    const mem2 = await store.store({ content: 'second', category: 'learning', tags: [] });
    await store.attachSymbol({ symbolKey: 'shared-key', memoryId: mem1.id });
    await store.attachSymbol({ symbolKey: 'shared-key', memoryId: mem2.id });

    const results = await store.getBySymbol('shared-key', { limit: 1 });
    expect(results.length).toBe(1);
  });

  it('returns multiple sources in one query', async () => {
    const mem = await store.store({ content: 'memory content', category: 'learning', tags: [] });
    const prompt = await store.storeSessionPrompt('sess-multi', 'prompt');
    const summary = await store.storeSessionSummary({
      sessionId: 'sess-multi',
      promptId: prompt.id,
      summary: 'summary content',
    });

    await store.attachSymbol({ symbolKey: 'multi-key', memoryId: mem.id });
    await store.attachSymbol({ symbolKey: 'multi-key', sessionItemId: summary.id });

    const results = await store.getBySymbol('multi-key', {
      sources: ['memory', 'session_summary'],
    });

    expect(results.length).toBe(2);
    const sources = results.map(r => r.source);
    expect(sources).toContain('memory');
    expect(sources).toContain('session_summary');
  });
});
