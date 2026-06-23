import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore, scoreSessionRow } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import type { SessionForgetResult, SessionPurgeResult } from '../src/types.js';

// ── Shared fixture ────────────────────────────────────────────────────

async function makeStore(): Promise<{
  store: SqliteMemoryStore;
  embedder: MockEmbedder;
  dbPath: string;
  tmpDir: string;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-decay-'));
  const dbPath = join(tmpDir, 'test.db');
  const embedder = new MockEmbedder();
  const store = new SqliteMemoryStore(dbPath, embedder);
  await store.initialize();
  return {
    store,
    embedder,
    dbPath,
    tmpDir,
    cleanup: async () => {
      store.close();
      await rm(tmpDir, { recursive: true, force: true });
    },
  };
}

/**
 * Build a legacy v2 database (session tables WITHOUT lifecycle columns) with
 * pre-existing rows, then leave it at PRAGMA user_version = 2 so the next
 * initialize() must run the v2→v3 migration. Returns the dbPath.
 */
async function makeLegacyV2Db(): Promise<{ dbPath: string; tmpDir: string }> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-legacy-'));
  const dbPath = join(tmpDir, 'legacy.db');
  const db = new DatabaseSync(dbPath);
  db.exec(`
    PRAGMA journal_mode=WAL;
    CREATE TABLE memories (
      id TEXT PRIMARY KEY, content TEXT NOT NULL, category TEXT NOT NULL, tags TEXT NOT NULL,
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL, access_count INTEGER NOT NULL DEFAULT 0,
      last_accessed_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1, origin TEXT NOT NULL DEFAULT 'unknown',
      source_tool TEXT, session_id TEXT, content_hash TEXT,
      status TEXT NOT NULL DEFAULT 'active', review_reason TEXT, reviewed_at TEXT
    );
    CREATE TABLE session_prompts (
      id TEXT PRIMARY KEY, session_id TEXT NOT NULL, raw_prompt TEXT NOT NULL,
      created_at TEXT NOT NULL, embedding BLOB
    );
    CREATE TABLE session_responses (
      id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt_id TEXT NOT NULL REFERENCES session_prompts(id),
      full_response TEXT NOT NULL, created_at TEXT NOT NULL, embedding BLOB
    );
    CREATE TABLE session_tool_calls (
      id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt_id TEXT REFERENCES session_prompts(id),
      response_id TEXT REFERENCES session_responses(id), tool_name TEXT NOT NULL, args_safe TEXT,
      result_summary TEXT, importance TEXT, created_at TEXT NOT NULL, embedding BLOB
    );
    CREATE TABLE session_summaries (
      id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt_id TEXT REFERENCES session_prompts(id),
      response_id TEXT REFERENCES session_responses(id), summary TEXT NOT NULL, decisions TEXT,
      key_titles TEXT, memory_ids TEXT, artifact_refs TEXT, created_at TEXT NOT NULL, embedding BLOB
    );
  `);
  // Seed rows with a known created_at so we can assert backfill defaults.
  const stamp = '2024-01-15T10:00:00.000Z';
  db.prepare(
    `INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, embedding) VALUES (?, ?, ?, ?, ?)`,
  ).run('leg-p1', 'leg-sess', 'legacy prompt', stamp, Buffer.from([1, 2, 3]));
  db.prepare(
    `INSERT INTO session_responses (id, session_id, prompt_id, full_response, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?)`,
  ).run('leg-r1', 'leg-sess', 'leg-p1', 'legacy response', stamp, Buffer.from([4, 5, 6]));
  db.prepare(
    `INSERT INTO session_tool_calls (id, session_id, prompt_id, response_id, tool_name, result_summary, importance, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run('leg-t1', 'leg-sess', 'leg-p1', 'leg-r1', 'legacy_tool', 'legacy call', 'normal', stamp, Buffer.from([7, 8]));
  db.prepare(
    `INSERT INTO session_summaries (id, session_id, prompt_id, response_id, summary, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)`,
  ).run('leg-s1', 'leg-sess', 'leg-p1', 'leg-r1', 'legacy summary', stamp, Buffer.from([9]));
  db.exec(`PRAGMA user_version = 2`);
  db.close();
  return { dbPath, tmpDir };
}

// ── Schema v3 migration (tasks 2.1, 2.2, 2.6) ─────────────────────────

describe('schema v3 migration: lifecycle columns on session tables', () => {
  let dbPath: string;
  let tmpDir: string;

  afterEach(async () => {
    if (tmpDir) await rm(tmpDir, { recursive: true, force: true });
  });

  it('upgrades a v2 database to v3 and backfills lifecycle defaults', async () => {
    ({ dbPath, tmpDir } = await makeLegacyV2Db());

    const embedder = new MockEmbedder();
    const store = new SqliteMemoryStore(dbPath, embedder);
    await store.initialize();
    store.close();

    const db = new DatabaseSync(dbPath);
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBe(4);

    // Prompt row: defaults backfilled from created_at
    const p = db.prepare(`SELECT * FROM session_prompts WHERE id = 'leg-p1'`).get() as Record<string, unknown>;
    expect(p['updated_at']).toBe('2024-01-15T10:00:00.000Z');
    expect(p['last_accessed_at']).toBe('2024-01-15T10:00:00.000Z');
    expect(p['access_count']).toBe(0);
    expect(p['status']).toBe('active');

    // Response / tool_call / summary share the same backfill semantics
    for (const tbl of ['session_responses', 'session_tool_calls', 'session_summaries']) {
      const row = db.prepare(`SELECT * FROM ${tbl} LIMIT 1`).get() as Record<string, unknown>;
      expect(row['updated_at']).toBe('2024-01-15T10:00:00.000Z');
      expect(row['last_accessed_at']).toBe('2024-01-15T10:00:00.000Z');
      expect(row['access_count']).toBe(0);
      expect(row['status']).toBe('active');
    }

    // Existing payload columns preserved
    const promptContent = db.prepare(`SELECT raw_prompt, embedding FROM session_prompts WHERE id = 'leg-p1'`).get() as any;
    expect(promptContent.raw_prompt).toBe('legacy prompt');
    expect(promptContent.embedding).toBeTruthy();
    db.close();
  });

  it('is idempotent — re-initializing a v3 db is a no-op', async () => {
    ({ dbPath, tmpDir } = await makeLegacyV2Db());

    const embedder = new MockEmbedder();
    const store1 = new SqliteMemoryStore(dbPath, embedder);
    await store1.initialize();
    store1.close();

    // Second initialize must not throw and must keep user_version at 4
    const store2 = new SqliteMemoryStore(dbPath, embedder);
    await expect(store2.initialize()).resolves.toBeUndefined();
    store2.close();

    const db = new DatabaseSync(dbPath);
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBe(4);
    // Row still present and intact after two migrations
    const p = db.prepare(`SELECT raw_prompt, status FROM session_prompts WHERE id = 'leg-p1'`).get() as any;
    expect(p.raw_prompt).toBe('legacy prompt');
    expect(p.status).toBe('active');
    db.close();
  });

  it('fresh database reaches v3 with lifecycle columns present', async () => {
    const tmp = await mkdtemp(join(tmpdir(), 'amm-fresh-'));
    tmpDir = tmp;
    dbPath = join(tmp, 'fresh.db');

    const embedder = new MockEmbedder();
    const store = new SqliteMemoryStore(dbPath, embedder);
    await store.initialize();
    store.close();

    const db = new DatabaseSync(dbPath);
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBe(4);

    const cols = (tbl: string) =>
      (db.prepare(`PRAGMA table_info(${tbl})`).all() as { name: string }[]).map(c => c.name);
    for (const tbl of ['session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries']) {
      const names = cols(tbl);
      expect(names).toContain('updated_at');
      expect(names).toContain('access_count');
      expect(names).toContain('last_accessed_at');
      expect(names).toContain('status');
    }
    db.close();
  });
});

// ── Read-path decay & access tracking (tasks 2.4, 2.5, 2.7) ───────────

describe('getSession decay-aware retrieval', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('excludes forgotten rows by default', async () => {
    const sid = 'decay-1';
    await env.store.storeSessionPrompt(sid, 'keep me');
    await env.store.storeSessionPrompt(sid, 'hide me');

    const db = new DatabaseSync(env.dbPath);
    // Fetch both ids, then mark the second as forgotten directly.
    const rows = db.prepare(`SELECT id, raw_prompt FROM session_prompts WHERE session_id = ? ORDER BY raw_prompt`).all(sid) as any[];
    db.close();
    const forgottenId = rows.find((r: any) => r.raw_prompt === 'hide me').id;

    await env.store.forgetSessionItem(forgottenId);

    const ledger = await env.store.getSession(sid);
    expect(ledger.prompts.map(p => p.rawPrompt)).toEqual(['keep me']);
  });

  it('returns forgotten rows when includeForgotten is true', async () => {
    const sid = 'decay-2';
    const a = await env.store.storeSessionPrompt(sid, 'alpha');
    await env.store.forgetSessionItem(a.id);

    const ledger = await env.store.getSession(sid, { includeForgotten: true });
    expect(ledger.prompts).toHaveLength(1);
    expect(ledger.prompts[0].status).toBe('forgotten');
  });

  it('increments access_count and updates last_accessed_at on read', async () => {
    const sid = 'access-1';
    const prompt = await env.store.storeSessionPrompt(sid, 'read me');

    // Baseline: access_count = 0 after creation
    const before = await env.store.getSession(sid);
    const p0 = before.prompts.find(p => p.id === prompt.id)!;
    expect(p0.accessCount).toBe(0);

    // Read again → access_count becomes 1
    await env.store.getSession(sid);
    const ledger = await env.store.getSession(sid);

    const p = ledger.prompts.find(p => p.id === prompt.id)!;
    expect(p.accessCount).toBe(2);
    expect(p.lastAccessedAt).toBeTruthy();
    // last_accessed_at should be >= created_at
    expect(p.lastAccessedAt! >= p.createdAt).toBe(true);
  });
});

// ── Session decay scoring helper (task 2.5) ───────────────────────────

describe('session decay scoring helper', () => {
  it('scores a recently-updated session row near 1.0', () => {
    const now = new Date().toISOString();
    const row = { updated_at: now, access_count: 5, last_accessed_at: now };
    const factor = scoreSessionRow(row);
    expect(factor).toBeGreaterThan(0.5);
    expect(factor).toBeLessThanOrEqual(1.0);
  });

  it('decays an old, never-accessed session row toward 0', () => {
    const oldStamp = '2020-01-01T00:00:00.000Z';
    const row = { updated_at: oldStamp, access_count: 0, last_accessed_at: oldStamp };
    const factor = scoreSessionRow(row);
    expect(factor).toBeLessThan(0.5);
  });
});

// ── forgetSessionItem / purgeSessionItem store methods ────────────────

describe('forgetSessionItem — soft delete across all four entity types', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  async function seedAll() {
    const prompt = await env.store.storeSessionPrompt('fg-sess', 'prompt text');
    const response = await env.store.storeSessionResponse('fg-sess', prompt.id, 'response text');
    const toolCall = await env.store.storeSessionToolCall({
      sessionId: 'fg-sess', promptId: prompt.id, responseId: response.id,
      toolName: 't', resultSummary: 'call summary',
    });
    const summary = await env.store.storeSessionSummary({ sessionId: 'fg-sess', summary: 'sum text' });
    return { prompt, response, toolCall, summary };
  }

  it('soft-deletes a prompt → reports type "prompt" + success', async () => {
    const { prompt } = await seedAll();
    const result: SessionForgetResult = await env.store.forgetSessionItem(prompt.id);
    expect(result).toEqual({ type: 'prompt', success: true });

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT status FROM session_prompts WHERE id = ?`).get(prompt.id) as { status: string };
    db.close();
    expect(row.status).toBe('forgotten');
  });

  it('soft-deletes a response → reports type "response"', async () => {
    const { response } = await seedAll();
    const result = await env.store.forgetSessionItem(response.id);
    expect(result.type).toBe('response');
    expect(result.success).toBe(true);
  });

  it('soft-deletes a tool_call → reports type "tool_call"', async () => {
    const { toolCall } = await seedAll();
    const result = await env.store.forgetSessionItem(toolCall.id);
    expect(result.type).toBe('tool_call');
    expect(result.success).toBe(true);
  });

  it('soft-deletes a summary → reports type "summary"', async () => {
    const { summary } = await seedAll();
    const result = await env.store.forgetSessionItem(summary.id);
    expect(result.type).toBe('summary');
    expect(result.success).toBe(true);
  });

  it('throws NOT_FOUND for an unknown id', async () => {
    await expect(env.store.forgetSessionItem('no-such-id'))
      .rejects.toThrow(/not found|NOT_FOUND/i);
  });

  it('refuses deletion when the id is ambiguous across tables', async () => {
    // Force the same id into two tables to simulate an (extremely rare) collision.
    const { prompt } = await seedAll();
    const db = new DatabaseSync(env.dbPath);
    db.prepare(`UPDATE session_summaries SET id = ? WHERE id = (SELECT id FROM session_summaries LIMIT 1)`).run(prompt.id);
    db.close();

    await expect(env.store.forgetSessionItem(prompt.id))
      .rejects.toThrow(/ambiguous|multiple/i);
    // Neither row was forgotten
    const db2 = new DatabaseSync(env.dbPath);
    const pRow = db2.prepare(`SELECT status FROM session_prompts WHERE id = ?`).get(prompt.id) as { status: string };
    db2.close();
    expect(pRow.status).toBe('active');
  });
});

describe('purgeSessionItem — hard delete row + embedding', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  async function seedAll() {
    const prompt = await env.store.storeSessionPrompt('pg-sess', 'prompt text');
    const response = await env.store.storeSessionResponse('pg-sess', prompt.id, 'response text');
    const toolCall = await env.store.storeSessionToolCall({
      sessionId: 'pg-sess', promptId: prompt.id, responseId: response.id,
      toolName: 't', resultSummary: 'call summary',
    });
    const summary = await env.store.storeSessionSummary({ sessionId: 'pg-sess', summary: 'sum text' });
    return { prompt, response, toolCall, summary };
  }

  it('hard-deletes a summary row AND its embedding → reports type "summary"', async () => {
    const { summary } = await seedAll();
    const result: SessionPurgeResult = await env.store.purgeSessionItem(summary.id);
    expect(result).toEqual({ type: 'summary', success: true });

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT id FROM session_summaries WHERE id = ?`).get(summary.id);
    const emb = db.prepare(`SELECT embedding FROM session_summaries WHERE id = ?`).get(summary.id);
    db.close();
    expect(row).toBeUndefined();
    expect(emb).toBeUndefined();
  });

  it('hard-deletes a prompt → reports type "prompt"', async () => {
    const { prompt } = await seedAll();
    const result = await env.store.purgeSessionItem(prompt.id);
    expect(result.type).toBe('prompt');
    expect(result.success).toBe(true);

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT id FROM session_prompts WHERE id = ?`).get(prompt.id);
    db.close();
    expect(row).toBeUndefined();
  });

  it('hard-deletes a response → reports type "response"', async () => {
    const { response } = await seedAll();
    const result = await env.store.purgeSessionItem(response.id);
    expect(result.type).toBe('response');
  });

  it('hard-deletes a tool_call → reports type "tool_call"', async () => {
    const { toolCall } = await seedAll();
    const result = await env.store.purgeSessionItem(toolCall.id);
    expect(result.type).toBe('tool_call');
  });

  it('throws NOT_FOUND for an unknown id', async () => {
    await expect(env.store.purgeSessionItem('does-not-exist'))
      .rejects.toThrow(/not found|NOT_FOUND/i);
  });

  it('does not delete other session items when purging one', async () => {
    const { prompt, summary } = await seedAll();
    await env.store.purgeSessionItem(summary.id);

    const db = new DatabaseSync(env.dbPath);
    const stillThere = db.prepare(`SELECT id FROM session_prompts WHERE id = ?`).get(prompt.id);
    db.close();
    expect(stillThere).toBeTruthy();
  });
});
