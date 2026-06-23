import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MemoryNotFoundError } from '../src/errors.js';
import { MockEmbedder } from './mocks.js';

// ── Shared fixture ────────────────────────────────────────────────────

async function makeStore(): Promise<{
  store: SqliteMemoryStore;
  embedder: MockEmbedder;
  dbPath: string;
  tmpDir: string;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr2-'));
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

// ── T-003: updateMemoryStatus lifecycle transitions ───────────────────

describe('T-003: updateMemoryStatus transitions', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  async function seed(id = 'lc-1', status?: 'active' | 'needs_review' | 'archived') {
    await env.store.store({
      id, content: `lifecycle ${id}`, category: 'learning', tags: [],
      ...(status ? { status } : {}),
    } as any);
  }

  it('active → needs_review works and sets reason/reviewedAt', async () => {
    await seed('lc-1');
    const updated = await env.store.updateMemoryStatus('lc-1', 'needs_review', 'stale signal');
    expect(updated.status).toBe('needs_review');
    expect(updated.reviewReason).toBe('stale signal');
    expect(updated.reviewedAt).toBeTruthy();
    // version increments
    expect(updated.version).toBe(2);
  });

  it('needs_review → active works when reason provided', async () => {
    await seed('lc-2', 'needs_review');
    const updated = await env.store.updateMemoryStatus('lc-2', 'active', 'reviewed and confirmed');
    expect(updated.status).toBe('active');
    expect(updated.reviewReason).toBe('reviewed and confirmed');
  });

  it('needs_review → active REJECTED when no reason provided', async () => {
    await seed('lc-3', 'needs_review');
    await expect(env.store.updateMemoryStatus('lc-3', 'active'))
      .rejects.toThrow(/reason|transition|invalid/i);
    // Status unchanged
    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT status FROM memories WHERE id = ?`).get('lc-3') as { status: string };
    db.close();
    expect(row.status).toBe('needs_review');
  });

  it('needs_review → archived works (no reason required)', async () => {
    await seed('lc-4', 'needs_review');
    const updated = await env.store.updateMemoryStatus('lc-4', 'archived');
    expect(updated.status).toBe('archived');
  });

  it('active → archived works when reason provided', async () => {
    await seed('lc-5');
    const updated = await env.store.updateMemoryStatus('lc-5', 'archived', 'superseded by mem-x');
    expect(updated.status).toBe('archived');
    expect(updated.reviewReason).toBe('superseded by mem-x');
  });

  it('active → archived REJECTED when no reason provided', async () => {
    await seed('lc-6');
    await expect(env.store.updateMemoryStatus('lc-6', 'archived'))
      .rejects.toThrow(/reason|transition|invalid/i);
    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT status FROM memories WHERE id = ?`).get('lc-6') as { status: string };
    db.close();
    expect(row.status).toBe('active');
  });

  it('archived → active REJECTED', async () => {
    await seed('lc-7', 'archived');
    await expect(env.store.updateMemoryStatus('lc-7', 'active', 'try revive'))
      .rejects.toThrow(/archived|transition|invalid/i);
  });

  it('archived → needs_review REJECTED', async () => {
    await seed('lc-8', 'archived');
    await expect(env.store.updateMemoryStatus('lc-8', 'needs_review'))
      .rejects.toThrow(/archived|transition|invalid/i);
  });

  it('same status (active → active) updates reason/reviewedAt', async () => {
    await seed('lc-9');
    const updated = await env.store.updateMemoryStatus('lc-9', 'active', 'touched during audit');
    expect(updated.status).toBe('active');
    expect(updated.reviewReason).toBe('touched during audit');
    expect(updated.reviewedAt).toBeTruthy();
  });

  it('same status (needs_review → needs_review) updates reason', async () => {
    await seed('lc-10', 'needs_review');
    const updated = await env.store.updateMemoryStatus('lc-10', 'needs_review', 'escalated');
    expect(updated.status).toBe('needs_review');
    expect(updated.reviewReason).toBe('escalated');
  });

  it('throws MemoryNotFoundError for non-existent id', async () => {
    await expect(env.store.updateMemoryStatus('does-not-exist', 'needs_review'))
      .rejects.toBeInstanceOf(MemoryNotFoundError);
  });

  it('does not change other columns (content, category, tags preserved)', async () => {
    await env.store.store({
      id: 'lc-preserve', content: 'keep me', category: 'architecture', tags: ['x', 'y'],
    } as any);
    const updated = await env.store.updateMemoryStatus('lc-preserve', 'needs_review', 'check');
    expect(updated.content).toBe('keep me');
    expect(updated.category).toBe('architecture');
    expect(updated.tags).toEqual(['x', 'y']);
  });
});

// ── T-004: session context (in-memory) ────────────────────────────────

describe('T-004: session context (in-memory)', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('returns null before first set', () => {
    expect(env.store.getSessionContext()).toBeNull();
  });

  it('set/get round-trip', () => {
    env.store.setSessionContext('sess-rt', 'proj-1');
    expect(env.store.getSessionContext()).toEqual({ sessionId: 'sess-rt', projectId: 'proj-1' });
  });

  it('overwrite replaces previous context', () => {
    env.store.setSessionContext('sess-a', 'proj-1');
    env.store.setSessionContext('sess-b', 'proj-2');
    expect(env.store.getSessionContext()).toEqual({ sessionId: 'sess-b', projectId: 'proj-2' });
  });
});

// ── T-004: session ledger store methods ───────────────────────────────

describe('T-004: session ledger store methods', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('storeSessionPrompt returns SessionPrompt with id and persists row', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-1', 'How do I fix the flaky test?');
    expect(prompt.id).toBeTruthy();
    expect(prompt.sessionId).toBe('sess-1');
    expect(prompt.rawPrompt).toBe('How do I fix the flaky test?');
    expect(prompt.createdAt).toBeTruthy();

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT * FROM session_prompts WHERE id = ?`).get(prompt.id) as any;
    db.close();
    expect(row.raw_prompt).toBe('How do I fix the flaky test?');
  });

  it('storeSessionPrompt computes and persists embedding', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-emb', 'embed me');
    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT embedding FROM session_prompts WHERE id = ?`).get(prompt.id) as { embedding: Buffer | null };
    db.close();
    expect(row.embedding).not.toBeNull();
    expect(row.embedding!.length).toBeGreaterThan(0);
  });

  it('storeSessionResponse requires a valid promptId', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-2', 'question');
    const resp = await env.store.storeSessionResponse('sess-2', prompt.id, 'answer text');
    expect(resp.id).toBeTruthy();
    expect(resp.sessionId).toBe('sess-2');
    expect(resp.promptId).toBe(prompt.id);
    expect(resp.fullResponse).toBe('answer text');

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT * FROM session_responses WHERE id = ?`).get(resp.id) as any;
    db.close();
    expect(row.prompt_id).toBe(prompt.id);
  });

  it('storeSessionResponse rejects unknown promptId', async () => {
    await expect(env.store.storeSessionResponse('sess-x', 'no-such-prompt', 'resp'))
      .rejects.toThrow(/prompt|not found|foreign/i);
  });

  it('storeSessionToolCall stores with prompt and response ids', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-3', 'run a search');
    const resp = await env.store.storeSessionResponse('sess-3', prompt.id, 'results');
    const tc = await env.store.storeSessionToolCall({
      sessionId: 'sess-3',
      promptId: prompt.id,
      responseId: resp.id,
      toolName: 'search',
      argsSafe: '{"q":"foo"}',
      resultSummary: 'found 3 results',
      importance: 'normal',
    });
    expect(tc.id).toBeTruthy();
    expect(tc.promptId).toBe(prompt.id);
    expect(tc.responseId).toBe(resp.id);
    expect(tc.toolName).toBe('search');
    expect(tc.importance).toBe('normal');
  });

  it('storeSessionToolCall works without prompt/response ids', async () => {
    const tc = await env.store.storeSessionToolCall({
      sessionId: 'sess-4',
      toolName: 'background_task',
      resultSummary: 'cleaned temp files',
    });
    expect(tc.id).toBeTruthy();
    expect(tc.promptId).toBeUndefined();
    expect(tc.responseId).toBeUndefined();
    expect(tc.toolName).toBe('background_task');
  });

  it('storeSessionToolCall computes embedding from resultSummary or toolName', async () => {
    env.embedder.callLog.length = 0;
    await env.store.storeSessionToolCall({
      sessionId: 'sess-tc-emb',
      toolName: 'no_summary_tool',
      // no resultSummary → should embed toolName
    });
    expect(env.embedder.callLog).toContain('no_summary_tool');

    env.embedder.callLog.length = 0;
    await env.store.storeSessionToolCall({
      sessionId: 'sess-tc-emb',
      toolName: 't2',
      resultSummary: 'summary wins',
    });
    expect(env.embedder.callLog).toContain('summary wins');
  });

  it('storeSessionToolCall rejects invalid importance', async () => {
    await expect(env.store.storeSessionToolCall({
      sessionId: 'sess-imp',
      toolName: 't',
      importance: 'critical' as any,
    })).rejects.toThrow(/importance/i);
  });

  it('storeSessionSummary stores all structured fields as JSON arrays', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-5', 'summarize');
    const resp = await env.store.storeSessionResponse('sess-5', prompt.id, 'long answer');
    const summary = await env.store.storeSessionSummary({
      sessionId: 'sess-5',
      promptId: prompt.id,
      responseId: resp.id,
      summary: 'Implemented session ledger',
      decisions: ['use SQLite', 'store embeddings as BLOB'],
      keyTitles: ['session-ledger.ts', 'sqlite-store.ts'],
      memoryIds: ['mem-1', 'mem-2'],
      artifactRefs: ['docs/session.md'],
    });
    expect(summary.id).toBeTruthy();
    expect(summary.summary).toBe('Implemented session ledger');
    expect(summary.decisions).toEqual(['use SQLite', 'store embeddings as BLOB']);
    expect(summary.keyTitles).toEqual(['session-ledger.ts', 'sqlite-store.ts']);
    expect(summary.memoryIds).toEqual(['mem-1', 'mem-2']);
    expect(summary.artifactRefs).toEqual(['docs/session.md']);

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(`SELECT * FROM session_summaries WHERE id = ?`).get(summary.id) as any;
    db.close();
    expect(JSON.parse(row.decisions)).toEqual(['use SQLite', 'store embeddings as BLOB']);
    expect(JSON.parse(row.memory_ids)).toEqual(['mem-1', 'mem-2']);
  });

  it('storeSessionSummary works with minimal fields', async () => {
    const summary = await env.store.storeSessionSummary({
      sessionId: 'sess-min',
      summary: 'bare summary',
    });
    expect(summary.summary).toBe('bare summary');
    expect(summary.decisions).toBeUndefined();
  });

  it('storeSessionSummary computes embedding from summary + decisions', async () => {
    env.embedder.callLog.length = 0;
    await env.store.storeSessionSummary({
      sessionId: 'sess-sum-emb',
      summary: 'core summary text',
      decisions: ['decision one', 'decision two'],
    });
    // embedder should have been called with combined text
    const combined = env.embedder.callLog.find(t => t.includes('core summary text') && t.includes('decision one'));
    expect(combined).toBeTruthy();
  });
});

// ── T-004: getSession ─────────────────────────────────────────────────

describe('T-004: getSession', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('returns all 4 types ordered by created_at ASC', async () => {
    const sid = 'sess-ordered';
    const p1 = await env.store.storeSessionPrompt(sid, 'first prompt');
    // Small delay so created_at differs; MockEmbedder is sync so we force ordering
    const p2 = await env.store.storeSessionPrompt(sid, 'second prompt');
    const r1 = await env.store.storeSessionResponse(sid, p1.id, 'first response');
    await env.store.storeSessionToolCall({ sessionId: sid, promptId: p1.id, responseId: r1.id, toolName: 't1' });
    await env.store.storeSessionToolCall({ sessionId: sid, promptId: p2.id, toolName: 't2' });
    await env.store.storeSessionSummary({ sessionId: sid, summary: 'final summary' });

    const ledger = await env.store.getSession(sid);
    expect(ledger.sessionId).toBe(sid);
    expect(ledger.prompts).toHaveLength(2);
    expect(ledger.responses).toHaveLength(1);
    expect(ledger.toolCalls).toHaveLength(2);
    expect(ledger.summaries).toHaveLength(1);

    // Ordering: first prompt before second
    expect(ledger.prompts[0].rawPrompt).toBe('first prompt');
    expect(ledger.prompts[1].rawPrompt).toBe('second prompt');
  });

  it('limit truncates each type independently', async () => {
    const sid = 'sess-limit';
    await env.store.storeSessionPrompt(sid, 'p1');
    await env.store.storeSessionPrompt(sid, 'p2');
    await env.store.storeSessionPrompt(sid, 'p3');

    const ledger = await env.store.getSession(sid, { limit: 2 });
    expect(ledger.prompts).toHaveLength(2);
  });

  it('returns empty arrays for unknown session', async () => {
    const ledger = await env.store.getSession('never-existed');
    expect(ledger.prompts).toEqual([]);
    expect(ledger.responses).toEqual([]);
    expect(ledger.toolCalls).toEqual([]);
    expect(ledger.summaries).toEqual([]);
  });

  it('includeSummary:false omits summaries', async () => {
    const sid = 'sess-nosum';
    await env.store.storeSessionPrompt(sid, 'q');
    await env.store.storeSessionSummary({ sessionId: sid, summary: 'should be hidden' });

    const ledger = await env.store.getSession(sid, { includeSummary: false });
    expect(ledger.prompts).toHaveLength(1);
    expect(ledger.summaries).toEqual([]);
  });
});

// ── T-004: listSessions ───────────────────────────────────────────────

describe('T-004: listSessions', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('returns distinct sessions ordered by most recent activity DESC', async () => {
    // Session A: older activity
    await env.store.storeSessionPrompt('sess-old', 'old question');

    // Session B: newer activity
    await env.store.storeSessionPrompt('sess-new', 'new question');

    const sessions = await env.store.listSessions();
    expect(sessions.length).toBeGreaterThanOrEqual(2);
    const ids = sessions.map(s => s.sessionId);
    expect(ids).toContain('sess-old');
    expect(ids).toContain('sess-new');
    // Most recent first
    expect(ids.indexOf('sess-new')).toBeLessThan(ids.indexOf('sess-old'));
  });

  it('promptCount reflects number of prompts in each session', async () => {
    await env.store.storeSessionPrompt('sess-count', 'q1');
    await env.store.storeSessionPrompt('sess-count', 'q2');
    const sessions = await env.store.listSessions();
    const match = sessions.find(s => s.sessionId === 'sess-count');
    expect(match).toBeDefined();
    expect(match!.promptCount).toBe(2);
  });

  it('includes sessions that only have responses/tool_calls/summaries', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-mixed', 'seed');
    await env.store.storeSessionResponse('sess-mixed', prompt.id, 'resp');
    await env.store.storeSessionToolCall({ sessionId: 'sess-mixed', toolName: 't' });

    const sessions = await env.store.listSessions();
    const ids = sessions.map(s => s.sessionId);
    expect(ids).toContain('sess-mixed');
  });

  it('respects limit parameter', async () => {
    await env.store.storeSessionPrompt('s1', 'a');
    await env.store.storeSessionPrompt('s2', 'b');
    await env.store.storeSessionPrompt('s3', 'c');

    const sessions = await env.store.listSessions(2);
    expect(sessions).toHaveLength(2);
  });

  it('returns empty array when no sessions exist', async () => {
    const sessions = await env.store.listSessions();
    expect(sessions).toEqual([]);
  });
});

// ── T-004: linkPromptToMemory ─────────────────────────────────────────

describe('T-004: linkPromptToMemory', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('creates a link row with relation', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-link', 'q');
    const mem = await env.store.store({ content: 'linked memory', category: 'learning', tags: [] } as any);

    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from');

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(
      `SELECT * FROM session_memory_links WHERE prompt_id = ? AND memory_id = ?`,
    ).get(prompt.id, mem.id) as any;
    db.close();
    expect(row).toBeTruthy();
    expect(row.relation).toBe('created_from');
  });

  it('is idempotent (calling twice does not duplicate)', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-idem', 'q');
    const mem = await env.store.store({ content: 'idem', category: 'learning', tags: [] } as any);

    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from');
    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from');

    const db = new DatabaseSync(env.dbPath);
    const rows = db.prepare(
      `SELECT * FROM session_memory_links WHERE prompt_id = ? AND memory_id = ?`,
    ).all(prompt.id, mem.id) as any[];
    db.close();
    expect(rows).toHaveLength(1);
  });

  it('stores responseId when provided', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-rid', 'q');
    const resp = await env.store.storeSessionResponse('sess-rid', prompt.id, 'a');
    const mem = await env.store.store({ content: 'm', category: 'learning', tags: [] } as any);

    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from', resp.id);

    const db = new DatabaseSync(env.dbPath);
    const row = db.prepare(
      `SELECT response_id FROM session_memory_links WHERE prompt_id = ? AND memory_id = ?`,
    ).get(prompt.id, mem.id) as { response_id: string | null };
    db.close();
    expect(row.response_id).toBe(resp.id);
  });

  it('is idempotent even with NULL responseId (NULL-safe dedup)', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-null', 'q');
    const mem = await env.store.store({ content: 'm', category: 'learning', tags: [] } as any);

    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from');      // no responseId
    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from');      // again, no responseId

    const db = new DatabaseSync(env.dbPath);
    const rows = db.prepare(
      `SELECT * FROM session_memory_links WHERE prompt_id = ? AND memory_id = ? AND response_id IS NULL`,
    ).all(prompt.id, mem.id) as any[];
    db.close();
    expect(rows).toHaveLength(1);
  });
});

// ── T-004: auto-link in store() ───────────────────────────────────────

describe('T-004: auto-link during store()', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('after storeSessionPrompt + store, link exists with relation "created_from"', async () => {
    env.store.setSessionContext('sess-auto-1', 'proj');
    const prompt = await env.store.storeSessionPrompt('sess-auto-1', 'user asks about X');

    const mem = await env.store.store({
      content: 'answer about X', category: 'learning', tags: [],
    } as any);

    const db = new DatabaseSync(env.dbPath);
    const link = db.prepare(
      `SELECT * FROM session_memory_links WHERE memory_id = ?`,
    ).get(mem.id) as any;
    db.close();

    expect(link).toBeTruthy();
    expect(link.prompt_id).toBe(prompt.id);
    expect(link.relation).toBe('created_from');
    expect(link.response_id).toBeNull();
  });

  it('after storeSessionPrompt + storeSessionResponse + store, link includes responseId', async () => {
    env.store.setSessionContext('sess-auto-2', 'proj');
    const prompt = await env.store.storeSessionPrompt('sess-auto-2', 'user asks about Y');
    const resp = await env.store.storeSessionResponse('sess-auto-2', prompt.id, 'agent answer about Y');

    const mem = await env.store.store({
      content: 'memory about Y', category: 'learning', tags: [],
    } as any);

    const db = new DatabaseSync(env.dbPath);
    const link = db.prepare(
      `SELECT * FROM session_memory_links WHERE memory_id = ?`,
    ).get(mem.id) as any;
    db.close();

    expect(link).toBeTruthy();
    expect(link.prompt_id).toBe(prompt.id);
    expect(link.response_id).toBe(resp.id);
  });

  it('does NOT create a link when no session context is set', async () => {
    // No setSessionContext call
    await env.store.storeSessionPrompt('sess-noctx', 'orphan prompt');
    const mem = await env.store.store({ content: 'no link', category: 'learning', tags: [] } as any);

    const db = new DatabaseSync(env.dbPath);
    const link = db.prepare(
      `SELECT * FROM session_memory_links WHERE memory_id = ?`,
    ).get(mem.id) as any;
    db.close();
    expect(link).toBeUndefined();
  });

  it('does NOT fail store when session context is set but no prompt exists (fire-and-forget)', async () => {
    env.store.setSessionContext('sess-empty', 'proj');
    // No prompt stored for this session
    await expect(env.store.store({
      content: 'still works', category: 'learning', tags: [],
    } as any)).resolves.toBeTruthy();
  });

  it('links to the MOST RECENT prompt in the active session', async () => {
    env.store.setSessionContext('sess-recent', 'proj');
    const p1 = await env.store.storeSessionPrompt('sess-recent', 'older prompt');
    const p2 = await env.store.storeSessionPrompt('sess-recent', 'newer prompt');

    const mem = await env.store.store({ content: 'm', category: 'learning', tags: [] } as any);

    const db = new DatabaseSync(env.dbPath);
    const link = db.prepare(
      `SELECT prompt_id FROM session_memory_links WHERE memory_id = ?`,
    ).get(mem.id) as { prompt_id: string };
    db.close();
    expect(link.prompt_id).toBe(p2.id);
    expect(link.prompt_id).not.toBe(p1.id);
  });
});

// ── T-004: auto-link in update() ──────────────────────────────────────

describe('T-004: auto-link during update()', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('update() with active session context also creates a link', async () => {
    env.store.setSessionContext('sess-upd', 'proj');
    const prompt = await env.store.storeSessionPrompt('sess-upd', 'original question');

    const mem = await env.store.store({
      id: 'upd-1', content: 'v1', category: 'learning', tags: [],
    } as any);

    // Clear links from the store() auto-link so we can verify update() creates its own
    const db0 = new DatabaseSync(env.dbPath);
    db0.exec(`DELETE FROM session_memory_links`);
    db0.close();

    await env.store.update('upd-1', { content: 'v2' });

    const db = new DatabaseSync(env.dbPath);
    const link = db.prepare(
      `SELECT * FROM session_memory_links WHERE memory_id = ?`,
    ).get(mem.id) as any;
    db.close();
    expect(link).toBeTruthy();
    expect(link.prompt_id).toBe(prompt.id);
    expect(link.relation).toBe('created_from');
  });
});

// ── T-004: embeddings persisted on all session entity types ───────────

describe('T-004: embeddings persisted on session entities', () => {
  let env: Awaited<ReturnType<typeof makeStore>>;

  beforeEach(async () => { env = await makeStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('each session entity type has a non-null embedding BLOB', async () => {
    const prompt = await env.store.storeSessionPrompt('sess-emb-all', 'prompt text');
    const resp = await env.store.storeSessionResponse('sess-emb-all', prompt.id, 'response text');
    const tc = await env.store.storeSessionToolCall({
      sessionId: 'sess-emb-all', promptId: prompt.id, responseId: resp.id,
      toolName: 't', resultSummary: 'tool summary',
    });
    const summary = await env.store.storeSessionSummary({
      sessionId: 'sess-emb-all', summary: 'session summary',
    });

    const db = new DatabaseSync(env.dbPath);
    const pRow = db.prepare(`SELECT embedding FROM session_prompts WHERE id = ?`).get(prompt.id) as any;
    const rRow = db.prepare(`SELECT embedding FROM session_responses WHERE id = ?`).get(resp.id) as any;
    const tRow = db.prepare(`SELECT embedding FROM session_tool_calls WHERE id = ?`).get(tc.id) as any;
    const sRow = db.prepare(`SELECT embedding FROM session_summaries WHERE id = ?`).get(summary.id) as any;
    db.close();

    expect(pRow.embedding).toBeTruthy();
    expect(pRow.embedding.length).toBeGreaterThan(0);
    expect(rRow.embedding).toBeTruthy();
    expect(rRow.embedding.length).toBeGreaterThan(0);
    expect(tRow.embedding).toBeTruthy();
    expect(tRow.embedding.length).toBeGreaterThan(0);
    expect(sRow.embedding).toBeTruthy();
    expect(sRow.embedding.length).toBeGreaterThan(0);
  });
});
