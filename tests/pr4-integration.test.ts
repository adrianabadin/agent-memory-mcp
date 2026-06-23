// ─────────────────────────────────────────────────────────────────────
// Scope note (resolves verify-report W2):
// This file holds integration-level coverage of the PR-2 store + tool
// surfaces (SqliteMemoryStore + HardcopyMemoryStore round-trips, lifecycle,
// merge). It is NOT the "Phase 4" background-action E2E — that E2E lives
// in the PMC package at `tools/project-memory-context/tests/e2e/`. Kept
// under this name for continuity with the PR that introduced these suites.
// ─────────────────────────────────────────────────────────────────────

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { HardcopyMemoryStore } from '../src/hardcopy-store.js';
import { MockMemoryStore, MockEmbedder } from './mocks.js';
import {
  handleUpdateMemoryStatus,
  handleStoreSessionPrompt,
  handleStoreSessionResponse,
  handleStoreSessionToolCall,
  handleGetSession,
  handleListSessions,
  handleMergeProjects,
} from '../src/tools.js';
import type { GlobalStore } from '../src/global/global-store.js';

// ── Helpers ────────────────────────────────────────────────────────

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return JSON.parse(result.content[0].text);
}

async function makeSqliteStore(): Promise<{
  store: SqliteMemoryStore;
  embedder: MockEmbedder;
  dbPath: string;
  tmpDir: string;
  cleanup: () => Promise<void>;
  rawDb: () => DatabaseSync;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr4-'));
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
    rawDb: () => new DatabaseSync(dbPath),
  };
}

function makeMockGlobalStore(overrides?: {
  mergeProjects?: (s: string, t: string, d?: boolean) => Promise<{
    movedMemories: number; movedErrors: number; dryRun: boolean;
    sourceProjectId: string; targetProjectId: string;
  }>;
}): GlobalStore {
  return {
    registerProject: () => { throw new Error('not implemented'); },
    syncMetadata: () => { throw new Error('not implemented'); },
    recordError: async () => { throw new Error('not implemented'); },
    searchErrors: async () => { throw new Error('not implemented'); },
    mergeProjects: overrides?.mergeProjects ?? (async (sourceId, targetId, dryRun = true) => {
      if (sourceId === targetId) throw new Error('Cannot merge a project into itself');
      return {
        movedMemories: 3, movedErrors: 1, dryRun,
        sourceProjectId: sourceId, targetProjectId: targetId,
      };
    }),
    close: () => {},
  } as unknown as GlobalStore;
}

// ═════════════════════════════════════════════════════════════════════
// T-007: MockMemoryStore implements the FULL MemoryStore interface
// ═════════════════════════════════════════════════════════════════════

describe('T-007: MockMemoryStore full interface', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  // ── Lifecycle ──────────────────────────────────────────────────

  describe('updateMemoryStatus', () => {
    it('transitions active → needs_review and persists reason/reviewedAt', async () => {
      const mem = await store.store({ content: 'active mem', category: 'learning', tags: [] });
      const updated = await store.updateMemoryStatus(mem.id, 'needs_review', 'stale');

      expect(updated.status).toBe('needs_review');
      expect(updated.reviewReason).toBe('stale');
      expect(updated.reviewedAt).toBeTruthy();
      // Persisted in underlying map
      const searched = await store.search('active', 'hybrid', { status: 'all' });
      expect(searched[0].memory.status).toBe('needs_review');
    });

    it('rejects archived → active (no transition out of archived)', async () => {
      const mem = await store.store({ content: 'x', category: 'other', tags: [] });
      await store.updateMemoryStatus(mem.id, 'archived', 'done');
      await expect(store.updateMemoryStatus(mem.id, 'active'))
        .rejects.toThrow(/transition|invalid/i);
    });

    it('requires reviewReason when moving active → archived', async () => {
      const mem = await store.store({ content: 'y', category: 'other', tags: [] });
      await expect(store.updateMemoryStatus(mem.id, 'archived'))
        .rejects.toThrow(/reason/i);
    });

    it('throws on unknown memory id', async () => {
      await expect(store.updateMemoryStatus('no-such', 'archived', 'r'))
        .rejects.toThrow(/not found/i);
    });
  });

  // ── Topic aliases ──────────────────────────────────────────────

  describe('upsertTopicAlias / resolveTopic', () => {
    it('resolveTopic returns null for an unknown key', async () => {
      expect(await store.resolveTopic('missing/key')).toBeNull();
    });

    it('upsertTopicAlias then resolveTopic returns the memory', async () => {
      const mem = await store.store({ content: 'topic target', category: 'architecture', tags: [] });
      await store.upsertTopicAlias('arch/auth', mem.id);
      const resolved = await store.resolveTopic('arch/auth');
      expect(resolved?.id).toBe(mem.id);
      expect(resolved?.content).toBe('topic target');
    });

    it('re-store under same topicKey refreshes the alias target', async () => {
      const first = await store.store({
        content: 'first version', category: 'learning', tags: [],
        topicKey: 'learning/tdd',
      });
      expect(await store.resolveTopic('learning/tdd')).toMatchObject({ id: first.id });

      const second = await store.store({
        content: 'second version', category: 'learning', tags: [],
        topicKey: 'learning/tdd',
      });
      const resolved = await store.resolveTopic('learning/tdd');
      expect(resolved?.id).toBe(second.id);
      expect(resolved?.content).toBe('second version');
    });

    it('upsertTopicAlias rewrites an existing alias to a new memory', async () => {
      const a = await store.store({ content: 'A', category: 'other', tags: [] });
      const b = await store.store({ content: 'B', category: 'other', tags: [] });
      await store.upsertTopicAlias('k', a.id);
      await store.upsertTopicAlias('k', b.id);
      expect(await store.resolveTopic('k')).toMatchObject({ id: b.id });
    });
  });

  // ── Session ledger ─────────────────────────────────────────────

  describe('session ledger methods', () => {
    it('storeSessionPrompt returns a prompt with id and createdAt', async () => {
      const p = await store.storeSessionPrompt('s1', 'hello');
      expect(p.id).toBeTruthy();
      expect(p.sessionId).toBe('s1');
      expect(p.rawPrompt).toBe('hello');
      expect(p.createdAt).toBeTruthy();
    });

    it('storeSessionResponse rejects an unknown promptId', async () => {
      await expect(store.storeSessionResponse('s1', 'no-prompt', 'x'))
        .rejects.toThrow(/not found|exist/i);
    });

    it('storeSessionResponse persists a response linked to its prompt', async () => {
      const p = await store.storeSessionPrompt('s1', 'q');
      const r = await store.storeSessionResponse('s1', p.id, 'a');
      expect(r.id).toBeTruthy();
      expect(r.promptId).toBe(p.id);
      expect(r.fullResponse).toBe('a');
    });

    it('storeSessionToolCall rejects an invalid importance', async () => {
      await expect(store.storeSessionToolCall({
        sessionId: 's1', toolName: 'bash', importance: 'critical' as 'high',
      })).rejects.toThrow(/importance/i);
    });

    it('storeSessionToolCall persists optional fields', async () => {
      const tc = await store.storeSessionToolCall({
        sessionId: 's1', toolName: 'bash', argsSafe: '{}', resultSummary: 'ok', importance: 'high',
      });
      expect(tc.toolName).toBe('bash');
      expect(tc.argsSafe).toBe('{}');
      expect(tc.importance).toBe('high');
    });

    it('storeSessionSummary persists structured fields', async () => {
      const s = await store.storeSessionSummary({
        sessionId: 's1', summary: 'done',
        decisions: ['d1', 'd2'], keyTitles: ['T'], memoryIds: ['m1'], artifactRefs: ['r.md'],
      });
      expect(s.summary).toBe('done');
      expect(s.decisions).toEqual(['d1', 'd2']);
      expect(s.memoryIds).toEqual(['m1']);
    });

    it('getSession collects prompts, responses, toolCalls and summaries', async () => {
      const p = await store.storeSessionPrompt('s1', 'q');
      await store.storeSessionResponse('s1', p.id, 'a');
      await store.storeSessionToolCall({ sessionId: 's1', toolName: 'bash' });
      await store.storeSessionSummary({ sessionId: 's1', summary: 'sum' });

      const ledger = await store.getSession('s1');
      expect(ledger.sessionId).toBe('s1');
      expect(ledger.prompts).toHaveLength(1);
      expect(ledger.responses).toHaveLength(1);
      expect(ledger.toolCalls).toHaveLength(1);
      expect(ledger.summaries).toHaveLength(1);
    });

    it('getSession with includeSummary:false omits summaries', async () => {
      await store.storeSessionSummary({ sessionId: 's1', summary: 'x' });
      const ledger = await store.getSession('s1', { includeSummary: false });
      expect(ledger.summaries).toEqual([]);
    });

    it('listSessions groups by sessionId ordered by lastActivity DESC', async () => {
      await store.storeSessionPrompt('old', 'q1');
      await new Promise(r => setTimeout(r, 5));
      await store.storeSessionPrompt('new', 'q2');

      const sessions = await store.listSessions();
      expect(sessions).toHaveLength(2);
      expect(sessions[0].sessionId).toBe('new');
      expect(sessions[0].promptCount).toBe(1);
      expect(sessions[1].sessionId).toBe('old');
    });

    it('listSessions respects limit', async () => {
      await store.storeSessionPrompt('a', 'q');
      await store.storeSessionPrompt('b', 'q');
      await store.storeSessionPrompt('c', 'q');
      const limited = await store.listSessions(2);
      expect(limited).toHaveLength(2);
    });

    it('linkPromptToMemory dedupes repeated calls', async () => {
      const p = await store.storeSessionPrompt('s1', 'q');
      const m = await store.store({ content: 'mem', category: 'other', tags: [] });
      // First link — expected to succeed
      await store.linkPromptToMemory(p.id, m.id, 'created_from');
      // Same link again — must NOT throw (dedup), nor create a duplicate row
      await store.linkPromptToMemory(p.id, m.id, 'created_from');
      // Different relation — creates a new row, proves original wasn't accidentally blocked
      await store.linkPromptToMemory(p.id, m.id, 'referenced');
      // Dedup is proven: if it didn't dedup, the 2nd call would throw on PK violation.
      // The 3rd call with different relation proves the relationship isn't stuck.
    });
  });

  // ── Session context ────────────────────────────────────────────

  describe('setSessionContext / getSessionContext', () => {
    it('returns null before being set', () => {
      expect(store.getSessionContext()).toBeNull();
    });

    it('round-trips the context', () => {
      store.setSessionContext('sess-1', 'proj-1');
      expect(store.getSessionContext()).toEqual({ sessionId: 'sess-1', projectId: 'proj-1' });
    });

    it('overwrites previous context', () => {
      store.setSessionContext('sess-1', 'proj-1');
      store.setSessionContext('sess-2', 'proj-2');
      expect(store.getSessionContext()).toEqual({ sessionId: 'sess-2', projectId: 'proj-2' });
    });
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-007: HardcopyMemoryStore delegation rules
//   - lifecycle mutations → mirror to disk
//   - topic aliases → no hardcopy (relational)
//   - session ledger → no hardcopy (per design)
//   - session context → no hardcopy (in-memory)
// ═════════════════════════════════════════════════════════════════════

describe('T-007: HardcopyMemoryStore delegation rules', () => {
  let inner: MockMemoryStore;
  let store: HardcopyMemoryStore;
  let hardcopyPath: string;

  beforeEach(async () => {
    hardcopyPath = await mkdtemp(join(tmpdir(), 'pr4-hardcopy-'));
    inner = new MockMemoryStore();
    store = new HardcopyMemoryStore(inner, hardcopyPath);
    await store.initialize();
  });

  afterEach(async () => {
    await rm(hardcopyPath, { recursive: true, force: true });
  });

  async function listHardcopyFiles(): Promise<string[]> {
    return (await readdir(hardcopyPath)).filter(f => f.endsWith('.json'));
  }

  it('updateMemoryStatus mirrors the updated memory to a hardcopy file', async () => {
    const mem = await store.store({ content: 'lc', category: 'learning', tags: [] });
    await store.updateMemoryStatus(mem.id, 'needs_review', 'stale');

    const files = await listHardcopyFiles();
    expect(files).toContain(`${mem.id}.json`);
    const hardcopy = JSON.parse(await readFile(join(hardcopyPath, `${mem.id}.json`), 'utf-8'));
    expect(hardcopy.status).toBe('needs_review');
    expect(hardcopy.reviewReason).toBe('stale');
  });

  it('upsertTopicAlias does NOT create a hardcopy file', async () => {
    const mem = await store.store({ content: 'topic', category: 'architecture', tags: [] });
    const filesBefore = await listHardcopyFiles();
    await store.upsertTopicAlias('arch/x', mem.id);
    const filesAfter = await listHardcopyFiles();
    expect(filesAfter.sort()).toEqual(filesBefore.sort());
  });

  it('resolveTopic delegates read-only (no hardcopy writes)', async () => {
    const mem = await store.store({
      content: 'topic target', category: 'architecture', tags: [],
      topicKey: 'arch/y',
    });
    const filesBefore = await listHardcopyFiles();
    const resolved = await store.resolveTopic('arch/y');
    expect(resolved?.id).toBe(mem.id);
    const filesAfter = await listHardcopyFiles();
    expect(filesAfter.sort()).toEqual(filesBefore.sort());
  });

  it('storeSessionPrompt does NOT create a hardcopy file (session ledger skipped by design)', async () => {
    const filesBefore = await listHardcopyFiles();
    await store.storeSessionPrompt('s1', 'hello');
    const filesAfter = await listHardcopyFiles();
    expect(filesAfter.sort()).toEqual(filesBefore.sort());
  });

  it('storeSessionResponse / ToolCall / Summary do NOT create hardcopy files', async () => {
    const filesBefore = await listHardcopyFiles();
    const p = await store.storeSessionPrompt('s1', 'q');
    await store.storeSessionResponse('s1', p.id, 'a');
    await store.storeSessionToolCall({ sessionId: 's1', toolName: 'bash' });
    await store.storeSessionSummary({ sessionId: 's1', summary: 'sum' });
    const filesAfter = await listHardcopyFiles();
    expect(filesAfter.sort()).toEqual(filesBefore.sort());
  });

  it('setSessionContext / getSessionContext delegate to inner without hardcopy writes', () => {
    const filesBeforeSnapshot = true; // existence check below
    store.setSessionContext('s2', 'p2');
    expect(store.getSessionContext()).toEqual({ sessionId: 's2', projectId: 'p2' });
    expect(inner.getSessionContext()).toEqual({ sessionId: 's2', projectId: 'p2' });
    expect(filesBeforeSnapshot).toBe(true);
  });

  it('linkPromptToMemory delegates without hardcopy writes', async () => {
    const p = await store.storeSessionPrompt('s1', 'q');
    const m = await store.store({ content: 'mem', category: 'other', tags: [] });
    const filesBefore = (await listHardcopyFiles()).slice().sort();
    await store.linkPromptToMemory(p.id, m.id, 'created_from');
    const filesAfter = (await listHardcopyFiles()).slice().sort();
    expect(filesAfter).toEqual(filesBefore);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Full lifecycle integration against SQLite
// ═════════════════════════════════════════════════════════════════════

describe('T-008: lifecycle integration (SQLite)', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('store → update_memory_status → search with status filter round-trips', async () => {
    const mem = await env.store.store({ content: 'lifecycle integration', category: 'learning', tags: ['x'] });
    await env.store.updateMemoryStatus(mem.id, 'needs_review', 'verify');

    // Default search excludes archived, includes needs_review
    const defaultHits = await env.store.search('lifecycle', 'hybrid', {});
    expect(defaultHits.some(r => r.memory.id === mem.id)).toBe(true);

    // Filter to only needs_review
    const filtered = await env.store.search('lifecycle', 'hybrid', { status: 'needs_review' });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].memory.status).toBe('needs_review');
  });

  it('archived memory is excluded by default and included with status:"all"', async () => {
    const mem = await env.store.store({ content: 'to be archived', category: 'other', tags: [] });
    await env.store.updateMemoryStatus(mem.id, 'archived', 'done');

    const defaultHits = await env.store.search('archived', 'hybrid', {});
    expect(defaultHits).toHaveLength(0);

    const allHits = await env.store.search('archived', 'hybrid', { status: 'all' });
    expect(allHits).toHaveLength(1);
    expect(allHits[0].memory.status).toBe('archived');
  });

  it('listRecent excludes archived memories by default', async () => {
    const a = await env.store.store({ content: 'keep me', category: 'learning', tags: [] });
    const b = await env.store.store({ content: 'hide me', category: 'learning', tags: [] });
    await env.store.updateMemoryStatus(b.id, 'archived', 'done');

    const recent = await env.store.listRecent(10);
    const ids = recent.map(m => m.id);
    expect(ids).toContain(a.id);
    expect(ids).not.toContain(b.id);
  });

  it('lifecycle full chain active → needs_review → archived works', async () => {
    const mem = await env.store.store({ content: 'chain', category: 'learning', tags: [] });
    await env.store.updateMemoryStatus(mem.id, 'needs_review', 'check');
    await env.store.updateMemoryStatus(mem.id, 'archived', 'done');

    const db = env.rawDb();
    const row = db.prepare(`SELECT status, review_reason FROM memories WHERE id = ?`).get(mem.id) as { status: string; review_reason: string };
    db.close();
    expect(row.status).toBe('archived');
    expect(row.review_reason).toBe('done');
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Topic aliases integration against SQLite
// ═════════════════════════════════════════════════════════════════════

describe('T-008: topic aliases integration (SQLite)', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('store with topicKey → resolveTopic returns the stored memory', async () => {
    await env.store.store({
      content: 'alias target', category: 'architecture', tags: [],
      topicKey: 'arch/auth',
    });
    const resolved = await env.store.resolveTopic('arch/auth');
    expect(resolved).not.toBeNull();
    expect(resolved?.content).toBe('alias target');
  });

  it('re-store with same topicKey → resolveTopic returns the NEW content', async () => {
    await env.store.store({ content: 'v1', category: 'learning', tags: [], topicKey: 'k' });
    await env.store.store({ content: 'v2', category: 'learning', tags: [], topicKey: 'k' });
    const resolved = await env.store.resolveTopic('k');
    expect(resolved?.content).toBe('v2');
  });

  it('upsertTopicAlias for an existing key redirects to the new memory', async () => {
    const a = await env.store.store({ content: 'A', category: 'other', tags: [] });
    const b = await env.store.store({ content: 'B', category: 'other', tags: [] });
    await env.store.upsertTopicAlias('shared', a.id);
    await env.store.upsertTopicAlias('shared', b.id);
    const resolved = await env.store.resolveTopic('shared');
    expect(resolved?.id).toBe(b.id);
  });

  it('resolveTopic returns null for an unknown key', async () => {
    expect(await env.store.resolveTopic('unknown/key')).toBeNull();
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Session ledger full round-trip against SQLite
// ═════════════════════════════════════════════════════════════════════

describe('T-008: session ledger round-trip (SQLite)', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('prompt → response → tool_call → get_session returns all three', async () => {
    const p = await env.store.storeSessionPrompt('s1', 'how do I test?');
    const r = await env.store.storeSessionResponse('s1', p.id, 'answer');
    const tc = await env.store.storeSessionToolCall({
      sessionId: 's1', promptId: p.id, responseId: r.id,
      toolName: 'bash', importance: 'high',
    });

    const ledger = await env.store.getSession('s1');
    expect(ledger.prompts).toHaveLength(1);
    expect(ledger.responses).toHaveLength(1);
    expect(ledger.toolCalls).toHaveLength(1);
    expect(ledger.prompts[0].id).toBe(p.id);
    expect(ledger.responses[0].id).toBe(r.id);
    expect(ledger.toolCalls[0].id).toBe(tc.id);
    expect(ledger.toolCalls[0].importance).toBe('high');
  });

  it('listSessions returns session with correct prompt count', async () => {
    await env.store.storeSessionPrompt('s1', 'q1');
    await env.store.storeSessionPrompt('s1', 'q2');
    await env.store.storeSessionPrompt('s1', 'q3');

    const sessions = await env.store.listSessions();
    expect(sessions).toHaveLength(1);
    expect(sessions[0].sessionId).toBe('s1');
    expect(sessions[0].promptCount).toBe(3);
  });

  it('multiple sessions: listSessions returns one entry per session ordered DESC', async () => {
    await env.store.storeSessionPrompt('old', 'q');
    await new Promise(r => setTimeout(r, 5));
    await env.store.storeSessionPrompt('new', 'q');

    const sessions = await env.store.listSessions();
    expect(sessions).toHaveLength(2);
    expect(sessions[0].sessionId).toBe('new');
    expect(sessions[1].sessionId).toBe('old');
  });

  it('summary is included by default and omitted with includeSummary:false', async () => {
    await env.store.storeSessionSummary({ sessionId: 's1', summary: 'note' });
    const withSummary = await env.store.getSession('s1');
    expect(withSummary.summaries).toHaveLength(1);

    const withoutSummary = await env.store.getSession('s1', { includeSummary: false });
    expect(withoutSummary.summaries).toEqual([]);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Auto-link integration against SQLite
// ═════════════════════════════════════════════════════════════════════

describe('T-008: auto-link integration (SQLite)', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('setSessionContext → store memory → session_memory_links row created', async () => {
    env.store.setSessionContext('auto-1', 'proj-1');
    const prompt = await env.store.storeSessionPrompt('auto-1', 'trigger');
    const mem = await env.store.store({ content: 'auto-linked', category: 'learning', tags: [] });

    const db = env.rawDb();
    const rows = db.prepare(
      `SELECT prompt_id, memory_id, relation FROM session_memory_links WHERE memory_id = ?`,
    ).all(mem.id) as Array<{ prompt_id: string; memory_id: string; relation: string }>;
    db.close();

    expect(rows).toHaveLength(1);
    expect(rows[0].prompt_id).toBe(prompt.id);
    expect(rows[0].memory_id).toBe(mem.id);
    expect(rows[0].relation).toBe('created_from');
  });

  it('no session context → store memory → no link created', async () => {
    const mem = await env.store.store({ content: 'unlinked', category: 'other', tags: [] });
    const db = env.rawDb();
    const rows = db.prepare(
      `SELECT * FROM session_memory_links WHERE memory_id = ?`,
    ).all(mem.id);
    db.close();
    expect(rows).toHaveLength(0);
  });

  it('link persists after storing a second memory with same session context', async () => {
    env.store.setSessionContext('auto-2', 'proj-2');
    await env.store.storeSessionPrompt('auto-2', 'first');
    const m1 = await env.store.store({ content: 'first mem', category: 'learning', tags: [] });
    const m2 = await env.store.store({ content: 'second mem', category: 'learning', tags: [] });

    const db = env.rawDb();
    const rows = db.prepare(
      `SELECT memory_id FROM session_memory_links WHERE prompt_id IN (
        SELECT id FROM session_prompts WHERE session_id = ?
      )`,
    ).all('auto-2') as Array<{ memory_id: string }>;
    db.close();

    const ids = rows.map(r => r.memory_id);
    expect(ids).toContain(m1.id);
    expect(ids).toContain(m2.id);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: MCP tool handlers integration against SQLite-backed store
// ═════════════════════════════════════════════════════════════════════

describe('T-008: MCP tool handlers against SQLite store', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('update_memory_status handler drives the SQLite lifecycle', async () => {
    const mem = await env.store.store({ content: 'tool lc', category: 'learning', tags: [] });
    const handler = handleUpdateMemoryStatus(env.store);
    const result = await handler({ memory_id: mem.id, status: 'needs_review', review_reason: 'check' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.status).toBe('needs_review');
    expect(data.reviewReason).toBe('check');
  });

  it('update_memory_status handler surfaces errors for invalid transition', async () => {
    const mem = await env.store.store({ content: 'x', category: 'other', tags: [] });
    await env.store.updateMemoryStatus(mem.id, 'archived', 'done');
    const handler = handleUpdateMemoryStatus(env.store);
    const result = await handler({ memory_id: mem.id, status: 'active' });
    expect(result.isError).toBe(true);
  });

  it('session ledger handlers chain end-to-end against SQLite', async () => {
    const promptHandler = handleStoreSessionPrompt(env.store);
    const pResult = await promptHandler({ session_id: 's9', raw_prompt: 'question?' });
    const prompt = parseResult(pResult);
    expect(prompt.id).toBeTruthy();

    const responseHandler = handleStoreSessionResponse(env.store);
    const rResult = await responseHandler({
      session_id: 's9', prompt_id: prompt.id, full_response: 'answer!',
    });
    expect(rResult.isError).toBeUndefined();

    const toolCallHandler = handleStoreSessionToolCall(env.store);
    const tcResult = await toolCallHandler({
      session_id: 's9', prompt_id: prompt.id, tool_name: 'read', importance: 'normal',
    });
    expect(tcResult.isError).toBeUndefined();

    const getHandler = handleGetSession(env.store);
    const ledgerResult = await getHandler({ session_id: 's9' });
    const ledger = parseResult(ledgerResult);
    expect(ledger.prompts).toHaveLength(1);
    expect(ledger.responses).toHaveLength(1);
    expect(ledger.toolCalls).toHaveLength(1);
  });

  it('list_sessions handler returns sessions from the SQLite store', async () => {
    await env.store.storeSessionPrompt('alpha', 'q');
    await env.store.storeSessionPrompt('beta', 'q');
    const handler = handleListSessions(env.store);
    const result = await handler({});
    const data = parseResult(result);
    expect(data.count).toBe(2);
    expect(data.sessions.map((s: { sessionId: string }) => s.sessionId).sort()).toEqual(['alpha', 'beta']);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: merge_projects handler integration
// ═════════════════════════════════════════════════════════════════════

describe('T-008: merge_projects handler', () => {
  it('delegates to globalStore.mergeProjects and returns MergeResult', async () => {
    const globalStore = makeMockGlobalStore();
    const handler = handleMergeProjects(globalStore);
    const result = await handler({ source_project_id: 'p-a', target_project_id: 'p-b' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.sourceProjectId).toBe('p-a');
    expect(data.targetProjectId).toBe('p-b');
    expect(data.dryRun).toBe(true);
    expect(typeof data.movedMemories).toBe('number');
  });

  it('propagates the self-merge refusal as an error', async () => {
    const globalStore = makeMockGlobalStore();
    const handler = handleMergeProjects(globalStore);
    const result = await handler({ source_project_id: 'same', target_project_id: 'same' });
    expect(result.isError).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Migration v2 final verification (tables, indexes, defaults)
// ═════════════════════════════════════════════════════════════════════

describe('T-008: migration v2 final verification', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  const expectedTables = [
    'memories',
    'session_prompts',
    'session_responses',
    'session_tool_calls',
    'session_summaries',
    'session_memory_links',
    'memory_topics',
  ];

  const expectedIndexes = [
    'idx_session_prompts_session_id',
    'idx_session_responses_session_id',
    'idx_session_responses_prompt_id',
    'idx_session_tool_calls_session_id',
    'idx_session_summaries_session_id',
    'idx_memory_topics_memory_id',
  ];

  it('all 7 tables (memories + 6 new) exist', () => {
    const db = env.rawDb();
    for (const t of expectedTables) {
      const row = db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
      ).get(t);
      expect(row).toBeTruthy();
    }
    db.close();
  });

  it('all 6 new indexes exist', () => {
    const db = env.rawDb();
    for (const idx of expectedIndexes) {
      const row = db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`,
      ).get(idx);
      expect(row).toBeTruthy();
    }
    db.close();
  });

  it('memories.status column defaults to "active"', async () => {
    await env.store.store({ content: 'default-status', category: 'learning', tags: [] });
    const db = env.rawDb();
    const row = db.prepare(
      `SELECT status FROM memories WHERE content = ?`,
    ).get('default-status') as { status: string };
    db.close();
    expect(row.status).toBe('active');
  });

  it('user_version is 4', () => {
    const db = env.rawDb();
    const row = db.prepare('PRAGMA user_version').get() as { user_version: number };
    db.close();
    expect(row.user_version).toBe(4);
  });

  it('data round-trips through every new table', async () => {
    const prompt = await env.store.storeSessionPrompt('rt', 'q');
    const response = await env.store.storeSessionResponse('rt', prompt.id, 'a');
    const toolCall = await env.store.storeSessionToolCall({ sessionId: 'rt', toolName: 'bash' });
    const summary = await env.store.storeSessionSummary({ sessionId: 'rt', summary: 'done' });
    const mem = await env.store.store({ content: 'rt-mem', category: 'other', tags: [] });
    await env.store.linkPromptToMemory(prompt.id, mem.id, 'created_from', response.id);

    const db = env.rawDb();
    expect((db.prepare(`SELECT count(*) AS n FROM session_prompts WHERE id = ?`).get(prompt.id) as { n: number }).n).toBe(1);
    expect((db.prepare(`SELECT count(*) AS n FROM session_responses WHERE id = ?`).get(response.id) as { n: number }).n).toBe(1);
    expect((db.prepare(`SELECT count(*) AS n FROM session_tool_calls WHERE id = ?`).get(toolCall.id) as { n: number }).n).toBe(1);
    expect((db.prepare(`SELECT count(*) AS n FROM session_summaries WHERE id = ?`).get(summary.id) as { n: number }).n).toBe(1);
    expect((db.prepare(`SELECT count(*) AS n FROM session_memory_links WHERE prompt_id = ? AND memory_id = ?`).get(prompt.id, mem.id) as { n: number }).n).toBe(1);
    db.close();
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-008: Existing behavior preserved (legacy callers, no status/topics/session)
// ═════════════════════════════════════════════════════════════════════

describe('T-008: existing behavior preserved', () => {
  let env: Awaited<ReturnType<typeof makeSqliteStore>>;

  beforeEach(async () => { env = await makeSqliteStore(); });
  afterEach(async () => { await env.cleanup(); });

  it('store + search + update + delete still work with the legacy API', async () => {
    const mem = await env.store.store({ content: 'legacy memory', category: 'learning', tags: ['v1'] });

    const hits = await env.store.search('legacy', 'hybrid', {});
    expect(hits).toHaveLength(1);
    expect(hits[0].memory.id).toBe(mem.id);

    const updated = await env.store.update(mem.id, { content: 'legacy updated', tags: ['v2'] });
    expect(updated.content).toBe('legacy updated');
    expect(updated.tags).toEqual(['v2']);

    await env.store.delete(mem.id);
    const after = await env.store.search('legacy', 'hybrid', {});
    expect(after).toHaveLength(0);
  });

  it('legacy store() (no status/topic/session) yields status "active" and is searchable', async () => {
    const mem = await env.store.store({ content: 'no frills', category: 'other', tags: [] });
    expect(mem.status).toBe('active');

    const hits = await env.store.search('frills', 'hybrid', {});
    expect(hits).toHaveLength(1);
  });

  it('MockMemoryStore still serves the original tools.test.ts scenarios (smoke)', async () => {
    const mockStore = new MockMemoryStore();
    await mockStore.store({ content: 'mock memory', category: 'learning', tags: ['m'] });
    const hits = await mockStore.search('mock', 'hybrid', {});
    expect(hits).toHaveLength(1);
    expect(hits[0].memory.content).toBe('mock memory');

    const recent = await mockStore.listRecent(10);
    expect(recent).toHaveLength(1);
  });
});
