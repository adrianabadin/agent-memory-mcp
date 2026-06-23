// ─────────────────────────────────────────────────────────────────────
// Phase 4 (PR #4) — End-to-end symbol resolution chain.
//
// These tests cross the full feature surface introduced by PRs #1–#3 and
// prove the spec's hardest end-to-end guarantees that no single earlier
// suite covers on its own:
//
//   1. Soft-link DURABILITY: a symbol_key attached to a memory survives a
//      full store close/reopen on the same .db file. This is the spec's
//      "later graph rebuilds do not null or rewrite [symbolKey]" guarantee —
//      the memory row is owned by the store, never by the volatile graph, so
//      reopening the DB (the closest store-level analogue of a graph rebuild)
//      must still resolve it via getBySymbol.
//
//   2. SOFT-MISS resolution: a renamed/removed symbol yields an empty result
//      from getBySymbol (not an error), while a still-valid sibling key
//      resolves normally — proving unresolved joins are tolerated.
//
//   3. CROSS-PROJECT session resolution: symbol-linked session items resolve
//      per project with default-deny on legacy (null project_id) rows, tying
//      together attachSymbol + setSessionContext + searchSessions + getBySymbol
//      in one real-store flow.
//
//   4. MIXED-BATCH resolution through the REAL store (not a mock): valid keys
//      resolve, unknown keys soft-miss, and the batch never fails.
// ─────────────────────────────────────────────────────────────────────

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import type { UnifiedSearchQuery } from '../src/types.js';

// ── Helpers ──────────────────────────────────────────────────────────

async function makeStoreEnv(): Promise<{
  store: SqliteMemoryStore;
  dbPath: string;
  reopen: () => Promise<SqliteMemoryStore>;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-symbol-e2e-'));
  const dbPath = join(tmpDir, 'test.db');
  let store = new SqliteMemoryStore(dbPath, new MockEmbedder());
  await store.initialize();
  return {
    get store() { return store; },
    dbPath,
    // Reopen the SAME .db file with a fresh store instance — the store-level
    // analogue of "the graph was rebuilt while the memory DB stayed put".
    reopen: async () => {
      store.close();
      store = new SqliteMemoryStore(dbPath, new MockEmbedder());
      await store.initialize();
      return store;
    },
    cleanup: async () => {
      store.close();
      await rm(tmpDir, { recursive: true, force: true });
    },
  } as {
    store: SqliteMemoryStore;
    dbPath: string;
    reopen: () => Promise<SqliteMemoryStore>;
    cleanup: () => Promise<void>;
  };
}

const SYMBOL_A = 'ts|src/auth/login.ts|function|global|login|2';
const SYMBOL_RENAMED = 'ts|src/auth/login.ts|function|global|authenticate|2';

// ═════════════════════════════════════════════════════════════════════
// E2E-1: Soft-link durability across store reopen (graph-rebuild analogue)
// ═════════════════════════════════════════════════════════════════════

describe('E2E-1: symbol_key survives a store close/reopen (rebuild durability)', () => {
  let env: Awaited<ReturnType<typeof makeStoreEnv>>;
  beforeEach(async () => { env = await makeStoreEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('attach → store memory → close → reopen → getBySymbol still resolves the exact key', async () => {
    const mem = await env.store.store({ content: 'login mints a JWT', category: 'architecture', tags: [] });
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, memoryId: mem.id });

    // Before reopen: resolves.
    const before = await env.store.getBySymbol(SYMBOL_A);
    expect(before).toHaveLength(1);
    expect(before[0].id).toBe(mem.id);

    // Simulate a graph rebuild: the memory DB is untouched, store reopened.
    const reopened = await env.reopen();

    const after = await reopened.getBySymbol(SYMBOL_A);
    expect(after).toHaveLength(1);
    expect(after[0].id).toBe(mem.id);
    expect(after[0].symbolKey).toBe(SYMBOL_A); // exact key retained, not rewritten
    expect(after[0].content).toBe('login mints a JWT');
  });

  it('a memory with NO symbol_key stays unlinked after reopen (no spurious links)', async () => {
    const linked = await env.store.store({ content: 'linked', category: 'learning', tags: [] });
    await env.store.store({ content: 'unlinked', category: 'learning', tags: [] });
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, memoryId: linked.id });

    const reopened = await env.reopen();
    const hits = await reopened.getBySymbol(SYMBOL_A);
    // Triangulation: only the explicitly-linked memory resolves; the unlinked
    // one must NOT appear, proving resolution keys off symbol_key not content.
    expect(hits).toHaveLength(1);
    expect(hits[0].id).toBe(linked.id);
  });
});

// ═════════════════════════════════════════════════════════════════════
// E2E-2: Soft-miss for a renamed/removed symbol (no error, sibling resolves)
// ═════════════════════════════════════════════════════════════════════

describe('E2E-2: soft-miss resolution for unresolved symbols', () => {
  let env: Awaited<ReturnType<typeof makeStoreEnv>>;
  beforeEach(async () => { env = await makeStoreEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('querying a renamed key soft-misses (empty, not error) while the original key resolves', async () => {
    const mem = await env.store.store({ content: 'auth flow notes', category: 'architecture', tags: [] });
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, memoryId: mem.id });

    // The symbol was renamed in the graph — its NEW key has no linked rows.
    const renamedHits = await env.store.getBySymbol(SYMBOL_RENAMED);
    expect(renamedHits).toEqual([]); // soft-miss: empty, no throw

    // The original durable key still resolves the record.
    const originalHits = await env.store.getBySymbol(SYMBOL_A);
    expect(originalHits).toHaveLength(1);
    expect(originalHits[0].id).toBe(mem.id);
  });
});

// ═════════════════════════════════════════════════════════════════════
// E2E-3: Cross-project session resolution chain (default-deny on legacy)
// ═════════════════════════════════════════════════════════════════════

describe('E2E-3: cross-project symbol-linked session resolution', () => {
  let env: Awaited<ReturnType<typeof makeStoreEnv>>;
  beforeEach(async () => { env = await makeStoreEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('session summaries scope per project AND resolve by symbol across the chain', async () => {
    // Project A session: summary stored, context set (backfills project_id),
    // then the summary is symbol-attached.
    const summaryA = await env.store.storeSessionSummary({ sessionId: 'sess-a', summary: 'login summary alpha A' });
    env.store.setSessionContext('sess-a', 'project-a');
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, sessionItemId: summaryA.id });

    // Project B session: a DIFFERENT project, same symbol attached to its summary.
    const summaryB = await env.store.storeSessionSummary({ sessionId: 'sess-b', summary: 'login summary alpha B' });
    env.store.setSessionContext('sess-b', 'project-b');
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, sessionItemId: summaryB.id });

    // (a) Project-scoped session search: project-a only sees its own session rows.
    const scoped = await env.store.searchSessions({
      query: 'login',
      projectId: 'project-a',
    } as UnifiedSearchQuery);
    expect(scoped.length).toBeGreaterThan(0);
    for (const r of scoped) {
      expect(r.sessionId).toBe('sess-a'); // default-deny excludes project-b's sess-b
    }
    expect(scoped.some(r => r.sessionId === 'sess-b')).toBe(false);

    // (b) Symbol resolution returns BOTH projects' summaries (symbol is global;
    // project scoping is a session-search concern, not a symbol-link concern).
    const bySymbol = await env.store.getBySymbol(SYMBOL_A, { sources: ['session_summary'] });
    const ids = bySymbol.map(r => r.id);
    expect(ids).toContain(summaryA.id);
    expect(ids).toContain(summaryB.id);
  });

  it('legacy session rows (null project_id) are excluded from a project-scoped search by default', async () => {
    // Legacy row: stored WITHOUT a session context → project_id stays null.
    await env.store.storeSessionSummary({ sessionId: 'sess-legacy', summary: 'legacy login note alpha' });

    // A scoped row in project-a (context set after store → backfill applies).
    await env.store.storeSessionSummary({ sessionId: 'sess-a', summary: 'scoped login note alpha' });
    env.store.setSessionContext('sess-a', 'project-a');

    const results = await env.store.searchSessions({
      query: 'login',
      projectId: 'project-a',
    } as UnifiedSearchQuery);
    const sessionIds = new Set(results.map(r => r.sessionId));

    expect(sessionIds.has('sess-a')).toBe(true);        // in-project row present
    expect(sessionIds.has('sess-legacy')).toBe(false);  // legacy null-project row denied
  });

  it('includeLegacy:true opts the legacy null-project row back in', async () => {
    await env.store.storeSessionSummary({ sessionId: 'sess-legacy', summary: 'legacy login note alpha' });
    await env.store.storeSessionSummary({ sessionId: 'sess-a', summary: 'scoped login note alpha' });
    env.store.setSessionContext('sess-a', 'project-a');

    const results = await env.store.searchSessions({
      query: 'login',
      projectId: 'project-a',
      includeLegacy: true,
    } as UnifiedSearchQuery);
    const sessionIds = new Set(results.map(r => r.sessionId));

    // Triangulation vs the previous test: same setup, includeLegacy flips the
    // legacy row from excluded to included, proving the flag drives the filter.
    expect(sessionIds.has('sess-legacy')).toBe(true);
    expect(sessionIds.has('sess-a')).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════
// E2E-4: Mixed-batch resolution through the REAL store
// ═════════════════════════════════════════════════════════════════════

describe('E2E-4: mixed valid/unknown symbol batch (real store)', () => {
  let env: Awaited<ReturnType<typeof makeStoreEnv>>;
  beforeEach(async () => { env = await makeStoreEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('resolves valid keys and soft-misses unknown keys without failing the batch', async () => {
    const mem = await env.store.store({ content: 'real memory', category: 'learning', tags: [] });
    await env.store.attachSymbol({ symbolKey: SYMBOL_A, memoryId: mem.id });

    const keys = [SYMBOL_A, 'ts|src/ghost.ts|function|global|missing|0'];
    const batch = await Promise.all(
      keys.map(async (key) => ({ key, hits: await env.store.getBySymbol(key) })),
    );

    const valid = batch.find(b => b.key === SYMBOL_A)!;
    const unknown = batch.find(b => b.key !== SYMBOL_A)!;

    expect(valid.hits).toHaveLength(1);
    expect(valid.hits[0].content).toBe('real memory');
    expect(unknown.hits).toEqual([]); // soft-miss, batch still completed
  });
});
