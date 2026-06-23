import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { SqliteMemoryStore, parseSearchQuery } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import {
  RAW_SESSION_SOURCES,
  type SessionSearchResult,
  type UnifiedSearchResult,
} from '../src/types.js';

// ── Shared fixture ────────────────────────────────────────────────────

async function makeStore(): Promise<{
  store: SqliteMemoryStore;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'amm-search-'));
  const dbPath = join(tmpDir, 'test.db');
  const store = new SqliteMemoryStore(dbPath, new MockEmbedder());
  await store.initialize();
  return {
    store,
    cleanup: async () => {
      store.close();
      await rm(tmpDir, { recursive: true, force: true });
    },
  };
}

/**
 * Seed a session with a summary plus raw prompt/response/tool-call rows so
 * compacted-first and raw opt-in behavior can be exercised. All content
 * shares the token "alpha" so a single keyword query hits every source.
 */
async function seedSession(store: SqliteMemoryStore, sessionId: string): Promise<{ promptId: string }> {
  const prompt = await store.storeSessionPrompt(sessionId, 'alpha prompt about widgets');
  const response = await store.storeSessionResponse(sessionId, prompt.id, 'alpha response describing the widget fix in great detail');
  await store.storeSessionToolCall({
    sessionId,
    promptId: prompt.id,
    responseId: response.id,
    toolName: 'alpha_tool',
    resultSummary: 'alpha tool result summary',
  });
  await store.storeSessionSummary({
    sessionId,
    promptId: prompt.id,
    summary: 'alpha session summary of the widget work',
    decisions: ['chose alpha approach'],
  });
  return { promptId: prompt.id };
}

// ── Task 1.1: parameter parsing & defaults (RED → GREEN) ──────────────

describe('parseSearchQuery: defaults & validation', () => {
  it('applies documented defaults', () => {
    const resolved = parseSearchQuery({ query: 'alpha' });
    expect(resolved.limit).toBe(10);
    expect(resolved.status).toBe('active');
    expect(resolved.includeContent).toBe(false);
    expect(resolved.maxContentChars).toBe(4000);
    expect(resolved.includeRawSessionItems).toBe(false);
    expect(resolved.sourceBalancing).toBe(true);
    expect(resolved.explain).toBe(false);
  });

  it('honors explicit overrides', () => {
    const resolved = parseSearchQuery({
      query: 'alpha',
      limit: 5,
      status: 'all',
      includeContent: true,
      maxContentChars: 1000,
      includeRawSessionItems: true,
      sourceBalancing: false,
      explain: true,
      sessionId: 'sess-1',
    });
    expect(resolved.limit).toBe(5);
    expect(resolved.status).toBe('all');
    expect(resolved.includeContent).toBe(true);
    expect(resolved.maxContentChars).toBe(1000);
    expect(resolved.includeRawSessionItems).toBe(true);
    expect(resolved.sourceBalancing).toBe(false);
    expect(resolved.explain).toBe(true);
    expect(resolved.sessionId).toBe('sess-1');
  });

  it('rejects an empty query', () => {
    expect(() => parseSearchQuery({ query: '   ' })).toThrow(/query/i);
  });

  it('rejects a non-positive limit', () => {
    expect(() => parseSearchQuery({ query: 'alpha', limit: 0 })).toThrow(/limit/i);
    expect(() => parseSearchQuery({ query: 'alpha', limit: -3 })).toThrow(/limit/i);
  });

  it('rejects a negative maxContentChars', () => {
    expect(() => parseSearchQuery({ query: 'alpha', maxContentChars: -1 })).toThrow(/maxContentChars/i);
  });

  it('rejects unknown sources', () => {
    expect(() =>
      parseSearchQuery({ query: 'alpha', sources: ['memory', 'bogus' as never] }),
    ).toThrow(/source/i);
  });

  it('defaults sources to session ledger sources when omitted', () => {
    const resolved = parseSearchQuery({ query: 'alpha' });
    expect(resolved.sources).toContain('session_summary');
    expect(resolved.sources.length).toBeGreaterThan(0);
  });

  // projectId is part of the unified query contract, but the session ledger
  // schema carries no per-row project association (only session_id). Silently
  // ignoring projectId would return cross-project rows under a project filter,
  // so the parser MUST reject it until the schema supports project scoping.
  it('rejects projectId for session search (no schema support yet)', () => {
    expect(() => parseSearchQuery({ query: 'alpha', projectId: 'proj-1' })).toThrow(/projectId/i);
  });

  it('still accepts queries without projectId', () => {
    expect(() => parseSearchQuery({ query: 'alpha', sessionId: 'sess-1' })).not.toThrow();
  });
});

// ── Task 1.3/1.4: result schema, compacted-first, balancing ───────────

describe('searchSessions: result schema & compacted-first', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
    await seedSession(store, 'sess-A');
  });

  afterEach(async () => {
    await cleanup();
  });

  it('returns results with the uniform schema', async () => {
    const results = await store.searchSessions({ query: 'alpha' });
    expect(results.length).toBeGreaterThan(0);
    const r = results[0];
    expect(typeof r.id).toBe('string');
    expect(typeof r.source).toBe('string');
    expect(typeof r.contentRef).toBe('string');
    expect(typeof r.score).toBe('number');
    expect(r.status).toBe('active');
    expect(typeof r.createdAt).toBe('string');
    expect(r.decisionHint).toBeDefined();
    expect(r.sessionId).toBe('sess-A');
    expect(typeof r.isCompact).toBe('boolean');
  });

  it('omits full content by default (includeContent=false)', async () => {
    const results = await store.searchSessions({ query: 'alpha' });
    for (const r of results) {
      expect(r.content).toBeUndefined();
    }
  });

  it('includes content when includeContent=true', async () => {
    const results = await store.searchSessions({ query: 'alpha', includeContent: true });
    const withContent = results.find(r => typeof r.content === 'string');
    expect(withContent).toBeDefined();
    expect(withContent!.content!.length).toBeGreaterThan(0);
  });

  it('searches compacted summaries first; raw rows are opt-in', async () => {
    const compacted = await store.searchSessions({ query: 'alpha' });
    const rawSources = compacted.map(r => r.source);
    // Default excludes raw prompt/response/tool_call sources.
    for (const raw of RAW_SESSION_SOURCES) {
      expect(rawSources).not.toContain(raw);
    }
    // Summary IS searched by default.
    expect(rawSources).toContain('session_summary');
  });

  it('includes raw session rows when includeRawSessionItems=true', async () => {
    const all = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
      limit: 20,
      sourceBalancing: false,
    });
    const sources = new Set(all.map(r => r.source));
    expect(sources.has('session_summary')).toBe(true);
    expect(
      sources.has('session_prompt') ||
      sources.has('session_response') ||
      sources.has('session_tool_call'),
    ).toBe(true);
  });

  it('respects the limit', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
      limit: 2,
    });
    expect(results.length).toBeLessThanOrEqual(2);
  });

  it('filters by sessionId', async () => {
    await seedSession(store, 'sess-B');
    const results = await store.searchSessions({
      query: 'alpha',
      sessionId: 'sess-A',
      includeRawSessionItems: true,
      limit: 50,
    });
    for (const r of results) {
      expect((r as SessionSearchResult).sessionId).toBe('sess-A');
    }
  });

  it('returns no duplicate ids', async () => {
    const results = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
      limit: 50,
    });
    const ids = results.map(r => r.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

// ── Task 1.3: source balancing ────────────────────────────────────────

describe('searchSessions: source balancing', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('prevents one verbose source from crowding out the others', async () => {
    // Many raw responses, one summary. With balancing on, the summary must
    // still surface within a small limit instead of being buried.
    for (let i = 0; i < 8; i++) {
      const p = await store.storeSessionPrompt('sess-bal', `alpha prompt ${i}`);
      await store.storeSessionResponse('sess-bal', p.id, `alpha response number ${i} with lots of detail`);
    }
    await store.storeSessionSummary({ sessionId: 'sess-bal', summary: 'alpha summary distinct gist' });

    const balanced = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
      sourceBalancing: true,
      limit: 4,
    });
    expect(balanced.some(r => r.source === 'session_summary')).toBe(true);
  });
});

// ── Task 1.6/1.7: DecisionHint ────────────────────────────────────────

describe('searchSessions: DecisionHint generation', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
  });

  afterEach(async () => {
    await cleanup();
  });

  it('summarizes long content to ≤200 chars and flags truncation', async () => {
    const long = 'alpha ' + 'x'.repeat(500);
    const p = await store.storeSessionPrompt('sess-hint', 'alpha short prompt');
    await store.storeSessionResponse('sess-hint', p.id, long);

    const results = await store.searchSessions({
      query: 'alpha',
      includeRawSessionItems: true,
      limit: 50,
    });
    const r = results.find(x => x.source === 'session_response');
    expect(r).toBeDefined();
    expect(r!.decisionHint.summary.length).toBeLessThanOrEqual(200);
    expect(r!.decisionHint.hasTruncatedDetail).toBe(true);
    expect(r!.decisionHint.relevanceScore).toBeGreaterThanOrEqual(0);
    expect(r!.decisionHint.relevanceScore).toBeLessThanOrEqual(1);
  });

  it('does not flag truncation for short content', async () => {
    await store.storeSessionSummary({ sessionId: 'sess-hint2', summary: 'alpha tiny summary' });
    const results = await store.searchSessions({ query: 'alpha' });
    const r = results[0];
    expect(r.decisionHint.hasTruncatedDetail).toBe(false);
  });

  it('exposes a contentRef handle for drill-down', async () => {
    const p = await store.storeSessionPrompt('sess-ref', 'alpha prompt ref');
    await store.storeSessionSummary({ sessionId: 'sess-ref', summary: 'alpha summary ref', promptId: p.id });
    const results = await store.searchSessions({ query: 'alpha' });
    const r = results[0];
    // contentRef encodes the source type and id for retrieval.
    expect(r.contentRef).toMatch(/^session_summary:/);
    expect(r.contentRef).toContain(r.id);
  });
});

// ── Task 1.8: explain mode ────────────────────────────────────────────

describe('searchSessions: explain mode', () => {
  let store: SqliteMemoryStore;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ store, cleanup } = await makeStore());
    await seedSession(store, 'sess-exp');
  });

  afterEach(async () => {
    await cleanup();
  });

  it('omits explain by default', async () => {
    const results = await store.searchSessions({ query: 'alpha' });
    for (const r of results) {
      expect(r.explain).toBeUndefined();
    }
  });

  it('includes scoring breakdown when explain=true', async () => {
    const results = await store.searchSessions({ query: 'alpha', explain: true });
    expect(results.length).toBeGreaterThan(0);
    const e = (results[0] as UnifiedSearchResult).explain;
    expect(e).toBeDefined();
    expect(typeof e!.keywordRank).toBe('number');
    expect(typeof e!.semanticCosine).toBe('number');
    expect(e!.sourceMultiplier).toBe(1);
    expect(typeof e!.decayFactor).toBe('number');
    expect(e!.accessMultiplier).toBe(1);
    expect(typeof e!.finalScore).toBe('number');
  });
});
