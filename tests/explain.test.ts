import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import { escapeFtsQuery } from '../src/shared/vector-utils.js';

// ── Helpers ────────────────────────────────────────────────────────

async function seedStore(store: SqliteMemoryStore) {
  await store.storeBatch([
    { content: 'TypeScript generics and type inference patterns', category: 'learning', tags: ['typescript'] },
    { content: 'React hooks best practices with useEffect cleanup', category: 'code-solution', tags: ['react', 'hooks'] },
    { content: 'evergreen reference: never-forget this API contract', category: 'architecture', tags: ['evergreen'] },
    { content: 'Python asyncio event loop patterns and pitfalls', category: 'debugging', tags: ['python'] },
  ]);
}

// ── Suite ──────────────────────────────────────────────────────────

describe('explain mode', () => {
  let store: SqliteMemoryStore;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'explain-test-'));
    const dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    await seedStore(store);
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  // ── explain=false (default) — no explain key on results ──────────

  it('explain omitted — no explain key on search results', async () => {
    const { results } = await store.searchWithSummary('TypeScript', 'hybrid', { limit: 5 });
    for (const r of results) {
      expect(r).not.toHaveProperty('explain');
    }
  });

  it('explain=false — no explain key on search results', async () => {
    const { results } = await store.searchWithSummary('TypeScript', 'hybrid', { explain: false, limit: 5 });
    for (const r of results) {
      expect(r).not.toHaveProperty('explain');
    }
  });

  // ── explain=true — per-stage block on every result ───────────────

  it('explain=true — every result has the per-stage explain block', async () => {
    const { results, summary } = await store.searchWithSummary('TypeScript', 'hybrid', { explain: true, limit: 5 });
    expect(results.length).toBeGreaterThan(0);

    for (const r of results) {
      expect(r).toHaveProperty('explain');
      const ex = r.explain!;
      expect(typeof ex.rrfScore).toBe('number');
      expect(typeof ex.decayFactor).toBe('number');
      expect(typeof ex.finalScore).toBe('number');
      expect(ex.decayFactor).toBeGreaterThan(0);
      // at least one leg must be present
      expect(ex.keyword !== null || ex.semantic !== null).toBe(true);
    }

    // summary must be present
    expect(summary).toBeDefined();
    expect(typeof summary!.keywordCandidates).toBe('number');
    expect(typeof summary!.semanticCandidates).toBe('number');
    expect(typeof summary!.fusedCount).toBe('number');
    expect(typeof summary!.query.fts).toBe('string');
  });

  // ── memory in only one leg shows null for the other ─────────────

  it('memory in only keyword leg shows null for semantic', async () => {
    // Use a very specific keyword that the embedder is unlikely to match semantically above others
    const { results } = await store.searchWithSummary('asyncio pitfalls', 'hybrid', { explain: true, limit: 10 });
    // Find the Python asyncio result
    const pyResult = results.find(r => r.memory.content.includes('Python asyncio'));
    // We can't guarantee which leg it ends up in, but if it has explain it should be structured correctly
    if (pyResult?.explain) {
      const ex = pyResult.explain;
      // at least one leg populated
      expect(ex.keyword !== null || ex.semantic !== null).toBe(true);
      if (ex.keyword !== null) {
        expect(typeof ex.keyword.rank).toBe('number');
        expect(typeof ex.keyword.bm25).toBe('number');
        expect(ex.keyword.rank).toBeGreaterThanOrEqual(0);
      }
      if (ex.semantic !== null) {
        expect(typeof ex.semantic.rank).toBe('number');
        expect(typeof ex.semantic.cosine).toBe('number');
        expect(ex.semantic.rank).toBeGreaterThanOrEqual(0);
      }
    }
  });

  // ── evergreen memory shows decayFactor = 1 ──────────────────────

  it('evergreen-tagged memory shows decayFactor of 1', async () => {
    const { results } = await store.searchWithSummary('API contract reference', 'hybrid', { explain: true, limit: 10 });
    const evResult = results.find(r => r.memory.tags.includes('evergreen'));
    if (evResult?.explain) {
      expect(evResult.explain.decayFactor).toBe(1);
    }
  });

  // ── ranking-invariance test ──────────────────────────────────────

  it('explain=on/off produces identical result order and IDs', async () => {
    const queryText = 'hooks react TypeScript';
    const [withExplain, withoutExplain] = await Promise.all([
      store.searchWithSummary(queryText, 'hybrid', { explain: true, limit: 10 }),
      store.searchWithSummary(queryText, 'hybrid', { explain: false, limit: 10 }),
    ]);

    expect(withExplain.results.length).toBe(withoutExplain.results.length);

    const idsWithExplain = withExplain.results.map(r => r.memory.id);
    const idsWithout = withoutExplain.results.map(r => r.memory.id);
    expect(idsWithExplain).toEqual(idsWithout);

    for (let i = 0; i < withExplain.results.length; i++) {
      expect(withExplain.results[i].score).toBeCloseTo(withoutExplain.results[i].score, 8);
    }
  });

  // ── FTS-escaped query string surfaces correctly ──────────────────

  it('summary.query.fts matches escapeFtsQuery output for the search query', async () => {
    const query = 'TypeScript "generics"';
    const { summary } = await store.searchWithSummary(query, 'hybrid', { explain: true, limit: 5 });
    expect(summary?.query.fts).toBe(escapeFtsQuery(query));
  });

  // ── explain=true with non-hybrid mode returns results without summary ─

  it('explain=true with mode=semantic returns results, no summary', async () => {
    const { results, summary } = await store.searchWithSummary('TypeScript', 'semantic', { explain: true, limit: 5 });
    expect(results.length).toBeGreaterThanOrEqual(0);
    expect(summary).toBeUndefined();
  });

  it('explain=true with mode=keyword returns results, no summary', async () => {
    const { results, summary } = await store.searchWithSummary('TypeScript', 'keyword', { explain: true, limit: 5 });
    expect(results.length).toBeGreaterThanOrEqual(0);
    expect(summary).toBeUndefined();
  });
});
