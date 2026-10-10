// tests/embedder-e5.test.ts
//
// M1 (E5 small migration): role-aware E5 embedding support.
//   - query:/passage: role prefixes applied at inference only
//   - token-accurate overlapping chunking (<=512 incl. prefix+specials)
//   - exact coverage (no token dropped, no silent truncation)
//   - sequential token-weighted normalized aggregation into ONE vector
//   - chunk policy version participates in model/cache identity
//   - BGE default (1024/cls) path unchanged: no prefix, no chunking
//
// All tokenizer/extractor interactions are mocked — no network, no model.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  TransformersEmbedder,
  E5_MODEL,
  E5_CHUNK_POLICY,
  E5_ROLE_PREFIXES,
  planTokenChunks,
  aggregateChunkVectors,
  type EmbedderTokenizer,
} from '../src/embedder.js';
import { EmbeddingCache } from '../src/embedding-cache.js';

// ── Mock tokenizer (whitespace word-level, deterministic) ────────────

interface MockTokenizer extends EmbedderTokenizer {
  encodeCalls: Array<{ text: string; addSpecialTokens: boolean }>;
}

function makeMockTokenizer(): MockTokenizer {
  const vocab = new Map<string, number>(); // word -> id (ids start at 10)
  const reverse = new Map<number, string>();
  const encodeCalls: Array<{ text: string; addSpecialTokens: boolean }> = [];
  return {
    encodeCalls,
    encode(text: string, opts?: { add_special_tokens?: boolean }): number[] {
      const addSpecialTokens = opts?.add_special_tokens !== false;
      encodeCalls.push({ text, addSpecialTokens });
      const words = text.split(/\s+/).filter(w => w.length > 0);
      const ids = words.map(w => {
        let id = vocab.get(w);
        if (id === undefined) {
          id = vocab.size + 10;
          vocab.set(w, id);
          reverse.set(id, w);
        }
        return id;
      });
      // CLS=1 / SEP=2 specials for empty-text special count.
      return addSpecialTokens ? [1, ...ids, 2] : ids;
    },
    decode(ids: number[], _opts?: { skip_special_tokens?: boolean }): string {
      void _opts;
      return ids.map(id => reverse.get(id) ?? `?${id}`).join(' ');
    },
  } as MockTokenizer;
}

// ── Mock extractor (deterministic per-string vectors, call recording) ─

function hashVector(text: string, dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < text.length; i++) v[i % dim] += (text.charCodeAt(i) + 1) / 997;
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return Float32Array.from(v, x => x / norm);
}

interface MockExtractor {
  (text: string | string[], options: { pooling: string; normalize: boolean }): Promise<{ data: Float32Array }>;
}

interface MockExtractorHandle {
  extractor: MockExtractor;
  calls: string[];
  sawConcurrent: boolean;
}

function makeMockExtractor(dim: number): MockExtractorHandle {
  const calls: string[] = [];
  const handle: MockExtractorHandle = { extractor: null as never, calls, sawConcurrent: false };
  let inflight = 0;
  handle.extractor = async (text) => {
    inflight++;
    if (inflight > 1) handle.sawConcurrent = true;
    const arr = Array.isArray(text) ? text : [text];
    for (const t of arr) calls.push(t);
    const data = new Float32Array(arr.length * dim);
    arr.forEach((t, i) => data.set(hashVector(t, dim), i * dim));
    inflight--;
    return { data };
  };
  return handle;
}

function makeE5(overrides?: {
  cache?: EmbeddingCache;
  tokenizer?: MockTokenizer;
  extractor?: MockExtractor;
}) {
  const tokenizer = overrides?.tokenizer ?? makeMockTokenizer();
  const handle = overrides?.extractor
    ? { extractor: overrides.extractor, calls: [] as string[], sawConcurrent: false }
    : makeMockExtractor(384);
  const embedder = new TransformersEmbedder(E5_MODEL, undefined, undefined, overrides?.cache, {
    tokenizer,
    extractor: handle.extractor,
  });
  return { embedder, tokenizer, handle };
}

const words = (n: number): string[] => Array.from({ length: n }, (_, i) => `w${i}`);

describe('E5 role prefixes (inference only)', () => {
  it('defaults to the passage role and prefixes at inference only', async () => {
    const { embedder, handle } = makeE5();
    await embedder.initialize();
    const vec = await embedder.embed('hello world');
    expect(handle.calls).toEqual(['passage: hello world']);
    expect(vec.length).toBe(384);
    // L2-normalized output.
    expect(Math.sqrt(vec.reduce((s, v) => s + v * v, 0))).toBeCloseTo(1, 5);
  });

  it('uses the query prefix for explicit query role', async () => {
    const { embedder, handle } = makeE5();
    await embedder.initialize();
    await embedder.embed('hello world', 'query');
    expect(handle.calls).toEqual(['query: hello world']);
  });

  it('exposes stable role prefixes and chunk policy identity', () => {
    expect(E5_ROLE_PREFIXES).toEqual({ query: 'query: ', passage: 'passage: ' });
    expect(E5_CHUNK_POLICY).toEqual({
      version: 'e5-fulltext-v1',
      maxTokens: 512,
      overlapTokens: 64,
    });
  });
});

describe('planTokenChunks — exact coverage, overlap, no truncation', () => {
  const covered = (spans: Array<{ start: number; end: number }>, n: number): number[] => {
    const counts = new Array<number>(n).fill(0);
    for (const s of spans) {
      for (let i = s.start; i < s.end; i++) counts[i]++;
    }
    return counts;
  };

  it('returns a single full span when tokens fit the budget', () => {
    expect(planTokenChunks(509, 509, 64)).toEqual([{ start: 0, end: 509 }]);
    expect(planTokenChunks(1, 509, 64)).toEqual([{ start: 0, end: 1 }]);
  });

  it('returns no spans for empty input', () => {
    expect(planTokenChunks(0, 509, 64)).toEqual([]);
  });

  it('covers every token exactly (no gaps) with exact overlap on a long input', () => {
    const n = 1000;
    const spans = planTokenChunks(n, 509, 64);
    expect(spans.length).toBe(3);
    expect(spans).toEqual([
      { start: 0, end: 509 },
      { start: 445, end: 954 },
      { start: 890, end: 1000 },
    ]);
    const counts = covered(spans, n);
    // Complete coverage: every token covered at least once, no overflow past n.
    expect(counts.every(c => c >= 1)).toBe(true);
    // Overlap is exactly 64 tokens between consecutive chunks.
    const totalLen = spans.reduce((s, sp) => s + (sp.end - sp.start), 0);
    expect(totalLen).toBe(n + 64 * (spans.length - 1));
    // No chunk exceeds the budget.
    expect(Math.max(...spans.map(s => s.end - s.start))).toBeLessThanOrEqual(509);
    // First chunk starts at 0, last chunk ends at n (no truncation).
    expect(spans[0].start).toBe(0);
    expect(spans[spans.length - 1].end).toBe(n);
  });

  it('emits a second chunk for a single token over budget (no silent truncation)', () => {
    const spans = planTokenChunks(510, 509, 64);
    expect(spans).toEqual([
      { start: 0, end: 509 },
      { start: 445, end: 510 },
    ]);
    expect(covered(spans, 510).every(c => c >= 1)).toBe(true);
  });

  it('rejects an invalid overlap (>= budget)', () => {
    expect(() => planTokenChunks(100, 10, 10)).toThrow(/overlap/);
    expect(() => planTokenChunks(100, 10, 11)).toThrow(/overlap/);
  });

  it('rejects an invalid budget', () => {
    expect(() => planTokenChunks(100, 0, 0)).toThrow(/budget/);
  });
});

describe('E5 long-text chunked embedding (end-to-end with mocks)', () => {
  // Budget math with the mock tokenizer: prefix "passage: " = 1 token,
  // specials (CLS+SEP) = 2 → content budget = 512 - 1 - 2 = 509.
  const CONTENT_BUDGET = 509;

  it('chunks by actual tokens, keeps every chunk <=512 incl. prefix+specials, and aggregates to one vector', async () => {
    const tokenizer = makeMockTokenizer();
    const { embedder, handle } = makeE5({ tokenizer });
    await embedder.initialize();

    const n = 1000;
    const text = words(n).join(' ');
    const vec = await embedder.embed(text);

    const spans = planTokenChunks(n, CONTENT_BUDGET, E5_CHUNK_POLICY.overlapTokens);
    expect(handle.calls.length).toBe(spans.length);
    expect(handle.sawConcurrent).toBe(false); // sequential, one chunk at a time

    // Each call re-encodes to <= 512 tokens INCLUDING prefix and specials,
    // and corresponds exactly to its chunk span (prefix at inference only).
    spans.forEach((span, i) => {
      const expected = E5_ROLE_PREFIXES.passage + words(n).slice(span.start, span.end).join(' ');
      expect(handle.calls[i]).toBe(expected);
      const encodedTokens = tokenizer.encode(handle.calls[i], { add_special_tokens: true });
      expect(encodedTokens.length).toBeLessThanOrEqual(512);
      expect(encodedTokens.length).toBe(span.end - span.start + 3); // +prefix +CLS +SEP
    });

    // Exact coverage: chunk spans reconstruct the full token sequence.
    const coveredSet = new Set<number>();
    for (const s of spans) for (let i = s.start; i < s.end; i++) coveredSet.add(i);
    expect(coveredSet.size).toBe(n);

    // One aggregated, normalized384-dim vector.
    expect(vec.length).toBe(384);
    expect(Math.sqrt(vec.reduce((s, v) => s + v * v, 0))).toBeCloseTo(1, 5);
  });

  it('aggregates sequentially, token-count weighted, then L2-normalizes', async () => {
    const tokenizer = makeMockTokenizer();
    const { embedder, handle } = makeE5({ tokenizer });
    await embedder.initialize();

    const n = 1000;
    const text = words(n).join(' ');
    const vec = await embedder.embed(text);

    const spans = planTokenChunks(n, CONTENT_BUDGET, E5_CHUNK_POLICY.overlapTokens);
    // Sequential embedding: expected vector = normalize(sum(w_i * v_i) / sum(w_i)).
    const weights = spans.map(s => s.end - s.start);
    const vectors = handle.calls.map(c => Array.from(hashVector(c, 384)));
    const dim = 384;
    const agg = new Array<number>(dim).fill(0);
    let wSum = 0;
    vectors.forEach((v, i) => {
      wSum += weights[i];
      for (let d = 0; d < dim; d++) agg[d] += weights[i] * v[d];
    });
    const norm = Math.sqrt(agg.reduce((s, v) => s + v * v, 0));
    const expected = agg.map(v => v / norm);
    vec.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(1e-5));
  });

  it('uses the query prefix for chunked query embedding too', async () => {
    const tokenizer = makeMockTokenizer();
    const { embedder, handle } = makeE5({ tokenizer });
    await embedder.initialize();
    const text = words(600).join(' ');
    await embedder.embed(text, 'query');
    for (const call of handle.calls) {
      expect(call.startsWith('query: ')).toBe(true);
    }
    expect(handle.calls.length).toBeGreaterThan(1);
  });

  it('embedBatch for E5 routes every text through the passage role', async () => {
    const { embedder, handle } = makeE5();
    await embedder.initialize();
    const vecs = await embedder.embedBatch(['alpha beta', 'gamma']);
    expect(vecs.length).toBe(2);
    expect(handle.calls).toEqual(['passage: alpha beta', 'passage: gamma']);
  });
});

describe('aggregateChunkVectors (public API for M2 handoff)', () => {
  it('computes the normalized weighted mean', () => {
    const out = aggregateChunkVectors([[1, 0], [0, 1]], [1, 1]);
    expect(out[0]).toBeCloseTo(Math.SQRT1_2, 10);
    expect(out[1]).toBeCloseTo(Math.SQRT1_2, 10);

    const weighted = aggregateChunkVectors([[2, 0], [0, 2]], [3, 1]);
    // normalize(3*[2,0] + 1*[0,2]) = normalize([6,2])
    const norm = Math.sqrt(40);
    expect(weighted[0]).toBeCloseTo(6 / norm, 10);
    expect(weighted[1]).toBeCloseTo(2 / norm, 10);
  });

  it('rejects empty input and mismatched dimensions', () => {
    expect(() => aggregateChunkVectors([], [])).toThrow(/empty/);
    expect(() => aggregateChunkVectors([[1, 0], [0]], [1, 1])).toThrow(/dimension/);
  });
});

describe('BGE default path unchanged (1024 / cls, no prefix, no chunking)', () => {
  it('embeds raw text in a single call regardless of length or role', async () => {
    const handle = makeMockExtractor(1024);
    const bge = new TransformersEmbedder('Xenova/bge-m3', undefined, undefined, undefined, {
      extractor: handle.extractor,
    });
    await bge.initialize();
    const longText = words(2000).join(' ');
    const vec = await bge.embed(longText);
    expect(handle.calls).toEqual([longText]); // raw, unprefixed, single pass
    expect(vec.length).toBe(1024);

    await bge.embed('some query', 'query');
    expect(handle.calls[1]).toBe('some query'); // roles do not affect BGE

    const batch = await bge.embedBatch(['a', 'b']);
    expect(batch.length).toBe(2);
    // Batch path unchanged: one extractor invocation with the full array.
    expect(handle.calls[2]).toBe('a');
    expect(handle.calls[3]).toBe('b');
  });

  it('reports a single-pass (no chunking) identity', () => {
    const bge = new TransformersEmbedder('Xenova/bge-m3', undefined, undefined, undefined, {
      extractor: makeMockExtractor(1024).extractor,
    });
    expect(bge.identity?.()).toEqual({
      model: 'Xenova/bge-m3',
      dimensions: 1024,
      pooling: 'cls',
      normalize: true,
      roleAware: false,
      chunkPolicy: null,
    });
  });

  it('reports the E5 full-text identity with chunk policy version', () => {
    const { embedder } = makeE5();
    expect(embedder.identity?.()).toEqual({
      model: E5_MODEL,
      dimensions: 384,
      pooling: 'mean',
      normalize: true,
      roleAware: true,
      chunkPolicy: E5_CHUNK_POLICY,
    });
  });
});

describe('cache identity (model / role / pooling / chunk policy)', () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'pmc-e5-cache-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('separates cache entries by role and reuses them across instances with identical identity', async () => {
    const cache = new EmbeddingCache(dir, E5_MODEL);
    const a = makeE5({ cache });
    await a.embedder.initialize();
    await a.embedder.embed('cached text', 'query');
    expect(a.handle.calls.length).toBe(1);
    // Cache hit: same text+role does not hit the model again.
    await a.embedder.embed('cached text', 'query');
    expect(a.handle.calls.length).toBe(1);
    // Different role is a different cache identity.
    await a.embedder.embed('cached text', 'passage');
    expect(a.handle.calls.length).toBe(2);

    // A second embedder with identical model/policy identity reuses the cache.
    const b = makeE5({ cache });
    await b.embedder.initialize();
    const q = await b.embedder.embed('cached text', 'query');
    const p = await b.embedder.embed('cached text', 'passage');
    expect(b.handle.calls.length).toBe(0); // both served from cache
    expect(q).toEqual(Array.from(hashVector('query: cached text', 384)));
    expect(p).toEqual(Array.from(hashVector('passage: cached text', 384)));
  });

  it('does not share entries across model identities in the same directory', async () => {
    const cacheA = new EmbeddingCache(dir, E5_MODEL);
    const cacheB = new EmbeddingCache(dir, 'Xenova/bge-m3');
    await cacheA.set('shared text', [1, 2, 3], 'qual');
    expect(await cacheB.get('shared text', 'qual')).toBeNull();
    expect(await cacheA.get('shared text', 'other-qual')).toBeNull();
    expect(await cacheA.get('other text', 'qual')).toBeNull();
    expect(await cacheA.get('shared text', 'qual')).toEqual([1, 2, 3]);
  });
});
