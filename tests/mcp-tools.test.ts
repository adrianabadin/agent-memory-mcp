import { describe, it, expect, beforeEach } from 'vitest';
import { MockMemoryStore } from './mocks.js';
import {
  handleAttachSymbol,
  handleGetBySymbol,
  handleEnrichSymbols,
} from '../src/tools.js';
import type { EnrichSymbolResult } from '../src/types.js';

// ── Helpers ────────────────────────────────────────────────────────

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return JSON.parse(result.content[0].text);
}

// ── attach_symbol ──────────────────────────────────────────────────

describe('attach_symbol', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    // Seed a memory to attach to.
    await store.store({
      content: 'Function that validates email addresses',
      category: 'code-solution',
      tags: ['validation', 'email'],
    });
  });

  it('attaches a symbol key to an existing memory', async () => {
    const memId = store.memories[0].id;
    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/validate.ts|function|global|validateEmail|1',
      memory_id: memId,
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.symbolKey).toBe('ts|src/validate.ts|function|global|validateEmail|1');
    expect(data.id).toBe(memId);
  });

  it('returns error for non-existent memory_id', async () => {
    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/foo.ts|function|global|bar|0',
      memory_id: 'nonexistent-id',
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('not found');
  });

  it('upserts: overwrites previous symbol_key on same record', async () => {
    const memId = store.memories[0].id;
    const handler = handleAttachSymbol(store);

    // First attach
    await handler({
      symbol_key: 'ts|src/old.ts|function|global|oldFn|0',
      memory_id: memId,
    });

    // Second attach — should overwrite
    const result = await handler({
      symbol_key: 'ts|src/new.ts|function|global|newFn|1',
      memory_id: memId,
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.symbolKey).toBe('ts|src/new.ts|function|global|newFn|1');
  });

  it('attaches a symbol key to a session summary', async () => {
    // Store a session summary
    const prompt = await store.storeSessionPrompt('sess-attach', 'test prompt');
    const summary = await store.storeSessionSummary({
      sessionId: 'sess-attach',
      promptId: prompt.id,
      summary: 'Test summary for attach',
    });

    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/mod.ts|class|module|MyClass|0',
      session_item_id: summary.id,
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.symbolKey).toBe('ts|src/mod.ts|class|module|MyClass|0');
    expect(data.id).toBe(summary.id);
  });

  it('returns error when neither memory_id nor session_item_id is provided', async () => {
    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/foo.ts|function|global|bar|0',
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('memory_id or session_item_id');
  });

  it('attaches a symbol key to a session prompt', async () => {
    const prompt = await store.storeSessionPrompt('sess-prompt-attach', 'test prompt for attach');

    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/input.ts|function|global|parseInput|1',
      session_item_id: prompt.id,
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.symbolKey).toBe('ts|src/input.ts|function|global|parseInput|1');
    expect(data.id).toBe(prompt.id);
  });

  it('returns error for non-existent session_item_id', async () => {
    const handler = handleAttachSymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/foo.ts|function|global|bar|0',
      session_item_id: 'nonexistent-session-item',
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('not found');
  });
});

// ── get_by_symbol ──────────────────────────────────────────────────

describe('get_by_symbol', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();

    // Seed memories — one with symbol_key, one without.
    const mem1 = await store.store({
      content: 'JWT auth middleware implementation',
      category: 'architecture',
      tags: ['auth', 'jwt'],
    });
    const mem2 = await store.store({
      content: 'Database connection pooling',
      category: 'architecture',
      tags: ['database'],
    });

    // Manually set symbol_key on memories via the internal Map (keyed by string id).
    const mem1Ref = (store as any).memoriesById.get(mem1.id) as any;
    mem1Ref.symbolKey = 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1';
    const mem2Ref = (store as any).memoriesById.get(mem2.id) as any;
    mem2Ref.symbolKey = 'ts|src/db/pool.ts|class|module|ConnectionPool|0';

    // Seed a session summary with symbol_key.
    const prompt = await store.storeSessionPrompt('sess-sym', 'test prompt');
    await store.storeSessionSummary({
      sessionId: 'sess-sym',
      promptId: prompt.id,
      summary: 'Session about JWT implementation',
    });
    // Set symbol_key on the summary.
    const summaries = store.summaries;
    if (summaries.length > 0) {
      (summaries[0] as any).symbolKey = 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1';
    }
  });

  it('returns memories linked to a symbol key', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1',
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results.length).toBeGreaterThanOrEqual(1);
    // All results should have the requested symbol key.
    for (const r of data.results) {
      expect(r.symbolKey).toBe('ts|src/auth/middleware.ts|function|global|jwtMiddleware|1');
    }
  });

  it('returns empty results for unknown symbol key (not an error)', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/nonexistent.ts|function|global|missing|0',
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results).toEqual([]);
  });

  it('respects sources filter: memory only', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1',
      sources: ['memory'],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    // Should only have memory results, no session results.
    for (const r of data.results) {
      expect(r.source).toBe('memory');
    }
  });

  it('respects limit parameter', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1',
      limit: 1,
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results.length).toBeLessThanOrEqual(1);
  });

  it('includes session items when source is specified', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1',
      sources: ['memory', 'session_summary'],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    // Should have at least one result (the memory, and possibly the summary).
    expect(data.results.length).toBeGreaterThanOrEqual(1);
    const sources = data.results.map((r: any) => r.source);
    expect(sources).toContain('memory');
  });

  it('returns results with correct shape', async () => {
    const handler = handleGetBySymbol(store);
    const result = await handler({
      symbol_key: 'ts|src/db/pool.ts|class|module|ConnectionPool|0',
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results.length).toBeGreaterThanOrEqual(1);
    const r = data.results[0];
    expect(r).toHaveProperty('id');
    expect(r).toHaveProperty('source');
    expect(r).toHaveProperty('symbolKey');
    expect(r).toHaveProperty('content');
    expect(r).toHaveProperty('createdAt');
    expect(r.symbolKey).toBe('ts|src/db/pool.ts|class|module|ConnectionPool|0');
  });
});

// ── enrich_symbols ─────────────────────────────────────────────────

describe('enrich_symbols', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    // Seed a memory with enrichment-like content.
    await store.store({
      content: 'Purpose: Validates JWT tokens in HTTP middleware.\nInputs: Authorization header.\nOutputs: Decoded token or 401.',
      category: 'architecture',
      tags: ['enrichment', 'jwt'],
    });
  });

  it('returns per-symbol results for a batch of symbol keys', async () => {
    const handler = handleEnrichSymbols(store);
    const result = await handler({
      symbol_keys: [
        'ts|src/auth/middleware.ts|function|global|jwtMiddleware|1',
        'ts|src/db/pool.ts|class|module|ConnectionPool|0',
      ],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results).toHaveLength(2);
    for (const r of data.results) {
      expect(r.symbolKey).toBeDefined();
      expect(typeof r.found).toBe('boolean');
    }
  });

  it('returns found: false for unknown symbol key without failing batch', async () => {
    const handler = handleEnrichSymbols(store);
    const result = await handler({
      symbol_keys: ['ts|src/nonexistent.ts|function|global|missing|0'],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results).toHaveLength(1);
    expect(data.results[0].found).toBe(false);
    expect(data.results[0].raw).toBeUndefined();
  });

  it('rejects empty symbol_keys array', async () => {
    const handler = handleEnrichSymbols(store);
    const result = await handler({ symbol_keys: [] });

    expect(result.isError).toBe(true);
  });

  it('rejects oversized symbol_keys array', async () => {
    const handler = handleEnrichSymbols(store);
    const keys = Array.from({ length: 51 }, (_, i) => `key-${i}`);
    const result = await handler({ symbol_keys: keys });

    expect(result.isError).toBe(true);
  });

  it('returns enrichment from resolver when available', async () => {
    const resolver = async (key: string): Promise<EnrichSymbolResult | null> => {
      if (key === 'ts|src/auth.ts|function|global|jwt|1') {
        return {
          symbolKey: key,
          found: true,
          raw: 'Purpose: Validates JWT tokens.\nInputs: Authorization header.\nOutputs: Decoded token.',
          structured: {
            purpose: 'Validates JWT tokens',
            inputs: 'Authorization header',
            outputs: 'Decoded token',
          },
        };
      }
      return null;
    };

    const handler = handleEnrichSymbols(store, resolver);
    const result = await handler({
      symbol_keys: ['ts|src/auth.ts|function|global|jwt|1'],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results).toHaveLength(1);
    expect(data.results[0].found).toBe(true);
    expect(data.results[0].raw).toContain('Validates JWT');
    expect(data.results[0].structured.purpose).toBe('Validates JWT tokens');
  });

  it('handles mixed batch: resolver-found + store-found + not-found', async () => {
    // Seed a memory with a symbol key for store-based fallback.
    const mem = await store.store({
      content: 'Database pool configuration details',
      category: 'architecture',
      tags: ['db'],
    });
    const memRef = (store as any).memoriesById.get(mem.id) as any;
    memRef.symbolKey = 'ts|src/db.ts|class|module|Pool|0';

    const resolver = async (key: string): Promise<EnrichSymbolResult | null> => {
      if (key === 'ts|src/auth.ts|function|global|jwt|1') {
        return { symbolKey: key, found: true, raw: 'JWT enrichment from resolver' };
      }
      return null;
    };

    const handler = handleEnrichSymbols(store, resolver);
    const result = await handler({
      symbol_keys: [
        'ts|src/auth.ts|function|global|jwt|1',      // resolver-found
        'ts|src/db.ts|class|module|Pool|0',           // store-found
        'ts|src/missing.ts|function|global|nope|0',   // not-found
      ],
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.results).toHaveLength(3);

    // Resolver-found
    expect(data.results[0].found).toBe(true);
    expect(data.results[0].raw).toBe('JWT enrichment from resolver');

    // Store-found
    expect(data.results[1].found).toBe(true);
    expect(data.results[1].raw).toContain('Database pool');

    // Not-found
    expect(data.results[2].found).toBe(false);
    expect(data.results[2].raw).toBeUndefined();
  });
});
