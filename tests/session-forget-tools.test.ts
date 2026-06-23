// ─────────────────────────────────────────────────────────────────────
// MCP tool-handler tests for the session forget / purge tools.
//
// Scope: the TOOL LAYER only — handler factories `handleForgetSessionItem`
// and `handlePurgeSessionItem`, and their `registerTools` wiring. These
// verify that the handlers:
//   - return { type, success } on the happy path,
//   - safely surface store errors (NOT_FOUND / AMBIGUOUS) as `isError`
//     results instead of throwing out of the MCP call,
//   - clearly distinguish soft forget (status='forgotten', row recoverable)
//     from hard purge (row + embedding removed, destructive).
//
// The STORE-LEVEL behavior (entity-type detection, ambiguity refusal,
// embedding removal) is covered by `session-decay-forget.test.ts`. These
// handler tests deliberately reuse that proven contract rather than
// duplicating it.
// ─────────────────────────────────────────────────────────────────────

import { describe, it, expect, beforeEach } from 'vitest';
import { MockMemoryStore } from './mocks.js';
import {
  handleForgetSessionItem,
  handlePurgeSessionItem,
  registerTools,
} from '../src/tools.js';
import type { MemoryStore } from '../src/types.js';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

// Reuse the same parse helper convention as the rest of the tool tests.
function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return JSON.parse(result.content[0].text);
}

// A store stub that throws on demand, used to exercise the handler's
// error-surfacing path for the (rare) AMBIGUOUS case which the
// MockMemoryStore cannot reach through its public API.
function makeThrowingStore(errorText: string): MemoryStore {
  const boom = async () => { throw new Error(errorText); };
  return {
    forgetSessionItem: boom,
    purgeSessionItem: boom,
  } as unknown as MemoryStore;
}

// ═════════════════════════════════════════════════════════════════════
// forget_session_item — soft delete
// ═════════════════════════════════════════════════════════════════════

describe('forget_session_item (handler)', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('soft-forgets a prompt and reports { type, success }', async () => {
    const prompt = await store.storeSessionPrompt('s1', 'forget me');
    const handler = handleForgetSessionItem(store);
    const result = await handler({ id: prompt.id });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data).toEqual({ type: 'prompt', success: true });

    // Soft delete: the row still exists, just marked forgotten (recoverable).
    expect(store.prompts[0].status).toBe('forgotten');
  });

  it('soft-forgets a summary and reports type "summary"', async () => {
    const summary = await store.storeSessionSummary({ sessionId: 's1', summary: 'to forget' });
    const handler = handleForgetSessionItem(store);
    const result = await handler({ id: summary.id });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data).toEqual({ type: 'summary', success: true });
    expect(store.summaries[0].status).toBe('forgotten');
  });

  it('returns an isError result for an unknown id (NOT_FOUND)', async () => {
    const handler = handleForgetSessionItem(store);
    const result = await handler({ id: 'does-not-exist' });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toMatch(/not found|NOT_FOUND/i);
  });

  it('returns an isError result for an ambiguous id (AMBIGUOUS)', async () => {
    const handler = handleForgetSessionItem(makeThrowingStore('AMBIGUOUS: id x matches multiple tables'));
    const result = await handler({ id: 'x' });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toMatch(/ambiguous|AMBIGUOUS/i);
  });
});

// ═════════════════════════════════════════════════════════════════════
// purge_session_item — hard delete (destructive)
// ═════════════════════════════════════════════════════════════════════

describe('purge_session_item (handler)', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('hard-purges a summary and reports { type, success }', async () => {
    const summary = await store.storeSessionSummary({ sessionId: 's1', summary: 'to purge' });
    const handler = handlePurgeSessionItem(store);
    const result = await handler({ id: summary.id });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data).toEqual({ type: 'summary', success: true });
  });

  it('is destructive: the purged item is removed from the store', async () => {
    const summary = await store.storeSessionSummary({ sessionId: 's1', summary: 'gone for good' });
    const id = summary.id;

    expect(store.summaries).toHaveLength(1);

    const handler = handlePurgeSessionItem(store);
    await handler({ id });

    // Hard delete: row is gone entirely (unlike forget, which keeps it).
    expect(store.summaries).toHaveLength(0);
  });

  it('hard-purges a prompt and reports type "prompt"', async () => {
    const prompt = await store.storeSessionPrompt('s1', 'purge me');
    const handler = handlePurgeSessionItem(store);
    const result = await handler({ id: prompt.id });

    const data = parseResult(result);
    expect(data).toEqual({ type: 'prompt', success: true });
    expect(store.prompts).toHaveLength(0);
  });

  it('returns an isError result for an unknown id (NOT_FOUND)', async () => {
    const handler = handlePurgeSessionItem(store);
    const result = await handler({ id: 'no-such-id' });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toMatch(/not found|NOT_FOUND/i);
  });

  it('returns an isError result for an ambiguous id (AMBIGUOUS)', async () => {
    const handler = handlePurgeSessionItem(makeThrowingStore('AMBIGUOUS: id y matches multiple tables'));
    const result = await handler({ id: 'y' });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toMatch(/ambiguous|AMBIGUOUS/i);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Soft vs hard distinction
// ═════════════════════════════════════════════════════════════════════

describe('forget vs purge semantics', () => {
  it('forget hides but keeps the row; purge removes it', async () => {
    const store = new MockMemoryStore();
    const a = await store.storeSessionPrompt('s1', 'soft');
    const b = await store.storeSessionPrompt('s1', 'hard');

    await handleForgetSessionItem(store)({ id: a.id });
    await handlePurgeSessionItem(store)({ id: b.id });

    // After forget: prompt 'a' still present, status forgotten.
    const forgotten = store.prompts.find(p => p.id === a.id);
    expect(forgotten).toBeDefined();
    expect(forgotten!.status).toBe('forgotten');

    // After purge: prompt 'b' is gone entirely.
    expect(store.prompts.find(p => p.id === b.id)).toBeUndefined();

    // Default getSession hides the forgotten row (decayed out of view).
    const ledger = await store.getSession('s1');
    expect(ledger.prompts).toHaveLength(0);
  });
});

// ═════════════════════════════════════════════════════════════════════
// registerTools wiring
// ═════════════════════════════════════════════════════════════════════

describe('registerTools wires forget_session_item + purge_session_item', () => {
  it('registers both forget and purge tools (and keeps prior session tools)', () => {
    const registered = new Map<string, string>(); // name -> description
    const fakeServer = {
      tool(name: string, description: string, _schema: unknown, _handler: unknown) {
        registered.set(name, description);
      },
    } as unknown as McpServer;

    registerTools(fakeServer, new MockMemoryStore());

    // New explicit-lifecycle tools present.
    expect(registered.has('forget_session_item')).toBe(true);
    expect(registered.has('purge_session_item')).toBe(true);

    // Tool descriptions must clearly distinguish soft forget from hard purge,
    // so the destructive tool is not exposed casually.
    const forgetDesc = registered.get('forget_session_item') ?? '';
    const purgeDesc = registered.get('purge_session_item') ?? '';
    expect(forgetDesc.toLowerCase()).toContain('soft');
    expect(forgetDesc.toLowerCase()).toMatch(/hidden|recoverable/);
    expect(purgeDesc.toLowerCase()).toMatch(/permanent|irreversible|destructive/);

    // No regression: the existing session-ledger tools are still registered.
    expect(registered.has('store_session_prompt')).toBe(true);
    expect(registered.has('get_session')).toBe(true);
    expect(registered.has('link_prompt_to_memory')).toBe(true);
  });
});
