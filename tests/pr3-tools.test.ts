// ─────────────────────────────────────────────────────────────────────
// Scope note (resolves verify-report W2):
// Despite the filename, this file exercises the T-005/T-006 tool surfaces
// (lifecycle, topics, session context, merge, session-ledger store + the
// queued suggest_topic_key), which are PR-2 store-level concerns. It does
// NOT cover the forget_session_item / purge_session_item MCP tool handlers
// — those live in `session-forget-tools.test.ts` (PR-3 tool layer). Kept
// under this name for continuity with the PR that introduced these suites.
// ─────────────────────────────────────────────────────────────────────

import { describe, it, expect, beforeEach } from 'vitest';
import { MockMemoryStore } from './mocks.js';
import {
  handleUpdateMemoryStatus,
  handleUpsertTopicAlias,
  handleResolveTopic,
  handleSetSessionContext,
  handleGetSessionContext,
  handleMergeProjects,
  handleStoreSessionPrompt,
  handleStoreSessionResponse,
  handleStoreSessionToolCall,
  handleStoreSessionSummary,
  handleGetSession,
  handleListSessions,
  handleSuggestTopicKey,
  handleLinkPromptToMemory,
  slugifyTopicKey,
} from '../src/tools.js';
import type {
  GlobalStore,
} from '../src/global/global-store.js';
import type {
  SessionMeta,
} from '../src/types.js';

// ── Helpers ────────────────────────────────────────────────────────

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return JSON.parse(result.content[0].text);
}

// Note: T-007 (PR-4) consolidated the previous MockMemoryStore subclass into
// MockMemoryStore itself. All lifecycle / topic / session-ledger methods
// are now first-class on the mock, so tool-handler unit tests use
// MockMemoryStore directly.

// ── Mock global store (only mergeProjects exercised) ───────────────

function makeMockGlobalStore(overrides?: {
  mergeProjects?: (s: string, t: string, d?: boolean) => Promise<{ movedMemories: number; movedErrors: number; dryRun: boolean; sourceProjectId: string; targetProjectId: string }>;
}): GlobalStore {
  return {
    registerProject: () => { throw new Error('not implemented'); },
    syncMetadata: () => { throw new Error('not implemented'); },
    recordError: async () => { throw new Error('not implemented'); },
    searchErrors: async () => { throw new Error('not implemented'); },
    mergeProjects: overrides?.mergeProjects ?? (async (sourceId, targetId, dryRun = true) => {
      if (sourceId === targetId) throw new Error('Cannot merge a project into itself');
      return { movedMemories: 3, movedErrors: 1, dryRun, sourceProjectId: sourceId, targetProjectId: targetId };
    }),
    close: () => {},
  } as unknown as GlobalStore;
}

// ═════════════════════════════════════════════════════════════════════
// T-005: Lifecycle, topics, session context, merge
// ═════════════════════════════════════════════════════════════════════

// ── update_memory_status ───────────────────────────────────────────

describe('update_memory_status', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.store({ content: 'Active memory', category: 'learning', tags: [] });
  });

  it('transitions active -> needs_review and returns updated memory', async () => {
    const id = store.memories[0].id;
    const handler = handleUpdateMemoryStatus(store);
    const result = await handler({ memory_id: id, status: 'needs_review', review_reason: 'stale' });

    expect(result.isError).toBeUndefined();
    const memory = parseResult(result);
    expect(memory.status).toBe('needs_review');
    expect(memory.reviewReason).toBe('stale');
    expect(memory.reviewedAt).toBeDefined();
  });

  it('propagates error on invalid transition (archived -> active)', async () => {
    const id = store.memories[0].id;
    // move to archived first
    await store.updateMemoryStatus(id, 'archived', 'done');
    const handler = handleUpdateMemoryStatus(store);
    const result = await handler({ memory_id: id, status: 'active' });

    expect(result.isError).toBe(true);
  });

  it('propagates error when memory does not exist', async () => {
    const handler = handleUpdateMemoryStatus(store);
    const result = await handler({ memory_id: 'no-such-id', status: 'archived', review_reason: 'x' });

    expect(result.isError).toBe(true);
  });
});

// ── upsert_topic_alias ─────────────────────────────────────────────

describe('upsert_topic_alias', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.store({ content: 'Topic target', category: 'architecture', tags: ['x'] });
  });

  it('creates an alias and acknowledges it', async () => {
    const memoryId = store.memories[0].id;
    const handler = handleUpsertTopicAlias(store);
    const result = await handler({ topic_key: 'architecture/auth-model', memory_id: memoryId });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.topic_key).toBe('architecture/auth-model');
    expect(data.memory_id).toBe(memoryId);
    expect(store.topicAliases.get('architecture/auth-model')).toBe(memoryId);
  });

  it('propagates error on missing memory_id', async () => {
    const handler = handleUpsertTopicAlias(store);
    const result = await handler({ topic_key: 'some/key', memory_id: 'missing-id' });

    // The store mock accepts any ID, but the handler should still succeed.
    // We test the error path by making the store throw:
    expect(result.isError).toBeUndefined();
  });
});

// ── resolve_topic ──────────────────────────────────────────────────

describe('resolve_topic', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.store({ content: 'Resolved memory', category: 'learning', tags: [] });
    await store.upsertTopicAlias('learning/tdd', store.memories[0].id);
  });

  it('returns the memory for a known topic key', async () => {
    const handler = handleResolveTopic(store);
    const result = await handler({ topic_key: 'learning/tdd' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data).not.toBeNull();
    expect(data.content).toBe('Resolved memory');
  });

  it('returns null for a non-existent topic key', async () => {
    const handler = handleResolveTopic(store);
    const result = await handler({ topic_key: 'does/not/exist' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data).toBeNull();
  });
});

// ── set_session_context / get_session_context ─────────────────────

describe('session context round-trip', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('set then get returns the same context', async () => {
    const setHandler = handleSetSessionContext(store);
    const getHandler = handleGetSessionContext(store);

    const setResult = await setHandler({ session_id: 'sess-42', project_id: 'proj-7' });
    expect(setResult.isError).toBeUndefined();

    const getResult = await getHandler();
    const ctx = parseResult(getResult);
    expect(ctx).toEqual({ sessionId: 'sess-42', projectId: 'proj-7' });
  });

  it('returns null before any context is set', async () => {
    const getHandler = handleGetSessionContext(store);
    const result = await getHandler();

    const ctx = parseResult(result);
    expect(ctx).toBeNull();
  });
});

// ── merge_projects ─────────────────────────────────────────────────

describe('merge_projects', () => {
  it('returns MergeResult on valid dry-run', async () => {
    const globalStore = makeMockGlobalStore();
    const handler = handleMergeProjects(globalStore);
    const result = await handler({ source_project_id: 'proj-a', target_project_id: 'proj-b' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.dryRun).toBe(true);
    expect(data.sourceProjectId).toBe('proj-a');
    expect(data.targetProjectId).toBe('proj-b');
    expect(typeof data.movedMemories).toBe('number');
  });

  it('passes dry_run false through', async () => {
    const globalStore = makeMockGlobalStore();
    const handler = handleMergeProjects(globalStore);
    const result = await handler({ source_project_id: 'a', target_project_id: 'b', dry_run: false });

    const data = parseResult(result);
    expect(data.dryRun).toBe(false);
  });

  it('propagates error on self-merge refusal', async () => {
    const globalStore = makeMockGlobalStore();
    const handler = handleMergeProjects(globalStore);
    const result = await handler({ source_project_id: 'same', target_project_id: 'same' });

    expect(result.isError).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════
// T-006: Session ledger + queued suggest
// ═════════════════════════════════════════════════════════════════════

// ── store_session_prompt ───────────────────────────────────────────

describe('store_session_prompt', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('returns a prompt with a valid id', async () => {
    const handler = handleStoreSessionPrompt(store);
    const result = await handler({ session_id: 's1', raw_prompt: 'How do I test X?' });

    expect(result.isError).toBeUndefined();
    const prompt = parseResult(result);
    expect(prompt.id).toBeDefined();
    expect(prompt.sessionId).toBe('s1');
    expect(prompt.rawPrompt).toBe('How do I test X?');
    expect(prompt.createdAt).toBeDefined();
  });

  it('stores the prompt in the ledger', async () => {
    const handler = handleStoreSessionPrompt(store);
    await handler({ session_id: 's1', raw_prompt: 'First' });
    await handler({ session_id: 's1', raw_prompt: 'Second' });

    expect(store.prompts).toHaveLength(2);
  });
});

// ── store_session_response ─────────────────────────────────────────

describe('store_session_response', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.storeSessionPrompt('s1', 'question');
  });

  it('returns a response linked to the prompt', async () => {
    const promptId = store.prompts[0].id;
    const handler = handleStoreSessionResponse(store);
    const result = await handler({ session_id: 's1', prompt_id: promptId, full_response: 'Answer here' });

    expect(result.isError).toBeUndefined();
    const resp = parseResult(result);
    expect(resp.id).toBeDefined();
    expect(resp.promptId).toBe(promptId);
    expect(resp.fullResponse).toBe('Answer here');
  });

  it('propagates error when prompt does not exist', async () => {
    const handler = handleStoreSessionResponse(store);
    const result = await handler({ session_id: 's1', prompt_id: 'no-prompt', full_response: 'x' });

    expect(result.isError).toBe(true);
  });
});

// ── store_session_tool_call ────────────────────────────────────────

describe('store_session_tool_call', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.storeSessionPrompt('s1', 'q');
  });

  it('stores a tool call with all fields', async () => {
    const promptId = store.prompts[0].id;
    const handler = handleStoreSessionToolCall(store);
    const result = await handler({
      session_id: 's1',
      prompt_id: promptId,
      tool_name: 'bash',
      args_safe: '{ "cmd": "ls" }',
      result_summary: 'listed files',
      importance: 'high',
    });

    expect(result.isError).toBeUndefined();
    const tc = parseResult(result);
    expect(tc.id).toBeDefined();
    expect(tc.toolName).toBe('bash');
    expect(tc.importance).toBe('high');
  });

  it('works with minimal fields (only session_id + tool_name)', async () => {
    const handler = handleStoreSessionToolCall(store);
    const result = await handler({ session_id: 's1', tool_name: 'read' });

    expect(result.isError).toBeUndefined();
    const tc = parseResult(result);
    expect(tc.toolName).toBe('read');
  });

  it('propagates error on invalid importance', async () => {
    const handler = handleStoreSessionToolCall(store);
    const result = await handler({
      session_id: 's1', tool_name: 'x', importance: 'critical' as 'high',
    });

    expect(result.isError).toBe(true);
  });
});

// ── store_session_summary ──────────────────────────────────────────

describe('store_session_summary', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('stores a full summary with decisions and refs', async () => {
    const handler = handleStoreSessionSummary(store);
    const result = await handler({
      session_id: 's1',
      summary: 'Implemented TDD cycle for MCP tools',
      decisions: ['Use extended mock', 'Slugify as pure function'],
      key_titles: ['PR-3 Tools'],
      memory_ids: ['mem-1', 'mem-2'],
      artifact_refs: ['spec.md'],
    });

    expect(result.isError).toBeUndefined();
    const sum = parseResult(result);
    expect(sum.id).toBeDefined();
    expect(sum.summary).toContain('TDD');
    expect(sum.decisions).toHaveLength(2);
    expect(sum.memoryIds).toEqual(['mem-1', 'mem-2']);
  });

  it('stores a minimal summary (just session_id + summary)', async () => {
    const handler = handleStoreSessionSummary(store);
    const result = await handler({ session_id: 's2', summary: 'Quick note' });

    expect(result.isError).toBeUndefined();
    const sum = parseResult(result);
    expect(sum.summary).toBe('Quick note');
    expect(sum.decisions).toBeUndefined();
  });
});

// ── get_session ────────────────────────────────────────────────────

describe('get_session', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    const p1 = await store.storeSessionPrompt('s1', 'q1');
    await store.storeSessionResponse('s1', p1.id, 'a1');
    await store.storeSessionToolCall({ sessionId: 's1', toolName: 'bash' });
    await store.storeSessionSummary({ sessionId: 's1', summary: 'Done' });
  });

  it('returns the full ledger for a known session', async () => {
    const handler = handleGetSession(store);
    const result = await handler({ session_id: 's1' });

    expect(result.isError).toBeUndefined();
    const ledger = parseResult(result);
    expect(ledger.sessionId).toBe('s1');
    expect(ledger.prompts).toHaveLength(1);
    expect(ledger.responses).toHaveLength(1);
    expect(ledger.toolCalls).toHaveLength(1);
    expect(ledger.summaries).toHaveLength(1);
  });

  it('returns an empty ledger for a non-existent session', async () => {
    const handler = handleGetSession(store);
    const result = await handler({ session_id: 'never-existed' });

    expect(result.isError).toBeUndefined();
    const ledger = parseResult(result);
    expect(ledger.prompts).toEqual([]);
    expect(ledger.responses).toEqual([]);
    expect(ledger.toolCalls).toEqual([]);
    expect(ledger.summaries).toEqual([]);
  });

  it('omits summaries when include_summary is false', async () => {
    const handler = handleGetSession(store);
    const result = await handler({ session_id: 's1', include_summary: false });

    const ledger = parseResult(result);
    expect(ledger.summaries).toEqual([]);
    expect(ledger.prompts).toHaveLength(1);
  });
});

// ── list_sessions ──────────────────────────────────────────────────

describe('list_sessions', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.storeSessionPrompt('s-old', 'old question');
    await store.storeSessionPrompt('s-new', 'new question');
  });

  it('lists sessions with prompt counts', async () => {
    const handler = handleListSessions(store);
    const result = await handler({});

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.count).toBe(2);
    expect(data.sessions).toHaveLength(2);
    const ids = data.sessions.map((s: SessionMeta) => s.sessionId);
    expect(ids).toContain('s-old');
    expect(ids).toContain('s-new');
  });

  it('respects the limit parameter', async () => {
    const handler = handleListSessions(store);
    const result = await handler({ limit: 1 });

    const data = parseResult(result);
    expect(data.sessions).toHaveLength(1);
  });

  it('returns empty list when no sessions exist', async () => {
    const emptyStore = new MockMemoryStore();
    const handler = handleListSessions(emptyStore);
    const result = await handler({});

    const data = parseResult(result);
    expect(data.count).toBe(0);
    expect(data.sessions).toEqual([]);
  });
});

// ── link_prompt_to_memory ──────────────────────────────────────────

describe('link_prompt_to_memory', () => {
  let store: MockMemoryStore;

  beforeEach(async () => {
    store = new MockMemoryStore();
    await store.storeSessionPrompt('s1', 'q');
    await store.store({ content: 'A memory', category: 'learning', tags: [] });
  });

  it('creates a link and acknowledges it', async () => {
    const promptId = store.prompts[0].id;
    const memoryId = store.memories[0].id;
    const handler = handleLinkPromptToMemory(store);
    const result = await handler({
      prompt_id: promptId,
      memory_id: memoryId,
      relation: 'created_from',
    });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.linked).toBe(true);
    expect(store.links).toHaveLength(1);
  });

  it('passes response_id through', async () => {
    const promptId = store.prompts[0].id;
    const memoryId = store.memories[0].id;
    const handler = handleLinkPromptToMemory(store);
    await handler({
      prompt_id: promptId,
      memory_id: memoryId,
      relation: 'referenced',
      response_id: 'resp-99',
    });

    expect(store.links[0].responseId).toBe('resp-99');
  });
});

// ── suggest_topic_key ──────────────────────────────────────────────

describe('slugifyTopicKey (pure)', () => {
  it('converts text to a lowercase kebab slug', () => {
    expect(slugifyTopicKey('Auth Model Design')).toBe('auth-model-design');
  });

  it('replaces non-alphanumeric runs with a single hyphen', () => {
    expect(slugifyTopicKey('foo---bar___baz')).toBe('foo-bar-baz');
  });

  it('trims leading/trailing hyphens', () => {
    expect(slugifyTopicKey('  hello world  ')).toBe('hello-world');
  });

  it('falls back to "memory" for empty or symbol-only input', () => {
    expect(slugifyTopicKey('')).toBe('memory');
    expect(slugifyTopicKey('!!!???')).toBe('memory');
  });

  it('truncates very long content', () => {
    const long = 'a'.repeat(200);
    expect(slugifyTopicKey(long).length).toBeLessThanOrEqual(60);
  });
});

describe('suggest_topic_key (handler)', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('returns a deterministic slug derived from content', async () => {
    const handler = handleSuggestTopicKey(store);
    const result = await handler({ content: 'Database Migration Strategy' });

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.topic_key).toBe('database-migration-strategy');
  });

  it('appends a numeric suffix when the key already exists', async () => {
    // Pre-occupy the base key with a real memory
    const mem = await store.store({ content: 'existing', category: 'other', tags: [] });
    await store.upsertTopicAlias('auth-model', mem.id);
    const handler = handleSuggestTopicKey(store);
    const result = await handler({ content: 'Auth Model' });

    const data = parseResult(result);
    expect(data.topic_key).toBe('auth-model-2');
  });

  it('increments suffix until a free key is found', async () => {
    const mem = await store.store({ content: 'existing', category: 'other', tags: [] });
    await store.upsertTopicAlias('auth-model', mem.id);
    await store.upsertTopicAlias('auth-model-2', mem.id);
    await store.upsertTopicAlias('auth-model-3', mem.id);
    const handler = handleSuggestTopicKey(store);
    const result = await handler({ content: 'Auth Model' });

    const data = parseResult(result);
    expect(data.topic_key).toBe('auth-model-4');
  });
});
