import type {
  CompareResult,
  Embedder,
  EmbeddingIdentity,
  EmbeddingRole,
  JudgeResult,
  Memory,
  MemoryCategory,
  MemoryRelation,
  MemoryStatus,
  MemoryStore,
  MemoryStats,
  PruneOptions,
  PruneResult,
  RelationType,
  SearchFilters,
  SearchMode,
  SearchResult,
  SessionForgetResult,
  SessionItemType,
  SessionLedger,
  SessionMeta,
  SessionPrompt,
  SessionPurgeResult,
  SessionResponse,
  SessionSummary,
  SessionToolCall,
  StoreRequest,
  StoreResult,
  SymbolLinkedResult,
  UpdateRequest,
} from '../src/types.js';
import { VALID_RELATIONS } from '../src/types.js';

// ── MockEmbedder ───────────────────────────────────────────────────
// Returns deterministic pseudo-vectors for testing. No model, no I/O.

export class MockEmbedder implements Embedder {
  readonly callLog: string[] = [];
  /** Role-qualified view of embed() calls (M1: query vs passage). */
  readonly roleLog: Array<{ text: string; role?: EmbeddingRole }> = [];

  constructor(private readonly identityValue?: EmbeddingIdentity) {}

  async initialize(): Promise<void> {}

  async embed(text: string, role?: EmbeddingRole): Promise<number[]> {
    this.callLog.push(text);
    this.roleLog.push({ text, role });
    return this.deterministicVector(text);
  }

  async embedBatch(texts: string[], role?: EmbeddingRole): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t, role)));
  }

  dimensions(): number {
    return 384;
  }

  /** Optional identity; identityless by default (custom-embedder test compat). */
  identity(): EmbeddingIdentity | undefined {
    return this.identityValue;
  }

  private deterministicVector(text: string): number[] {
    // Simple hash-based pseudo-vector for deterministic testing.
    const vector = new Array<number>(384).fill(0);
    for (let i = 0; i < text.length; i++) {
      vector[i % 384] += text.charCodeAt(i) / 1000;
    }
    // Normalise to unit length.
    const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    return magnitude > 0 ? vector.map(v => v / magnitude) : vector;
  }
}

// ── MockMemoryStore ────────────────────────────────────────────────
// In-memory implementation of the FULL MemoryStore interface.
// Uses Maps/arrays for each entity so tests can drive every code path
// (lifecycle, topics, session ledger, auto-link) without spinning up SQLite.
//
// Public field names (`memories`, `prompts`, `topicAliases`, `links`, …)
// are kept stable so existing tests can keep reading them as arrays/maps.

type SessionLink = {
  promptId?: string;
  responseId?: string;
  memoryId: string;
  relation: string;
  createdAt: string;
};

// Status transition table — mirrors SqliteMemoryStore.validateStatusTransition.
const VALID_TRANSITIONS: Record<MemoryStatus, MemoryStatus[]> = {
  active: ['needs_review', 'archived'],
  needs_review: ['active', 'archived'],
  archived: [],
};

// Transitions that REQUIRE a reviewReason.
function transitionRequiresReason(_from: MemoryStatus, to: MemoryStatus): boolean {
  return to === 'archived' || to === 'needs_review';
}

export class MockMemoryStore implements MemoryStore {
  // ── Internal entity storage (Map-backed for stable iteration) ──
  protected memoriesById = new Map<string, Memory>();
  protected topicAliasMap = new Map<string, string>(); // topic_key → memory_id
  protected promptsById = new Map<string, SessionPrompt>();
  protected responsesById = new Map<string, SessionResponse>();
  protected toolCallsById = new Map<string, SessionToolCall>();
  protected summariesById = new Map<string, SessionSummary>();
  protected linkRows: SessionLink[] = [];
  protected relationRows: MemoryRelation[] = [];

  // Insertion order for stable iteration / listRecent ordering.
  protected memoryOrder: string[] = [];

  private sessionCtx: { sessionId: string; projectId: string } | null = null;
  private idCounter = 0;

  // ── Public read-only views (backwards-compat with existing tests) ──

  /** Live array view of stored memories (use .find / .filter / .length). */
  get memories(): Memory[] {
    return this.memoryOrder.map(id => this.memoriesById.get(id)!).filter(Boolean);
  }

  /** Live array view of stored prompts. */
  get prompts(): SessionPrompt[] {
    return [...this.promptsById.values()];
  }

  /** Live array view of stored responses. */
  get responses(): SessionResponse[] {
    return [...this.responsesById.values()];
  }

  /** Live array view of stored tool calls. */
  get toolCalls(): SessionToolCall[] {
    return [...this.toolCallsById.values()];
  }

  /** Live array view of stored summaries. */
  get summaries(): SessionSummary[] {
    return [...this.summariesById.values()];
  }

  /** Live array view of stored prompt↔memory links. */
  get links(): SessionLink[] {
    return [...this.linkRows];
  }

  /** The topic alias map (topic_key → memory_id). */
  get topicAliases(): Map<string, string> {
    return this.topicAliasMap;
  }

  initialize(): Promise<void> {
    return Promise.resolve();
  }

  close(): void {}

  // ── Entity stores (legacy surface) ──

  async store(request: StoreRequest): Promise<Memory | StoreResult> {
    const id = request.id ?? this.nextId('mem');
    const existing = this.memoriesById.get(id);

    let status: MemoryStatus = request.status ?? 'active';
    let createdAt = new Date().toISOString();
    let version = 1;
    let provenance = request.provenance ? { ...request.provenance } : undefined;

    if (existing) {
      // Upsert semantics (matches SqliteMemoryStore.store()):
      // - preserve created_at
      // - merge provenance fields only when provided
      // - keep existing status unless explicitly provided
      createdAt = existing.createdAt;
      version = (existing.version ?? 1) + 1;
      status = request.status ? request.status : (existing.status ?? 'active');
      if (request.provenance) {
        provenance = { ...(existing.provenance ?? {}), ...request.provenance };
      } else {
        provenance = existing.provenance;
      }
    }

    const memory: Memory = {
      id,
      content: request.content,
      category: request.category,
      tags: [...request.tags],
      createdAt,
      updatedAt: new Date().toISOString(),
      version,
      ...(provenance ? { provenance } : {}),
      status,
      ...(existing?.reviewReason ? { reviewReason: existing.reviewReason } : {}),
      ...(existing?.reviewedAt ? { reviewedAt: existing.reviewedAt } : {}),
    };

    // Detect conflicts (opt-in) — simple substring match for mock.
    // W2 fix: detect BEFORE upsert. When candidates exist, DON'T store.
    if (request.detect_conflicts) {
      const threshold = request.conflict_threshold ?? 0.92;
      const candidates = this.memories
        .filter(m => m.id !== id)
        .filter(m => {
          const a = m.content.toLowerCase();
          const b = request.content.toLowerCase();
          // Simple token overlap for mock (real impl uses cosine)
          const tokensA = new Set(a.split(/\s+/).filter(t => t.length > 2));
          const tokensB = new Set(b.split(/\s+/).filter(t => t.length > 2));
          if (tokensA.size === 0 || tokensB.size === 0) return false;
          let overlap = 0;
          for (const t of tokensB) {
            if (tokensA.has(t)) overlap++;
          }
          return (overlap / tokensB.size) >= 0.6;
        })
        .map(m => ({
          id: m.id,
          similarity: 0.95, // mock always returns high similarity for token matches
          content_preview: m.content.slice(0, 200),
          createdAt: m.createdAt,
        }));

      if (candidates.length > 0) {
        return { judgment_required: true, candidates };
      }
    }

    // No conflict (or detect_conflicts not requested) — safe to store
    this.memoriesById.set(id, memory);
    if (!existing) {
      this.memoryOrder.push(id);
    }

    if (request.topicKey) {
      this.topicAliasMap.set(request.topicKey, id);
    }

    if (request.detect_conflicts) {
      return { stored: memory, judgment_required: false };
    }

    return memory;
  }

  async storeBatch(requests: StoreRequest[]): Promise<Memory[]> {
    const results: Memory[] = [];
    for (const r of requests) {
      const result = await this.store(r);
      // store() may return Memory or StoreResult; extract Memory in both cases.
      // When detect_conflicts=true and candidates exist, stored is absent (W2 fix).
      const memory = 'stored' in result ? result.stored : result;
      if (memory) {
        results.push(memory);
      }
    }
    return results;
  }

  async search(
    query: string,
    _mode: SearchMode,
    filters: SearchFilters,
  ): Promise<SearchResult[]> {
    const q = query.toLowerCase();
    let results = this.memories.filter(m =>
      m.content.toLowerCase().includes(q),
    );
    results = this.applyFilters(results, filters);
    // Pinned memories sort first
    results.sort((a, b) => {
      const aPinned = a.isPinned ? 1 : 0;
      const bPinned = b.isPinned ? 1 : 0;
      if (aPinned !== bPinned) return bPinned - aPinned;
      return 0;
    });
    const limit = filters.limit ?? 10;
    return results.slice(0, limit).map(m => ({ memory: m, score: 0.9 }));
  }

  async findRelated(memoryId: string, limit: number): Promise<SearchResult[]> {
    const target = this.memoriesById.get(memoryId);
    if (!target) throw new Error(`Memory ${memoryId} not found`);
    return this.memories
      .filter(m => m.id !== memoryId)
      .slice(0, limit)
      .map(m => ({ memory: m, score: 0.5 }));
  }

  async listRecent(limit: number, category?: MemoryCategory): Promise<Memory[]> {
    // Spec: list_recent MUST exclude archived by default (mirrors SqliteMemoryStore.search()).
    let results = this.memories.filter(m => (m.status ?? 'active') !== 'archived');
    if (category) {
      results = results.filter(m => m.category === category);
    }
    return results
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, limit);
  }

  async update(id: string, updates: UpdateRequest): Promise<Memory> {
    const existing = this.memoriesById.get(id);
    if (!existing) throw new Error(`Memory ${id} not found`);

    const updated: Memory = {
      ...existing,
      content: updates.content ?? existing.content,
      category: updates.category ?? existing.category,
      tags: updates.tags ?? existing.tags,
      status: updates.status ?? existing.status,
      updatedAt: new Date().toISOString(),
      version: (existing.version ?? 1) + 1,
    };
    if (updates.reviewReason !== undefined) {
      updated.reviewReason = updates.reviewReason;
    }
    this.memoriesById.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    if (!this.memoriesById.has(id)) throw new Error(`Memory ${id} not found`);
    this.memoriesById.delete(id);
    this.memoryOrder = this.memoryOrder.filter(x => x !== id);
    // Clean up any topic aliases pointing at this memory.
    for (const [key, memId] of this.topicAliasMap) {
      if (memId === id) this.topicAliasMap.delete(key);
    }
  }

  async stats(): Promise<MemoryStats> {
    const all = this.memories;
    const byCategory: Record<string, number> = {};
    for (const m of all) {
      byCategory[m.category] = (byCategory[m.category] ?? 0) + 1;
    }
    const sorted = [...all].sort((a, b) =>
      a.createdAt.localeCompare(b.createdAt),
    );
    return {
      totalMemories: all.length,
      byCategory,
      oldestMemory: sorted[0]?.createdAt ?? null,
      newestMemory: sorted.at(-1)?.createdAt ?? null,
      neverAccessed: all.length,
      belowPruneThreshold: 0,
      avgAccessCount: 0,
      mostAccessed: [],
    };
  }

  async prune(options: PruneOptions = {}): Promise<PruneResult> {
    const { dryRun = true } = options;
    return {
      pruned: 0,
      inspected: this.memories.length,
      dryRun,
      candidates: [],
    };
  }

  // ── Lifecycle ──────────────────────────────────────────────────

  async updateMemoryStatus(
    id: string,
    status: MemoryStatus,
    reviewReason?: string,
  ): Promise<Memory> {
    const existing = this.memoriesById.get(id);
    if (!existing) throw new Error(`Memory ${id} not found`);

    const from = existing.status ?? 'active';

    if (from === status) {
      // Same-status idempotent update — just refresh reason/reviewedAt.
      const updated: Memory = {
        ...existing,
        status,
        reviewReason,
        reviewedAt: new Date().toISOString(),
        version: (existing.version ?? 1) + 1,
        updatedAt: new Date().toISOString(),
      };
      this.memoriesById.set(id, updated);
      return updated;
    }

    const allowed = VALID_TRANSITIONS[from] ?? [];
    if (!allowed.includes(status)) {
      throw new Error(
        `Invalid status transition: ${from} -> ${status} (archived is terminal)`,
      );
    }

    if (transitionRequiresReason(from, status) && !reviewReason) {
      throw new Error(
        `review_reason required for transition ${from} -> ${status}`,
      );
    }

    const updated: Memory = {
      ...existing,
      status,
      reviewReason,
      reviewedAt: new Date().toISOString(),
      version: (existing.version ?? 1) + 1,
      updatedAt: new Date().toISOString(),
    };
    this.memoriesById.set(id, updated);
    return updated;
  }

  // ── Topic aliases ──────────────────────────────────────────────

  async upsertTopicAlias(topicKey: string, memoryId: string): Promise<void> {
    this.topicAliasMap.set(topicKey, memoryId);
  }

  async resolveTopic(topicKey: string): Promise<Memory | null> {
    const memoryId = this.topicAliasMap.get(topicKey);
    if (!memoryId) return null;
    return this.memoriesById.get(memoryId) ?? null;
  }

  // ── Session ledger ─────────────────────────────────────────────

  async storeSessionPrompt(sessionId: string, rawPrompt: string): Promise<SessionPrompt> {
    const prompt: SessionPrompt = {
      id: this.nextId('prompt'),
      sessionId,
      rawPrompt,
      createdAt: new Date().toISOString(),
    };
    this.promptsById.set(prompt.id, prompt);
    return prompt;
  }

  async storeSessionResponse(
    sessionId: string,
    promptId: string,
    fullResponse: string,
  ): Promise<SessionResponse> {
    if (!this.promptsById.has(promptId)) {
      throw new Error(`Prompt ${promptId} not found`);
    }
    const response: SessionResponse = {
      id: this.nextId('resp'),
      sessionId,
      promptId,
      fullResponse,
      createdAt: new Date().toISOString(),
    };
    this.responsesById.set(response.id, response);
    return response;
  }

  async storeSessionToolCall(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    toolName: string;
    argsSafe?: string;
    resultSummary?: string;
    importance?: string;
  }): Promise<SessionToolCall> {
    if (params.importance && !['low', 'normal', 'high'].includes(params.importance)) {
      throw new Error(
        `Invalid importance "${params.importance}": must be one of low, normal, high`,
      );
    }
    const call: SessionToolCall = {
      id: this.nextId('tc'),
      sessionId: params.sessionId,
      promptId: params.promptId,
      responseId: params.responseId,
      toolName: params.toolName,
      argsSafe: params.argsSafe,
      resultSummary: params.resultSummary,
      importance: params.importance,
      createdAt: new Date().toISOString(),
    };
    this.toolCallsById.set(call.id, call);
    return call;
  }

  async storeSessionSummary(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    summary: string;
    decisions?: string[];
    keyTitles?: string[];
    memoryIds?: string[];
    artifactRefs?: string[];
  }): Promise<SessionSummary> {
    const summary: SessionSummary = {
      id: this.nextId('sum'),
      sessionId: params.sessionId,
      promptId: params.promptId,
      responseId: params.responseId,
      summary: params.summary,
      decisions: params.decisions,
      keyTitles: params.keyTitles,
      memoryIds: params.memoryIds,
      artifactRefs: params.artifactRefs,
      createdAt: new Date().toISOString(),
    };
    this.summariesById.set(summary.id, summary);
    return summary;
  }

  async getSession(
    sessionId: string,
    opts?: { limit?: number; includeSummary?: boolean; includeForgotten?: boolean },
  ): Promise<SessionLedger> {
    const limit = opts?.limit;
    const includeSummary = opts?.includeSummary !== false;
    const includeForgotten = opts?.includeForgotten === true;
    const bySession = <T extends { sessionId: string; status?: string }>(items: T[]): T[] =>
      items.filter(x =>
        x.sessionId === sessionId &&
        (includeForgotten || (x.status ?? 'active') !== 'forgotten'),
      );

    const slice = <T extends { createdAt: string }>(items: T[]): T[] => {
      const sorted = [...items].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
      return typeof limit === 'number' ? sorted.slice(0, limit) : sorted;
    };

    return {
      sessionId,
      prompts: slice(bySession(this.prompts)),
      responses: slice(bySession(this.responses)),
      toolCalls: slice(bySession(this.toolCalls)),
      summaries: includeSummary ? slice(bySession(this.summaries)) : [],
    };
  }

  async listSessions(limit?: number): Promise<SessionMeta[]> {
    const ids = new Set<string>([
      ...this.prompts.map(p => p.sessionId),
      ...this.responses.map(r => r.sessionId),
      ...this.toolCalls.map(t => t.sessionId),
      ...this.summaries.map(s => s.sessionId),
    ]);

    const metas: SessionMeta[] = [];
    for (const sid of ids) {
      const sessionItems = [
        ...this.prompts.filter(p => p.sessionId === sid),
        ...this.responses.filter(r => r.sessionId === sid),
        ...this.toolCalls.filter(t => t.sessionId === sid),
        ...this.summaries.filter(s => s.sessionId === sid),
      ];
      metas.push({
        sessionId: sid,
        promptCount: this.prompts.filter(p => p.sessionId === sid).length,
        lastActivity:
          sessionItems.map(x => x.createdAt).sort().at(-1) ?? new Date().toISOString(),
      });
    }

    metas.sort((a, b) => b.lastActivity.localeCompare(a.lastActivity));
    return typeof limit === 'number' ? metas.slice(0, limit) : metas;
  }

  async linkPromptToMemory(
    promptId: string,
    memoryId: string,
    relation: string,
    responseId?: string,
  ): Promise<void> {
    const dup = this.linkRows.some(l =>
      l.promptId === promptId &&
      l.memoryId === memoryId &&
      l.relation === relation &&
      l.responseId === responseId,
    );
    if (!dup) {
      this.linkRows.push({
        promptId,
        responseId,
        memoryId,
        relation,
        createdAt: new Date().toISOString(),
      });
    }
  }

  // ── Session decay: forget / purge by id ─────────────────────────

  private detectSessionEntity(id: string): SessionItemType {
    const hits: SessionItemType[] = [];
    if (this.promptsById.has(id)) hits.push('prompt');
    if (this.responsesById.has(id)) hits.push('response');
    if (this.toolCallsById.has(id)) hits.push('tool_call');
    if (this.summariesById.has(id)) hits.push('summary');
    if (hits.length === 0) throw new Error(`NOT_FOUND: no session entity with id ${id}`);
    if (hits.length > 1) throw new Error(`AMBIGUOUS: id ${id} matches multiple session tables`);
    return hits[0];
  }

  async forgetSessionItem(id: string): Promise<SessionForgetResult> {
    const type = this.detectSessionEntity(id);
    const store = this.storeFor(type);
    const existing = store.get(id);
    if (existing) {
      store.set(id, { ...existing, status: 'forgotten' });
    }
    return { type, success: true };
  }

  async purgeSessionItem(id: string): Promise<SessionPurgeResult> {
    const type = this.detectSessionEntity(id);
    this.storeFor(type).delete(id);
    return { type, success: true };
  }

  private storeFor(type: SessionItemType): Map<string, any> {
    switch (type) {
      case 'prompt': return this.promptsById as Map<string, any>;
      case 'response': return this.responsesById as Map<string, any>;
      case 'tool_call': return this.toolCallsById as Map<string, any>;
      case 'summary': return this.summariesById as Map<string, any>;
    }
  }

  // ── Session context (in-memory state) ──────────────────────────

  setSessionContext(sessionId: string, projectId: string): void {
    this.sessionCtx = { sessionId, projectId };
  }

  getSessionContext(): { sessionId: string; projectId: string } | null {
    return this.sessionCtx;
  }

  // ── Symbol-linked memory context (v7) ─────────────────────────

  async attachSymbol(opts: { symbolKey: string; memoryId?: string; sessionItemId?: string }): Promise<{ id: string; symbolKey: string }> {
    const { symbolKey, memoryId, sessionItemId } = opts;
    if (memoryId) {
      const mem = this.memoriesById.get(memoryId);
      if (!mem) throw new Error(`Memory ${memoryId} not found`);
      // Update the object in-place (same reference as in the Map).
      (mem as any).symbolKey = symbolKey;
      return { id: memoryId, symbolKey };
    }
    if (sessionItemId) {
      // Search all session stores for the id.
      for (const [id, item] of this.summariesById) {
        if (id === sessionItemId) {
          (item as any).symbolKey = symbolKey;
          return { id: sessionItemId, symbolKey };
        }
      }
      for (const [id, item] of this.promptsById) {
        if (id === sessionItemId) {
          (item as any).symbolKey = symbolKey;
          return { id: sessionItemId, symbolKey };
        }
      }
      for (const [id, item] of this.responsesById) {
        if (id === sessionItemId) {
          (item as any).symbolKey = symbolKey;
          return { id: sessionItemId, symbolKey };
        }
      }
      for (const [id, item] of this.toolCallsById) {
        if (id === sessionItemId) {
          (item as any).symbolKey = symbolKey;
          return { id: sessionItemId, symbolKey };
        }
      }
      throw new Error(`Session item ${sessionItemId} not found`);
    }
    throw new Error('Either memory_id or session_item_id must be provided');
  }

  // ── Pin/unpin (v8) ─────────────────────────────────────────────

  async pin(id: string): Promise<Memory> {
    const existing = this.memoriesById.get(id);
    if (!existing) throw new Error(`Memory ${id} not found`);
    const updated: Memory = { ...existing, isPinned: true, updatedAt: new Date().toISOString() };
    this.memoriesById.set(id, updated);
    return updated;
  }

  async unpin(id: string): Promise<Memory> {
    const existing = this.memoriesById.get(id);
    if (!existing) throw new Error(`Memory ${id} not found`);
    const updated: Memory = { ...existing, isPinned: false, updatedAt: new Date().toISOString() };
    this.memoriesById.set(id, updated);
    return updated;
  }

  async getBySymbol(symbolKey: string, opts?: { sources?: string[]; limit?: number }): Promise<SymbolLinkedResult[]> {
    const sources = opts?.sources ?? ['memory', 'session_summary'];
    const limit = opts?.limit ?? 50;
    const results: SymbolLinkedResult[] = [];

    if (sources.includes('memory')) {
      for (const mem of this.memories) {
        if ((mem as any).symbolKey === symbolKey) {
          results.push({
            id: mem.id,
            source: 'memory',
            symbolKey,
            content: mem.content,
            createdAt: mem.createdAt,
          });
        }
      }
    }

    const sessionSources: Array<[string, Map<string, { sessionId: string; createdAt: string; symbolKey?: string; [k: string]: any }>, string]> = [];
    if (sources.includes('session_summary')) {
      sessionSources.push(['session_summary', this.summariesById as any, 'summary']);
    }
    if (sources.includes('session_prompt')) {
      sessionSources.push(['session_prompt', this.promptsById as any, 'rawPrompt']);
    }
    if (sources.includes('session_response')) {
      sessionSources.push(['session_response', this.responsesById as any, 'fullResponse']);
    }
    if (sources.includes('session_tool_call')) {
      sessionSources.push(['session_tool_call', this.toolCallsById as any, 'resultSummary']);
    }

    for (const [source, store, contentField] of sessionSources) {
      for (const [, item] of store) {
        if ((item as any).symbolKey === symbolKey) {
          results.push({
            id: item.id,
            source: source as SymbolLinkedResult['source'],
            symbolKey,
            content: (item as any)[contentField] ?? '',
            createdAt: item.createdAt,
          });
        }
      }
    }

    return results.slice(0, limit);
  }

  // ── Helpers ────────────────────────────────────────────────────

  /**
   * Deterministic, monotonic id generator. Stable prefixes make test
   * assertions readable (`prompt-1`, `resp-1`, etc.) while guaranteeing
   * uniqueness across entity types.
   */
  private nextId(prefix: string): string {
    this.idCounter += 1;
    return `${prefix}-${this.idCounter}`;
  }

  private applyFilters(memories: Memory[], filters: SearchFilters): Memory[] {
    let results = memories;

    // Status filter — mirrors SqliteMemoryStore behavior:
    //   undefined (default) → exclude 'archived'
    //   'all'                → no filter
    //   specific status      → filter to that status
    if (filters.status === undefined) {
      results = results.filter(m => (m.status ?? 'active') !== 'archived');
    } else if (filters.status !== 'all') {
      results = results.filter(m => (m.status ?? 'active') === filters.status);
    }

    if (filters.category) {
      results = results.filter(m => m.category === filters.category);
    }
    if (filters.tags && filters.tags.length > 0) {
      results = results.filter(m =>
        filters.tags!.some(t => m.tags.includes(t)),
      );
    }
    if (filters.after) {
      results = results.filter(m => m.createdAt >= filters.after!);
    }
    if (filters.before) {
      results = results.filter(m => m.createdAt <= filters.before!);
    }
    return results;
  }

  // ── Memory relations (v9) ──────────────────────────────────────

  addMemoryRelation(opts: {
    memoryIdA: string;
    memoryIdB: string;
    relation: RelationType;
    confidence?: number;
    reasoning?: string;
    markedBy?: string;
  }): MemoryRelation {
    if (!(VALID_RELATIONS as readonly string[]).includes(opts.relation)) {
      throw new Error(
        `Invalid relation '${opts.relation}': must be one of ${VALID_RELATIONS.join(', ')}`,
      );
    }

    const rel: MemoryRelation = {
      id: this.nextId('rel'),
      memoryIdA: opts.memoryIdA,
      memoryIdB: opts.memoryIdB,
      relation: opts.relation,
      createdAt: new Date().toISOString(),
      ...(opts.confidence !== undefined ? { confidence: opts.confidence } : {}),
      ...(opts.reasoning !== undefined ? { reasoning: opts.reasoning } : {}),
      ...(opts.markedBy !== undefined ? { markedBy: opts.markedBy } : {}),
    };

    this.relationRows.push(rel);
    return rel;
  }

  getMemoryRelation(memoryIdA: string, memoryIdB: string): MemoryRelation | null {
    return this.relationRows.find(r => r.memoryIdA === memoryIdA && r.memoryIdB === memoryIdB) ?? null;
  }

  listMemoryRelations(memoryId: string): MemoryRelation[] {
    return this.relationRows.filter(r => r.memoryIdA === memoryId || r.memoryIdB === memoryId);
  }

  // ── compare / judge (v9 T-006) ──────────────────────────────────

  async compare(idA: string, idB: string): Promise<CompareResult> {
    const memA = this.memoriesById.get(idA);
    if (!memA) throw new Error(`Memory ${idA} not found`);
    const memB = this.memoriesById.get(idB);
    if (!memB) throw new Error(`Memory ${idB} not found`);

    // Token-based similarity for mock
    const tokensA = new Set(memA.content.toLowerCase().split(/\s+/).filter(t => t.length > 2));
    const tokensB = new Set(memB.content.toLowerCase().split(/\s+/).filter(t => t.length > 2));
    const allTokens = new Set([...tokensA, ...tokensB]);
    let overlap = 0;
    for (const t of tokensA) {
      if (tokensB.has(t)) overlap++;
    }
    const similarity = allTokens.size > 0 ? overlap / allTokens.size : 0;

    const sharedEntities: string[] = [];
    for (const t of tokensA) {
      if (tokensB.has(t) && t.length > 3) sharedEntities.push(t);
    }

    const lenA = memA.content.length;
    const lenB = memB.content.length;
    const contentDiff = lenA > lenB
      ? `A is ${lenA - lenB} chars longer`
      : lenB > lenA
        ? `B is ${lenB - lenA} chars longer`
        : 'Same length';

    let recommendation: string;
    if (similarity >= 0.95) {
      recommendation = 'supersedes: near-duplicate content';
    } else if (similarity >= 0.85) {
      recommendation = 'conflicts_with: high similarity with differences';
    } else if (similarity >= 0.5) {
      recommendation = 'related: moderate semantic overlap';
    } else {
      recommendation = 'not_conflict: low similarity';
    }

    return { similarity, sharedEntities, contentDiff, recommendation };
  }

  async judge(judgmentId: string, relation: RelationType, reasoning?: string, confidence?: number): Promise<JudgeResult> {
    const colonIdx = judgmentId.indexOf(':');
    if (colonIdx <= 0 || colonIdx === judgmentId.length - 1) {
      throw new Error(
        `Invalid judgment_id '${judgmentId}': must be in format 'newId:existingId'`,
      );
    }

    if (!(VALID_RELATIONS as readonly string[]).includes(relation)) {
      throw new Error(
        `Invalid relation '${relation}': must be one of ${VALID_RELATIONS.join(', ')}`,
      );
    }

    const newId = judgmentId.slice(0, colonIdx);
    const existingId = judgmentId.slice(colonIdx + 1);

    if (!this.memoriesById.has(newId)) throw new Error(`Memory ${newId} not found`);
    if (!this.memoriesById.has(existingId)) throw new Error(`Memory ${existingId} not found`);

    this.addMemoryRelation({
      memoryIdA: newId,
      memoryIdB: existingId,
      relation,
      confidence,
      reasoning,
      markedBy: 'judge',
    });

    let cascaded: 'archived' | undefined;
    if (relation === 'supersedes') {
      await this.updateMemoryStatus(existingId, 'archived', `superseded by ${newId}`);
      cascaded = 'archived';
    }

    return { persisted: true, relation, ...(cascaded ? { cascaded } : {}) };
  }

  // ── Review queue (v9 T-008) ─────────────────────────────────────

  async listReviewQueue(opts?: { projectId?: string; limit?: number }): Promise<{ count: number; items: Memory[] }> {
    const limit = opts?.limit ?? 50;
    let items = this.memories.filter(m => m.status === 'needs_review');
    // Sort by reviewedAt ASC, then createdAt ASC (oldest first)
    items.sort((a, b) => {
      const aTime = a.reviewedAt ?? a.createdAt;
      const bTime = b.reviewedAt ?? b.createdAt;
      return aTime.localeCompare(bTime);
    });
    items = items.slice(0, limit);
    return { count: items.length, items };
  }
}
