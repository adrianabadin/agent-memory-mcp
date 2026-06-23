import type {
  Embedder,
  Memory,
  MemoryCategory,
  MemoryStatus,
  MemoryStore,
  MemoryStats,
  PruneOptions,
  PruneResult,
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
  UpdateRequest,
} from '../src/types.js';

// ── MockEmbedder ───────────────────────────────────────────────────
// Returns deterministic pseudo-vectors for testing. No model, no I/O.

export class MockEmbedder implements Embedder {
  readonly callLog: string[] = [];

  async initialize(): Promise<void> {}

  async embed(text: string): Promise<number[]> {
    this.callLog.push(text);
    return this.deterministicVector(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }

  dimensions(): number {
    return 384;
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

  async store(request: StoreRequest): Promise<Memory> {
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

    this.memoriesById.set(id, memory);
    if (!existing) {
      this.memoryOrder.push(id);
    }

    if (request.topicKey) {
      this.topicAliasMap.set(request.topicKey, id);
    }

    return memory;
  }

  async storeBatch(requests: StoreRequest[]): Promise<Memory[]> {
    return Promise.all(requests.map(r => this.store(r)));
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
}
