// ── Category taxonomy ──────────────────────────────────────────────

export const MEMORY_CATEGORIES = [
  'code-solution',
  'bug-fix',
  'architecture',
  'learning',
  'tool-usage',
  'debugging',
  'performance',
  'security',
  'observation',
  'personal',
  'relationship',
  'other',
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

// ── Domain objects ─────────────────────────────────────────────────

export type MemoryOrigin = 'extracted' | 'inferred' | 'ambiguous' | 'manual' | 'unknown';

// ── Lifecycle ────────────────────────────────────────────────────────

export type MemoryStatus = 'active' | 'needs_review' | 'archived';

export interface MemoryProvenance {
  origin: MemoryOrigin;
  sourceTool?: string;
  sessionId?: string;
  contentHash?: string;
}

export interface Memory {
  id: string;
  content: string;
  category: MemoryCategory;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  version: number;
  provenance?: MemoryProvenance;
  status?: MemoryStatus;
  reviewReason?: string;
  reviewedAt?: string;
  /** v7: soft symbol link (lang|path|kind|scope|name|arity). Survives graph rebuilds. */
  symbolKey?: string;
}

export interface ExplainPayload {
  keyword: { rank: number; bm25: number } | null;
  semantic: { rank: number; cosine: number } | null;
  rrfScore: number;
  decayFactor: number;
  finalScore: number;
}

export interface ExplainSummary {
  keywordCandidates: number;
  semanticCandidates: number;
  fusedCount: number;
  query: { fts: string };
}

export interface SearchResult {
  memory: Memory;
  score: number;
  explain?: ExplainPayload;
}

export interface MemoryStats {
  totalMemories: number;
  byCategory: Record<string, number>;
  oldestMemory: string | null;
  newestMemory: string | null;
  // Access tracking stats
  neverAccessed: number;
  belowPruneThreshold: number;
  avgAccessCount: number;
  mostAccessed: { id: string; content: string; count: number }[];
}

export interface PruneOptions {
  dryRun?: boolean;        // default true
  minStrength?: number;    // default 0.05
  maxDormantDays?: number; // default 90
}

export interface PruneResult {
  pruned: number;
  inspected: number;
  dryRun: boolean;
  candidates: { id: string; content: string; strength: number; reason: string }[];
}

// ── Session ledger ───────────────────────────────────────────────────

/**
 * Lifecycle status for a session-ledger record.
 * Mirrors memory lifecycle semantics so decay/forget reuse the same model.
 */
export type SessionItemStatus = 'active' | 'forgotten';

/**
 * Discriminator for the four session-ledger entity types. Used by
 * forget/purge results and entity-type detection.
 */
export type SessionItemType = 'prompt' | 'response' | 'tool_call' | 'summary';

/** Shared lifecycle fields carried by every session-ledger record. */
export interface SessionItemLifecycle {
  updatedAt?: string;
  accessCount?: number;
  lastAccessedAt?: string;
  status?: SessionItemStatus;
  /** v7: project scope for default-deny session search. Backfilled by setSessionContext. */
  projectId?: string;
  /** v7: soft symbol link (lang|path|kind|scope|name|arity). */
  symbolKey?: string;
}

export interface SessionPrompt extends SessionItemLifecycle {
  id: string;
  sessionId: string;
  rawPrompt: string;
  createdAt: string;
}

export interface SessionResponse extends SessionItemLifecycle {
  id: string;
  sessionId: string;
  promptId: string;
  fullResponse: string;
  createdAt: string;
}

export interface SessionToolCall extends SessionItemLifecycle {
  id: string;
  sessionId: string;
  promptId?: string;
  responseId?: string;
  toolName: string;
  argsSafe?: string;
  resultSummary?: string;
  importance?: string;
  createdAt: string;
}

export interface SessionSummary extends SessionItemLifecycle {
  id: string;
  sessionId: string;
  promptId?: string;
  responseId?: string;
  summary: string;
  decisions?: string[];
  keyTitles?: string[];
  memoryIds?: string[];
  artifactRefs?: string[];
  createdAt: string;
}

/** Result of soft-forgetting a session-ledger record by id. */
export interface SessionForgetResult {
  type: SessionItemType;
  success: boolean;
}

/** Result of hard-purging a session-ledger record (row + embedding) by id. */
export interface SessionPurgeResult {
  type: SessionItemType;
  success: boolean;
}

export interface SessionLedger {
  sessionId: string;
  prompts: SessionPrompt[];
  responses: SessionResponse[];
  toolCalls: SessionToolCall[];
  summaries: SessionSummary[];
}

export interface SessionMeta {
  sessionId: string;
  promptCount: number;
  lastActivity: string;
}

// ── Unified search surfaces (PR 1) ───────────────────────────────────

/**
 * Source discriminator for unified search results. PR 1 implements the
 * session-ledger subset (`session_*` + `memory`); the remaining sources
 * (`global_error`, `promoted_pattern`, `gist`) are reserved for later PRs.
 */
export type UnifiedSearchSource =
  | 'memory'
  | 'session_prompt'
  | 'session_response'
  | 'session_tool_call'
  | 'session_summary'
  | 'global_error'
  | 'promoted_pattern'
  | 'gist';

/** Session-ledger sources searchable in PR 1. */
export const SESSION_SEARCH_SOURCES: readonly UnifiedSearchSource[] = [
  'session_summary',
  'session_prompt',
  'session_response',
  'session_tool_call',
] as const;

/** Compacted (summary-first) sources searched before raw session rows. */
export const COMPACTED_SESSION_SOURCES: readonly UnifiedSearchSource[] = [
  'session_summary',
] as const;

/** Raw session sources surfaced only when `includeRawSessionItems` is true. */
export const RAW_SESSION_SOURCES: readonly UnifiedSearchSource[] = [
  'session_prompt',
  'session_response',
  'session_tool_call',
] as const;

/**
 * Query shape for unified search across memories, sessions, and global stores.
 * Defaults (applied by `parseSearchQuery`): limit=10, status='active',
 * includeContent=false, maxContentChars=4000, includeRawSessionItems=false,
 * sourceBalancing=true.
 */
export interface UnifiedSearchQuery {
  query: string;
  sources?: UnifiedSearchSource[];
  limit?: number;
  status?: string;
  projectId?: string;
  sessionId?: string;
  after?: string;
  before?: string;
  explain?: boolean;
  includeContent?: boolean;
  maxContentChars?: number;
  includeRawSessionItems?: boolean;
  sourceBalancing?: boolean;
  /**
   * v7: when true, legacy session rows with a NULL project_id are included in
   * project-scoped search results alongside the requested project's rows.
   * Defaults to false (default-deny: null project_id rows excluded).
   */
  includeLegacy?: boolean;
}

/**
 * Resolved query after defaults are applied. All optional fields of
 * `UnifiedSearchQuery` are made concrete so downstream code never re-checks
 * defaults.
 */
export interface ResolvedSearchQuery {
  query: string;
  sources: UnifiedSearchSource[];
  limit: number;
  status: string;
  projectId?: string;
  sessionId?: string;
  after?: string;
  before?: string;
  explain: boolean;
  includeContent: boolean;
  maxContentChars: number;
  includeRawSessionItems: boolean;
  sourceBalancing: boolean;
  /** v7: include legacy (null project_id) rows in project-scoped search. */
  includeLegacy: boolean;
}

/**
 * Agent-facing hint for deciding whether to fetch full content. `summary` is a
 * bounded snippet (≤200 chars), `relevanceScore` is normalized 0–1, and
 * `hasTruncatedDetail` flags that more content exists behind `contentRef`.
 */
export interface DecisionHint {
  summary: string;
  relevanceScore: number;
  hasTruncatedDetail: boolean;
}

/** Per-result scoring breakdown, present only when `explain=true`. */
export interface UnifiedSearchExplain {
  keywordRank: number;
  semanticCosine: number;
  sourceMultiplier: number;
  decayFactor: number;
  accessMultiplier: number;
  finalScore: number;
}

/** Uniform result schema across all unified search surfaces. */
export interface UnifiedSearchResult {
  id: string;
  source: UnifiedSearchSource;
  content?: string;
  contentRef: string;
  decisionHint: DecisionHint;
  score: number;
  status: string;
  createdAt: string;
  metadata?: Record<string, unknown>;
  explain?: UnifiedSearchExplain;
}

/** Session-specific result adding session id and compaction flag. */
export interface SessionSearchResult extends UnifiedSearchResult {
  sessionId: string;
  isCompact: boolean;
}

// ── Project merging ──────────────────────────────────────────────────

export interface MergeResult {
  movedMemories: number;
  movedErrors: number;
  dryRun: boolean;
  sourceProjectId: string;
  targetProjectId: string;
}

// ── Request shapes ─────────────────────────────────────────────────

export interface StoreRequest {
  content: string;
  category: MemoryCategory;
  tags: string[];
  provenance?: MemoryProvenance;
  id?: string;         // explicit ID for upsert
  status?: MemoryStatus;
  topicKey?: string;   // auto-create alias after store
}

export interface UpdateRequest {
  content?: string;
  category?: MemoryCategory;
  tags?: string[];
  status?: MemoryStatus;
  reviewReason?: string;
}

export interface SearchFilters {
  category?: MemoryCategory;
  tags?: string[];
  after?: string;
  before?: string;
  limit?: number;
  explain?: boolean;
  status?: MemoryStatus | 'all';  // 'all' = no filter; default excludes 'archived'
}

export type SearchMode = 'hybrid' | 'keyword' | 'semantic';

// ── Dependency interfaces ──────────────────────────────────────────

export interface Embedder {
  initialize(): Promise<void>;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  dimensions(): number;
}

export interface MemoryStore {
  initialize(): Promise<void>;
  store(request: StoreRequest): Promise<Memory>;
  storeBatch(requests: StoreRequest[]): Promise<Memory[]>;
  search(query: string, mode: SearchMode, filters: SearchFilters): Promise<SearchResult[]>;
  findRelated(memoryId: string, limit: number): Promise<SearchResult[]>;
  listRecent(limit: number, category?: MemoryCategory): Promise<Memory[]>;
  update(id: string, updates: UpdateRequest): Promise<Memory>;
  delete(id: string): Promise<void>;
  stats(): Promise<MemoryStats>;
  prune(options: PruneOptions): Promise<PruneResult>;

  // ── Lifecycle ──────────────────────────────────────────────────────
  updateMemoryStatus(id: string, status: MemoryStatus, reviewReason?: string): Promise<Memory>;

  // ── Topic aliases ──────────────────────────────────────────────────
  upsertTopicAlias(topicKey: string, memoryId: string): Promise<void>;
  resolveTopic(topicKey: string): Promise<Memory | null>;

  // ── Session ledger ─────────────────────────────────────────────────
  storeSessionPrompt(sessionId: string, rawPrompt: string): Promise<SessionPrompt>;
  storeSessionResponse(sessionId: string, promptId: string, fullResponse: string): Promise<SessionResponse>;
  storeSessionToolCall(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    toolName: string;
    argsSafe?: string;
    resultSummary?: string;
    importance?: string;
  }): Promise<SessionToolCall>;
  storeSessionSummary(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    summary: string;
    decisions?: string[];
    keyTitles?: string[];
    memoryIds?: string[];
    artifactRefs?: string[];
  }): Promise<SessionSummary>;
  getSession(sessionId: string, opts?: { limit?: number; includeSummary?: boolean; includeForgotten?: boolean }): Promise<SessionLedger>;
  listSessions(limit?: number): Promise<SessionMeta[]>;
  linkPromptToMemory(promptId: string, memoryId: string, relation: string, responseId?: string): Promise<void>;

  // ── Session decay & forget ─────────────────────────────────────────
  forgetSessionItem(id: string): Promise<SessionForgetResult>;
  purgeSessionItem(id: string): Promise<SessionPurgeResult>;

  // ── Session context (in-memory state) ──────────────────────────────
  setSessionContext(sessionId: string, projectId: string): void;
  getSessionContext(): { sessionId: string; projectId: string } | null;

  // ── Symbol-linked memory context (v7) ─────────────────────────────
  /** Attach a symbol key to an existing memory or session item. Upserts. */
  attachSymbol(opts: { symbolKey: string; memoryId?: string; sessionItemId?: string }): Promise<{ id: string; symbolKey: string }>;
  /** Return all memories and session items linked to a symbol key. */
  getBySymbol(symbolKey: string, opts?: { sources?: string[]; limit?: number }): Promise<SymbolLinkedResult[]>;
}

/** Result of looking up records by symbol key. */
export interface SymbolLinkedResult {
  id: string;
  source: 'memory' | 'session_prompt' | 'session_response' | 'session_tool_call' | 'session_summary';
  symbolKey: string;
  content: string;
  createdAt: string;
}

/** Per-symbol enrichment result (raw-canonical + optional structured fields). */
export interface EnrichSymbolResult {
  symbolKey: string;
  found: boolean;
  raw?: string;
  structured?: {
    purpose?: string;
    inputs?: string;
    outputs?: string;
  };
}
