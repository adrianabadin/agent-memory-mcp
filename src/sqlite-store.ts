import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';
import type {
  CandidateMemory,
  ChunkPolicyIdentity,
  CompareResult,
  Embedder,
  EmbeddingIdentity,
  ExplainSummary,
  JudgeResult,
  Memory,
  MemoryCategory,
  MemoryOrigin,
  MemoryRelation,
  MemoryStats,
  MemoryStatus,
  MemoryStore,
  PruneOptions,
  PruneResult,
  RelationType,
  ResolvedSearchQuery,
  SearchFilters,
  SearchMode,
  SearchResult,
  SessionForgetResult,
  SessionItemLifecycle,
  SessionItemStatus,
  SessionItemType,
  SessionLedger,
  SessionMeta,
  SessionPrompt,
  SessionPurgeResult,
  SessionResponse,
  SessionSearchResult,
  SessionSummary,
  SessionToolCall,
  StoreRequest,
  StoreResult,
  UnifiedSearchExplain,
  UnifiedSearchQuery,
  UnifiedSearchSource,
  UpdateRequest,
} from './types.js';
import {
  COMPACTED_SESSION_SOURCES,
  RAW_SESSION_SOURCES,
  SESSION_SEARCH_SOURCES,
  VALID_RELATIONS,
} from './types.js';
import { ConcurrentUpdateError, MemoryNotFoundError } from './errors.js';
import {
  computeDecayFactor,
  parseDecayHalfLife,
  importanceMultiplier,
  EVERGREEN_TAGS,
  transitionDecayState,
  reinforceMemory,
  evaluateDecayBatch,
  evaluateActivationScore,
  DEFAULT_FORGETTING_CONFIG,
} from './decay.js';
import type {
  ForgettingConfig,
  DecayState,
  DecayTransitionResult,
  ReinforcementResult,
  DecayBatchResult,
} from './decay.js';
import {
  embeddingToBlob,
  blobToEmbedding,
  cosineDistance,
  rrf,
  escapeFtsQuery,
} from './shared/vector-utils.js';

/**
 * Ledger/session embedding write helper: an EMPTY vector means "pending" (no
 * model available, e.g. the ledger-only capture-drain path) and is stored as
 * NULL — never as an invalid zero-byte BLOB. Session embedding columns are
 * nullable; readers treat NULL as pending (keyword-only scoring).
 */
export function embeddingToBlobOrNull(vector: number[]): Buffer | null {
  return vector.length > 0 ? embeddingToBlob(vector) : null;
}

// ── Embedding identity guard (M1: E5 small migration) ──────────────

/** Single-row key/value table stamped via {@link writeEmbeddingIdentity}. */
export const EMBEDDING_META_TABLE = 'embedding_meta';

/** Vector length of legacy (pre-identity) databases — the known BGE baseline. */
export const LEGACY_EXPECTED_DIMENSIONS = 1024;

/** Identity metadata persisted in an embedding_meta row. */
export interface StoredEmbeddingIdentity {
  model: string;
  dimensions: number;
  pooling: string;
  normalize: boolean;
  roleAware: boolean;
  chunkPolicyVersion: string | null;
  chunkPolicy: ChunkPolicyIdentity | null;
  stampedAt: string;
}

/** Thrown when the configured embedder is incompatible with the DB identity. */
export class EmbeddingIdentityMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EmbeddingIdentityMismatchError';
  }
}

/**
 * Read the stored embedding identity. READ-ONLY: never creates tables, so it
 * is safe to call before any schema mutation (open-time guard).
 */
export function readEmbeddingIdentity(db: DatabaseSync): StoredEmbeddingIdentity | null {
  const table = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`,
  ).get(EMBEDDING_META_TABLE) as { name: string } | undefined;
  if (!table) return null;
  const row = db.prepare(`SELECT * FROM ${EMBEDDING_META_TABLE} WHERE id = 1`).get() as
    | Record<string, unknown>
    | undefined;
  if (!row) return null;
  let chunkPolicy: ChunkPolicyIdentity | null = null;
  const policyJson = row['chunk_policy_json'];
  if (typeof policyJson === 'string' && policyJson.length > 0) {
    chunkPolicy = JSON.parse(policyJson) as ChunkPolicyIdentity;
  }
  return {
    model: String(row['model']),
    dimensions: Number(row['dimensions']),
    pooling: String(row['pooling']),
    normalize: Boolean(row['normalize']),
    roleAware: Boolean(row['role_aware']),
    chunkPolicyVersion: row['chunk_policy_version'] == null ? null : String(row['chunk_policy_version']),
    chunkPolicy,
    stampedAt: String(row['stamped_at']),
  };
}

/**
 * Stamp the embedding identity into the DB (upsert on id=1). Public API for
 * migration tooling (M2): a re-vectorized sibling DB must be stamped BEFORE
 * being opened with the new embedder, because the open-time guard rejects
 * unlabelled databases for non-legacy (384-dim) embedders.
 */
export function writeEmbeddingIdentity(db: DatabaseSync, identity: EmbeddingIdentity): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${EMBEDDING_META_TABLE} (
      id                  INTEGER PRIMARY KEY CHECK (id = 1),
      model               TEXT NOT NULL,
      dimensions          INTEGER NOT NULL,
      pooling             TEXT NOT NULL,
      normalize           INTEGER NOT NULL,
      role_aware          INTEGER NOT NULL,
      chunk_policy_version TEXT,
      chunk_policy_json   TEXT,
      stamped_at          TEXT NOT NULL
    );
  `);
  db.prepare(`
    INSERT INTO ${EMBEDDING_META_TABLE}
      (id, model, dimensions, pooling, normalize, role_aware, chunk_policy_version, chunk_policy_json, stamped_at)
    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      model = excluded.model,
      dimensions = excluded.dimensions,
      pooling = excluded.pooling,
      normalize = excluded.normalize,
      role_aware = excluded.role_aware,
      chunk_policy_version = excluded.chunk_policy_version,
      chunk_policy_json = excluded.chunk_policy_json,
      stamped_at = excluded.stamped_at
  `).run(
    identity.model,
    identity.dimensions,
    identity.pooling,
    identity.normalize ? 1 : 0,
    identity.roleAware ? 1 : 0,
    identity.chunkPolicy?.version ?? null,
    identity.chunkPolicy ? JSON.stringify(identity.chunkPolicy) : null,
    new Date().toISOString(),
  );
}

// ── Migration state guard (M2: E5 re-vectorization) ────────────

/** Target-only migration metadata table written by the M2 `reembed` tooling. */
export const MIGRATION_STATE_TABLE = 'embedding_migration_state';

/** Thrown when a DB carries a non-COMPLETE M2 re-vectorization stamp. */
export class MigrationStateIncompleteError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MigrationStateIncompleteError';
  }
}

/**
 * Lightweight M2 guard on an OPEN handle: a DB that carries the migration
 * state table must report status COMPLETE in its single row (id=1) —
 * INCOMPLETE, ERROR, any unknown status, or a missing row when the table
 * exists all throw. DBs without the table are not migration targets and
 * pass. Called from `guardEmbeddingIdentity` BEFORE any schema/migration
 * mutation, so an incomplete re-vectorization target can never be opened
 * (or migrated) by the runtime.
 */
export function assertMigrationStateComplete(db: DatabaseSync): void {
  const table = db
    .prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name=?`)
    .get(MIGRATION_STATE_TABLE) as { name: string } | undefined;
  if (!table) return;
  const row = db
    .prepare(`SELECT status FROM ${MIGRATION_STATE_TABLE} WHERE id = 1`)
    .get() as { status: string } | undefined;
  if (!row || row.status !== 'COMPLETE') {
    throw new MigrationStateIncompleteError(
      `Embedding migration INCOMPLETE in this database (status=${row?.status ?? 'missing'}). ` +
      `Finish the re-vectorization (reembed CLI --resume) before opening it with the runtime.`,
    );
  }
}

// ── Schema ─────────────────────────────────────────────────────────

const SCHEMA_SQL = `
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  category TEXT NOT NULL,
  tags TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  origin TEXT NOT NULL DEFAULT 'unknown',
  source_tool TEXT,
  session_id TEXT,
  content_hash TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  id UNINDEXED,
  content,
  tags,
  tokenize='unicode61',
  content='memories',
  content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
  id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
  INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
END;
`;

// ── Row type (internal) ───────────────────────────────────────────

interface MemoryRow {
  id: string;
  content: string;
  category: string;
  tags: string;
  created_at: string;
  updated_at: string;
  access_count: number;
  last_accessed_at: string;
  version: number;
  origin?: string;
  source_tool?: string | null;
  session_id?: string | null;
  content_hash?: string | null;
  status?: string | null;
  review_reason?: string | null;
  reviewed_at?: string | null;
  activation_score?: number;
  memory_state?: string;
  last_reinforced_at?: string | null;
  symbol_key?: string | null;
  is_pinned?: number;
}

// ── Utilities ─────────────────────────────────────────────────────

function rowToMemory(row: MemoryRow): Memory {
  const memory: Memory = {
    id: row.id,
    content: row.content,
    category: row.category as MemoryCategory,
    tags: JSON.parse(row.tags),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    version: row.version ?? 1,
  };
  const origin = (row.origin ?? 'unknown') as MemoryOrigin;
  if (origin !== 'unknown' || row.source_tool || row.session_id || row.content_hash) {
    memory.provenance = {
      origin,
      ...(row.source_tool ? { sourceTool: row.source_tool } : {}),
      ...(row.session_id ? { sessionId: row.session_id } : {}),
      ...(row.content_hash ? { contentHash: row.content_hash } : {}),
    };
  }
  if (row.status && row.status !== 'active') {
    memory.status = row.status as MemoryStatus;
  } else if (row.status === 'active') {
    memory.status = 'active';
  }
  if (row.review_reason) {
    memory.reviewReason = row.review_reason;
  }
  if (row.reviewed_at) {
    memory.reviewedAt = row.reviewed_at;
  }
  if (row.symbol_key) {
    memory.symbolKey = row.symbol_key;
  }
  memory.isPinned = !!row.is_pinned;
  return memory;
}

function isEvergreen(row: MemoryRow): boolean {
  try {
    const tags: string[] = JSON.parse(row.tags);
    return tags.some(t => EVERGREEN_TAGS.has(t));
  } catch {
    return false;
  }
}

const DECAY_HALF_LIFE_DAYS = parseDecayHalfLife(process.env.MEMORY_DECAY_HALF_LIFE);
const MS_PER_DAY = 86_400_000;

/** Decay states that trigger automatic needs_review flagging (T-008). */
const REVIEWABLE_DECAY_STATES = new Set(['dimmed', 'shadow', 'gist_only']);

/**
 * Predicate: should a decay transition auto-flag the memory for review?
 * True when the new state is dimmed/shadow/gist_only AND the transition
 * actually changed state. Pinned and non_forgettable checks are handled
 * upstream (query filter or early return).
 */
function shouldAutoFlagForReview(newState: string, fromState: string): boolean {
  return REVIEWABLE_DECAY_STATES.has(newState) && newState !== fromState;
}

/** Common English stop words excluded from shared entity extraction. */
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
  'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
  'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
  'our', 'their', 'not', 'no', 'nor', 'as', 'if', 'then', 'than', 'too',
  'very', 'just', 'about', 'above', 'after', 'again', 'all', 'also',
  'any', 'because', 'before', 'between', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'into', 'over', 'own', 'same', 'so',
]);

/**
 * Extract shared entities (significant tokens) between two content strings.
 * Returns tokens > 3 chars, not stop words, found in both contents.
 */
function extractSharedEntities(contentA: string, contentB: string): string[] {
  const tokensA = new Set(
    contentA.toLowerCase().split(/[^a-z0-9_]+/).filter(t => t.length > 3 && !STOP_WORDS.has(t)),
  );
  const tokensB = new Set(
    contentB.toLowerCase().split(/[^a-z0-9_]+/).filter(t => t.length > 3 && !STOP_WORDS.has(t)),
  );
  const shared: string[] = [];
  for (const t of tokensA) {
    if (tokensB.has(t)) shared.push(t);
  }
  return shared.slice(0, 10);
}

function applyDecay(row: MemoryRow, rawScore: number): number {
  if (DECAY_HALF_LIFE_DAYS <= 0 || isEvergreen(row)) return rawScore;
  const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
  const temporalFactor = computeDecayFactor(row.updated_at, effectiveHalfLife);
  const activationFactor = row.activation_score ?? 1.0;
  return rawScore * temporalFactor * activationFactor;
}

function applyDecayWithFactor(row: MemoryRow, rawScore: number): { score: number; factor: number } {
  if (DECAY_HALF_LIFE_DAYS <= 0 || isEvergreen(row)) return { score: rawScore, factor: 1 };
  const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
  const temporalFactor = computeDecayFactor(row.updated_at, effectiveHalfLife);
  const activationFactor = row.activation_score ?? 1.0;
  const factor = temporalFactor * activationFactor;
  return { score: rawScore * factor, factor };
}

/**
 * Session decay scoring helper (task 2.5).
 *
 * Reuses the same exponential-decay + access-importance model as memories
 * (computeDecayFactor × importanceMultiplier) so session-ledger records can be
 * ranked with retention semantics comparable to the memory lifecycle.
 *
 * This is a PURE function: given a row-shaped object with `updated_at`,
 * `access_count`, and `last_accessed_at`, it returns a multiplier in (0, 1].
 * Hook point: session search (not yet implemented) will multiply raw relevance
 * scores by this factor before ranking, exactly as `applyDecay` does for
 * memories. `getSession` already excludes `status='forgotten'` rows.
 */
export function scoreSessionRow(row: SessionLifecycleColumns): number {
  const updatedAt = row.updated_at ?? row.last_accessed_at ?? new Date().toISOString();
  if (DECAY_HALF_LIFE_DAYS <= 0) return 1;
  const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier({
    access_count: row.access_count ?? 0,
    updated_at: updatedAt,
    last_accessed_at: row.last_accessed_at ?? updatedAt,
  });
  return computeDecayFactor(updatedAt, effectiveHalfLife);
}

// ── Unified search: parameter parsing (PR 1) ───────────────────────

const VALID_UNIFIED_SOURCES: ReadonlySet<UnifiedSearchSource> = new Set<UnifiedSearchSource>([
  'memory', 'session_prompt', 'session_response', 'session_tool_call',
  'session_summary', 'global_error', 'promoted_pattern', 'gist',
]);

/**
 * Parse and validate a UnifiedSearchQuery, applying documented defaults:
 * limit=10, status='active', includeContent=false, maxContentChars=4000,
 * includeRawSessionItems=false, sourceBalancing=true, explain=false.
 *
 * When `sources` is omitted it defaults to the session-ledger sources. Throws
 * on an empty query, non-positive limit, negative maxContentChars, or an
 * unknown source value.
 */
export function parseSearchQuery(query: UnifiedSearchQuery): ResolvedSearchQuery {
  if (!query.query || !query.query.trim()) {
    throw new Error('INVALID_ARGUMENT: query must be a non-empty string');
  }
  const limit = query.limit ?? 10;
  if (!Number.isFinite(limit) || limit <= 0) {
    throw new Error(`INVALID_ARGUMENT: limit must be a positive number (got ${query.limit})`);
  }
  const maxContentChars = query.maxContentChars ?? 4000;
  if (!Number.isFinite(maxContentChars) || maxContentChars < 0) {
    throw new Error(`INVALID_ARGUMENT: maxContentChars must be >= 0 (got ${query.maxContentChars})`);
  }
  const sources = query.sources ?? [...SESSION_SEARCH_SOURCES];
  for (const s of sources) {
    if (!VALID_UNIFIED_SOURCES.has(s)) {
      throw new Error(`INVALID_ARGUMENT: unknown source '${s}'`);
    }
  }

  const resolved: ResolvedSearchQuery = {
    query: query.query.trim(),
    sources,
    limit,
    status: query.status ?? 'active',
    explain: query.explain ?? false,
    includeContent: query.includeContent ?? false,
    maxContentChars,
    includeRawSessionItems: query.includeRawSessionItems ?? false,
    sourceBalancing: query.sourceBalancing ?? true,
    // v7: projectId now flows through to project-scoped session search
    // (default-deny: null project_id rows excluded unless includeLegacy).
    includeLegacy: query.includeLegacy ?? false,
  };
  if (query.projectId) resolved.projectId = query.projectId;
  if (query.sessionId) resolved.sessionId = query.sessionId;
  if (query.after) resolved.after = query.after;
  if (query.before) resolved.before = query.before;
  return resolved;
}

/** Bounded snippet for DecisionHint summaries. */
const DECISION_HINT_MAX_CHARS = 200;

/**
 * Build a DecisionHint: a bounded snippet plus a normalized relevance score and
 * a flag indicating whether more detail exists behind the contentRef handle.
 */
function generateDecisionHint(
  content: string,
  relevanceScore: number,
  maxContentChars: number,
): { summary: string; relevanceScore: number; hasTruncatedDetail: boolean } {
  const trimmed = content.trim();
  const summary = trimmed.length > DECISION_HINT_MAX_CHARS
    ? `${trimmed.slice(0, DECISION_HINT_MAX_CHARS - 1)}…`
    : trimmed;
  const hasTruncatedDetail =
    trimmed.length > DECISION_HINT_MAX_CHARS || trimmed.length > maxContentChars;
  const clamped = Math.max(0, Math.min(1, relevanceScore));
  return { summary, relevanceScore: clamped, hasTruncatedDetail };
}

// ── Unified session search: internal scoring types ─────────────────

/** Raw row shape returned by the per-source session candidate query. */
interface SessionCandidateRow {
  id: string;
  session_id: string;
  content: string | null;
  created_at: string;
  status: string | null;
  embedding: Buffer | Uint8Array | null;
}

/** A session row after keyword/semantic scoring and RRF fusion. */
interface ScoredSessionCandidate {
  row: SessionCandidateRow;
  source: UnifiedSearchSource;
  content: string;
  keywordScore: number;
  semanticScore: number;
  score: number;
  keywordRank: number;
  semanticRank: number;
}

/** Per-source table/column metadata for the unified session search. */
const SESSION_SOURCE_META: Partial<Record<UnifiedSearchSource, {
  table: string;
  contentCol: string;
  isCompact: boolean;
}>> = {
  session_summary: { table: 'session_summaries', contentCol: 'summary', isCompact: true },
  session_prompt: { table: 'session_prompts', contentCol: 'raw_prompt', isCompact: false },
  session_response: { table: 'session_responses', contentCol: 'full_response', isCompact: false },
  session_tool_call: { table: 'session_tool_calls', contentCol: 'result_summary', isCompact: false },
};

/**
 * Round-robin interleave candidates across sources so no single verbose source
 * monopolizes the result list. Sources are visited in priority order
 * (compacted-first), each contributing its best remaining candidate per round.
 */
function interleaveBalanced(
  sources: UnifiedSearchSource[],
  perSource: Map<UnifiedSearchSource, ScoredSessionCandidate[]>,
): ScoredSessionCandidate[] {
  const cursors = new Map<UnifiedSearchSource, number>(sources.map(s => [s, 0]));
  const out: ScoredSessionCandidate[] = [];
  let remaining = true;
  while (remaining) {
    remaining = false;
    for (const source of sources) {
      const list = perSource.get(source) ?? [];
      const idx = cursors.get(source)!;
      if (idx < list.length) {
        out.push(list[idx]);
        cursors.set(source, idx + 1);
        remaining = true;
      }
    }
  }
  return out;
}

/** Flatten all source candidates and sort purely by fused score (best-first). */
function flattenByScore(
  sources: UnifiedSearchSource[],
  perSource: Map<UnifiedSearchSource, ScoredSessionCandidate[]>,
): ScoredSessionCandidate[] {
  const all: ScoredSessionCandidate[] = [];
  for (const source of sources) {
    all.push(...(perSource.get(source) ?? []));
  }
  return all.sort((a, b) => b.score - a.score);
}

// ── Session ledger row types + converters ──────────────────────────

/**
 * Snake-case lifecycle columns as they live in the DB. Shared by all four
 * session row types and used by the decay/forget helpers so they can read
 * raw rows without mapping to camelCase first.
 */
interface SessionLifecycleColumns {
  updated_at?: string | null;
  access_count?: number | null;
  last_accessed_at?: string | null;
  status?: string | null;
  project_id?: string | null;
  symbol_key?: string | null;
}

interface SessionPromptRow extends SessionLifecycleColumns {
  id: string;
  session_id: string;
  raw_prompt: string;
  created_at: string;
}

interface SessionResponseRow extends SessionLifecycleColumns {
  id: string;
  session_id: string;
  prompt_id: string;
  full_response: string;
  created_at: string;
}

interface SessionToolCallRow extends SessionLifecycleColumns {
  id: string;
  session_id: string;
  prompt_id: string | null;
  response_id: string | null;
  tool_name: string;
  args_safe: string | null;
  result_summary: string | null;
  importance: string | null;
  created_at: string;
}

interface SessionSummaryRow extends SessionLifecycleColumns {
  id: string;
  session_id: string;
  prompt_id: string | null;
  response_id: string | null;
  summary: string;
  decisions: string | null;
  key_titles: string | null;
  memory_ids: string | null;
  artifact_refs: string | null;
  created_at: string;
}

function rowToSessionPrompt(row: SessionPromptRow): SessionPrompt {
  const result: SessionPrompt = {
    id: row.id,
    sessionId: row.session_id,
    rawPrompt: row.raw_prompt,
    createdAt: row.created_at,
  };
  applySessionLifecycle(result, row);
  return result;
}

function rowToSessionResponse(row: SessionResponseRow): SessionResponse {
  const result: SessionResponse = {
    id: row.id,
    sessionId: row.session_id,
    promptId: row.prompt_id,
    fullResponse: row.full_response,
    createdAt: row.created_at,
  };
  applySessionLifecycle(result, row);
  return result;
}

function rowToSessionToolCall(row: SessionToolCallRow): SessionToolCall {
  const result: SessionToolCall = {
    id: row.id,
    sessionId: row.session_id,
    toolName: row.tool_name,
    createdAt: row.created_at,
  };
  if (row.prompt_id) result.promptId = row.prompt_id;
  if (row.response_id) result.responseId = row.response_id;
  if (row.args_safe) result.argsSafe = row.args_safe;
  if (row.result_summary) result.resultSummary = row.result_summary;
  if (row.importance) result.importance = row.importance;
  applySessionLifecycle(result, row);
  return result;
}

function rowToSessionSummary(row: SessionSummaryRow): SessionSummary {
  const result: SessionSummary = {
    id: row.id,
    sessionId: row.session_id,
    summary: row.summary,
    createdAt: row.created_at,
  };
  if (row.prompt_id) result.promptId = row.prompt_id;
  if (row.response_id) result.responseId = row.response_id;
  if (row.decisions) {
    try { result.decisions = JSON.parse(row.decisions); } catch { /* malformed JSON → omit */ }
  }
  if (row.key_titles) {
    try { result.keyTitles = JSON.parse(row.key_titles); } catch { /* omit */ }
  }
  if (row.memory_ids) {
    try { result.memoryIds = JSON.parse(row.memory_ids); } catch { /* omit */ }
  }
  if (row.artifact_refs) {
    try { result.artifactRefs = JSON.parse(row.artifact_refs); } catch { /* omit */ }
  }
  applySessionLifecycle(result, row);
  return result;
}

/**
 * Copy lifecycle columns from a raw DB row into a session item object.
 * Pure helper shared by all four converters. Only sets fields when the
 * column actually carries a value (backward-compatible with pre-v3 rows
 * surfaced before a migration run).
 */
function applySessionLifecycle<T extends SessionItemLifecycle>(target: T, row: SessionLifecycleColumns): void {
  if (row.updated_at != null) target.updatedAt = row.updated_at;
  if (row.access_count != null) target.accessCount = row.access_count;
  if (row.last_accessed_at != null) target.lastAccessedAt = row.last_accessed_at;
  if (row.status) target.status = row.status as SessionItemStatus;
  if (row.project_id) target.projectId = row.project_id;
  if (row.symbol_key) target.symbolKey = row.symbol_key;
}

// ── SqliteMemoryStore ─────────────────────────────────────────────

export class SqliteMemoryStore implements MemoryStore {
  private db!: DatabaseSync;
  // In-memory session context (not persisted; set by callers via MCP ENV).
  private sessionContext: { sessionId: string; projectId: string } | null = null;
  // T-013: sibling stores for multi-DB cross-project search.
  private siblingStores: SqliteMemoryStore[] = [];

  constructor(
    private readonly dbPath: string,
    private readonly embedder: Embedder,
  ) {}

  async initialize(): Promise<void> {
    // Close any existing handle first so re-initialization is safe and
    // does not leak file locks (important on Windows WAL databases).
    this.close();
    this.db = new DatabaseSync(this.dbPath);
    // Open-time embedding-identity guard: MUST run BEFORE any schema
    // mutation (SCHEMA_SQL / migrations) so an incompatible embedder can
    // never write vectors — or migrations — into a mismatched database.
    try {
      this.guardEmbeddingIdentity();
    } catch (err) {
      this.close(); // release the file lock (Windows) before propagating
      throw err;
    }
    this.db.exec(SCHEMA_SQL);
    this.runMigrations();
  }

  /**
   * Reject opening a database whose stored embedding identity does not match
   * the configured embedder, and stamp fresh databases on first open.
   *
   * Rules (M1 E5 small migration):
   *   - stored identity present:
   *       dimensions ALWAYS validated; model/pooling/chunk-policy-version
   *       validated when the embedder exposes identity(). Identityless custom
   *       embedders (test mocks) keep compat — only dims are checked.
   *   - no stored identity + fresh DB (no schema yet): stamp the identity.
   *   - no stored identity + legacy schema: known-compat for 1024-dim legacy
   *     embedders (BGE); a non-legacy (e.g. E5 384) embedder must NOT pretend
   *     to be the legacy model — throw and require an explicit stamp (M2).
   */
  private guardEmbeddingIdentity(): void {
    // M2: reject an INCOMPLETE re-vectorization target BEFORE any schema or
    // migration writes (initialize() runs this guard first).
    assertMigrationStateComplete(this.db);
    const stored = readEmbeddingIdentity(this.db);
    const identity = this.embedder.identity?.() ?? null;
    const dims = this.embedder.dimensions();
    // Expected vector length for the histogram audit: prefer the DB stamp,
    // then the embedder's identity, then its declared dimensions.
    const expectedDims = stored?.dimensions ?? identity?.dimensions ?? dims;

    if (stored) {
      if (dims !== stored.dimensions) {
        throw new EmbeddingIdentityMismatchError(
          `Embedding identity mismatch: DB '${this.dbPath}' is stamped for ${stored.model} ` +
          `(${stored.dimensions} dims) but the configured embedder reports ${dims} dims.`,
        );
      }
      if (identity) {
        if (identity.model !== stored.model) {
          throw new EmbeddingIdentityMismatchError(
            `Embedding identity mismatch: DB '${this.dbPath}' is stamped for model '${stored.model}' ` +
            `but the configured embedder is '${identity.model}'.`,
          );
        }
        if (identity.pooling !== stored.pooling) {
          throw new EmbeddingIdentityMismatchError(
            `Embedding identity mismatch: DB '${this.dbPath}' pooling '${stored.pooling}' ` +
            `vs embedder pooling '${identity.pooling}'.`,
          );
        }
        const identityPolicy = identity.chunkPolicy?.version ?? null;
        if (identityPolicy !== stored.chunkPolicyVersion) {
          throw new EmbeddingIdentityMismatchError(
            `Embedding identity mismatch: DB '${this.dbPath}' chunk policy '${stored.chunkPolicyVersion ?? 'none'}' ` +
            `vs embedder chunk policy '${identityPolicy ?? 'none'}'.`,
          );
        }
      }
      // The stamp does NOT override reality: audit the actual stored vectors.
      // Stamped (migrated) DBs are STRICT: no legacy placeholder tolerance.
      this.auditEmbeddingHistograms(expectedDims, true);
      return;
    }

    // No stored identity.
    const schemaPresent = this.tableExists('memories');
    if (!schemaPresent) {
      // Fresh database: safe to stamp the configured identity on first open.
      if (identity) writeEmbeddingIdentity(this.db, identity);
      return;
    }
    // Legacy (unlabelled) database with pre-existing schema: audit the actual
    // stored vector lengths against the target embedder dims before opening.
    // Unlabelled legacy DBs tolerate historical session placeholder junk
    // (0-byte / non-float32) as pending — stamped/migrated DBs never do.
    this.auditEmbeddingHistograms(expectedDims, false);
    if (!identity) return; // identityless custom embedder — compat
    if (identity.dimensions === LEGACY_EXPECTED_DIMENSIONS) return; // known legacy BGE compat
    throw new EmbeddingIdentityMismatchError(
      `Refusing to open unlabelled legacy DB '${this.dbPath}' with non-legacy embedder ` +
      `${identity.model} (${identity.dimensions} dims). Known legacy databases carry ` +
      `${LEGACY_EXPECTED_DIMENSIONS}-dim vectors; stamp the intended identity explicitly ` +
      `(writeEmbeddingIdentity) before re-vectorizing.`,
    );
  }

  /**
   * Cheap open-time audit of ALL existing non-null embedding columns via SQL
   * typeof/length counts (no corpus load, no vector math — finiteness/norms
   * are validated during M2 migration, not at open):
   *   - NULL                       → pending (allowed, e.g. ledger-only writes);
   *   - non-BLOB non-NULL          → rejected;
   *   - BLOB with length ≠ dims*4  → rejected (dim mismatch), even when the
   *     stored metadata stamp agrees with the embedder.
   *
   * `strict=false` (UNLABELLED legacy DBs ONLY): historical session placeholder
   * junk (zero-byte or not a multiple of 4, hence not a float32 vector) is
   * tolerated as pending; new writes never produce it (embeddingToBlobOrNull
   * emits only NULL or dims*4 blobs) and M2 normalizes those rows. Any
   * E5-stamped/migrated DB (`strict=true`) rejects every non-NULL invalid or
   * non-dims*4 blob, including 0-byte and 3-byte ones; NULL pending remains
   * allowed everywhere.
   */
  private auditEmbeddingHistograms(expectedDims: number, strict: boolean): void {
    const expectedBytes = expectedDims * 4;
    const targets: Array<{ table: string; allowLegacyEmpty: boolean }> = [
      { table: 'memory_embeddings', allowLegacyEmpty: false },
      { table: 'session_prompts', allowLegacyEmpty: true },
      { table: 'session_responses', allowLegacyEmpty: true },
      { table: 'session_tool_calls', allowLegacyEmpty: true },
      { table: 'session_summaries', allowLegacyEmpty: true },
    ];
    for (const t of targets) {
      if (!this.tableExists(t.table)) continue;
      const rows = this.db.prepare(
        `SELECT typeof(embedding) AS etype, length(embedding) AS elen, COUNT(*) AS n
         FROM ${t.table} GROUP BY etype, elen`,
      ).all() as unknown as Array<{ etype: string; elen: number | null; n: number }>;
      for (const r of rows) {
        if (r.etype === 'null') continue; // pending vectors (ledger-only path)
        if (r.etype !== 'blob') {
          throw new EmbeddingIdentityMismatchError(
            `Embedding audit failed: ${this.dbPath}.${t.table}.embedding has ${r.n} ` +
            `non-blob value(s) (typeof=${r.etype}).`,
          );
        }
        const bytes = r.elen ?? 0;
        if (bytes % 4 !== 0 || bytes === 0) {
          // Not a float32 vector of the expected length.
          if (!strict && t.allowLegacyEmpty) continue; // unlabelled-legacy pending junk
          if (!strict) {
            throw new EmbeddingIdentityMismatchError(
              `Embedding audit failed: ${this.dbPath}.${t.table}.embedding has ${r.n} corrupt blob(s) ` +
              `(${bytes} bytes, not a multiple of 4 or zero-byte).`,
            );
          }
          throw new EmbeddingIdentityMismatchError(
            `Embedding audit failed: ${this.dbPath}.${t.table}.embedding has ${r.n} invalid blob(s) ` +
            `(${bytes} bytes; stamped DBs require exactly ${expectedBytes} bytes, NULL = pending).`,
          );
        }
        if (bytes !== expectedBytes) {
          throw new EmbeddingIdentityMismatchError(
            `Embedding audit failed: ${this.dbPath}.${t.table}.embedding has ${r.n} blob(s) of ` +
            `${bytes} bytes, expected ${expectedBytes} (${expectedDims} dims × 4).`,
          );
        }
      }
    }
  }

  private tableExists(name: string): boolean {
    return Boolean(
      this.db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`).get(name),
    );
  }

  /**
   * Idempotent schema migrations. Bump PRAGMA user_version after each successful
   * migration so re-initialization is a no-op.
   *
   * v0 → v1: add provenance columns to `memories`.
   * v1 → v2: add memory lifecycle (status/review columns), session ledger tables,
   *          topic aliases, and supporting indexes.
   */
  private runMigrations(): void {
    const current = (this.db.prepare('PRAGMA user_version').get() as { user_version: number } | undefined)?.user_version ?? 0;
    if (current < 1) {
      this.addColumnIfMissing('memories', 'origin', "TEXT NOT NULL DEFAULT 'unknown'");
      this.addColumnIfMissing('memories', 'source_tool', 'TEXT');
      this.addColumnIfMissing('memories', 'session_id', 'TEXT');
      this.addColumnIfMissing('memories', 'content_hash', 'TEXT');
      this.db.exec('PRAGMA user_version = 1');
    }
    if (current < 2) {
      // ── Memory lifecycle columns ──
      this.addColumnIfMissing('memories', 'status', "TEXT NOT NULL DEFAULT 'active'");
      this.addColumnIfMissing('memories', 'review_reason', 'TEXT');
      this.addColumnIfMissing('memories', 'reviewed_at', 'TEXT');

      // ── Session ledger tables ──
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS session_prompts (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          raw_prompt TEXT NOT NULL,
          created_at TEXT NOT NULL,
          embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS session_responses (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          prompt_id TEXT NOT NULL REFERENCES session_prompts(id),
          full_response TEXT NOT NULL,
          created_at TEXT NOT NULL,
          embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS session_tool_calls (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          prompt_id TEXT REFERENCES session_prompts(id),
          response_id TEXT REFERENCES session_responses(id),
          tool_name TEXT NOT NULL,
          args_safe TEXT,
          result_summary TEXT,
          importance TEXT,
          created_at TEXT NOT NULL,
          embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS session_summaries (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          prompt_id TEXT REFERENCES session_prompts(id),
          response_id TEXT REFERENCES session_responses(id),
          summary TEXT NOT NULL,
          decisions TEXT,
          key_titles TEXT,
          memory_ids TEXT,
          artifact_refs TEXT,
          created_at TEXT NOT NULL,
          embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS session_memory_links (
          prompt_id TEXT REFERENCES session_prompts(id),
          response_id TEXT REFERENCES session_responses(id),
          memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
          relation TEXT NOT NULL,
          created_at TEXT NOT NULL,
          PRIMARY KEY (prompt_id, response_id, memory_id, relation)
        );

        CREATE TABLE IF NOT EXISTS memory_topics (
          topic_key TEXT NOT NULL,
          memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (topic_key, memory_id)
        );

        CREATE INDEX IF NOT EXISTS idx_session_prompts_session_id ON session_prompts(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_responses_session_id ON session_responses(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_responses_prompt_id ON session_responses(prompt_id);
        CREATE INDEX IF NOT EXISTS idx_session_tool_calls_session_id ON session_tool_calls(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_summaries_session_id ON session_summaries(session_id);
        CREATE INDEX IF NOT EXISTS idx_memory_topics_memory_id ON memory_topics(memory_id);
      `);

      this.db.exec('PRAGMA user_version = 2');
    }
    if (current < 3) {
      // ── Session ledger lifecycle (decay) columns ──
      // Brings session_prompts / session_responses / session_tool_calls /
      // session_summaries into the same retention/decay model as memories.
      // Defaults are backfilled from existing created_at so legacy rows are
      // "fresh" rather than NULL/empty after migration.
      const sessionTables = [
        'session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries',
      ];
      for (const tbl of sessionTables) {
        this.addColumnIfMissing(tbl, 'updated_at', 'TEXT');
        this.addColumnIfMissing(tbl, 'access_count', 'INTEGER NOT NULL DEFAULT 0');
        this.addColumnIfMissing(tbl, 'last_accessed_at', 'TEXT');
        this.addColumnIfMissing(tbl, 'status', "TEXT NOT NULL DEFAULT 'active'");
        // Backfill lifecycle defaults for pre-existing rows.
        this.db.exec(`UPDATE ${tbl} SET updated_at = created_at WHERE updated_at IS NULL`);
        this.db.exec(`UPDATE ${tbl} SET last_accessed_at = created_at WHERE last_accessed_at IS NULL`);
      }

      this.db.exec('PRAGMA user_version = 3');
    }
    if (current < 4) {
      // ── Gradual forgetting columns on memories ──
      this.addColumnIfMissing('memories', 'activation_score', 'REAL DEFAULT 1.0');
      this.addColumnIfMissing('memories', 'memory_state', "TEXT NOT NULL DEFAULT 'active'");
      this.addColumnIfMissing('memories', 'last_reinforced_at', 'TEXT');

      // ── memory_gists table ──
      // original_id is intentionally NOT a FK so the gist survives original purge.
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS memory_gists (
          id TEXT PRIMARY KEY,
          original_id TEXT NOT NULL,
          original_source TEXT NOT NULL,
          kind TEXT NOT NULL,
          core_fact TEXT NOT NULL,
          why_it_matters TEXT NOT NULL,
          revive_triggers TEXT NOT NULL,
          non_forgettable INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          original_pruned_at TEXT,
          status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE INDEX IF NOT EXISTS idx_memory_gists_original_id ON memory_gists(original_id);
        CREATE INDEX IF NOT EXISTS idx_memory_gists_status ON memory_gists(status);
        CREATE INDEX IF NOT EXISTS idx_memory_gists_kind ON memory_gists(kind);
      `);

      // ── memory_decay_log table ──
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS memory_decay_log (
          id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          from_state TEXT NOT NULL,
          to_state TEXT NOT NULL,
          reason TEXT NOT NULL,
          score_before REAL NOT NULL,
          score_after REAL NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_decay_log_memory_id ON memory_decay_log(memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_decay_log_created_at ON memory_decay_log(created_at);
      `);

      this.db.exec('PRAGMA user_version = 4');
    }

    // ── v5: Global promotion & audit trail ──
    if (current < 5) {
      this.addColumnIfMissing('memories', 'non_forgettable', 'INTEGER NOT NULL DEFAULT 0');

      // memory_revive_reviews: tracks revive decisions for revival-based promotion
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS memory_revive_reviews (
          id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          decision TEXT NOT NULL,
          reason TEXT,
          model TEXT,
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_revive_reviews_memory_id ON memory_revive_reviews(memory_id);
        CREATE INDEX IF NOT EXISTS idx_revive_reviews_decision ON memory_revive_reviews(decision);
      `);

      // global_promotion_log: immutable audit trail for promotions
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS global_promotion_log (
          id TEXT PRIMARY KEY,
          promoted_memory_id TEXT NOT NULL,
          source_ids TEXT NOT NULL,
          source_projects TEXT NOT NULL,
          rule_matched TEXT NOT NULL,
          confidence REAL NOT NULL,
          model TEXT,
          reviewer_provenance TEXT,
          created_at TEXT NOT NULL,
          UNIQUE(promoted_memory_id, source_ids)
        );
        CREATE INDEX IF NOT EXISTS idx_gpl_promoted_memory_id ON global_promotion_log(promoted_memory_id);
        CREATE INDEX IF NOT EXISTS idx_gpl_source_projects ON global_promotion_log(source_projects);
        CREATE INDEX IF NOT EXISTS idx_gpl_rule_matched ON global_promotion_log(rule_matched);
      `);

      this.db.exec('PRAGMA user_version = 5');
    }

    // ── v6: Revive review enrichment (gist_id + confidence) ──
    if (current < 6) {
      this.addColumnIfMissing('memory_revive_reviews', 'gist_id', 'TEXT');
      this.addColumnIfMissing('memory_revive_reviews', 'confidence', 'REAL');
      this.db.exec('PRAGMA user_version = 6');
    }

    // ── v7: Symbol-linked memory context (soft symbol_key + project_id) ──
    // Additive, nullable columns so existing rows are unaffected. symbol_key is
    // a soft FK (lang|path|kind|scope|name|arity) that survives graph rebuilds.
    // project_id scopes session-ledger rows to a project for default-deny
    // session search. No hard FKs to graph node IDs by design (Non-Goals).
    if (current < 7) {
      // memories: soft symbol link only (project scope is via DB isolation).
      this.addColumnIfMissing('memories', 'symbol_key', 'TEXT');
      this.db.exec(
        'CREATE INDEX IF NOT EXISTS idx_memories_symbol_key ON memories(symbol_key)',
      );

      // All 4 session-ledger tables: project_id + symbol_key + supporting indexes.
      const sessionTables = [
        'session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries',
      ];
      for (const tbl of sessionTables) {
        this.addColumnIfMissing(tbl, 'project_id', 'TEXT');
        this.addColumnIfMissing(tbl, 'symbol_key', 'TEXT');
        this.db.exec(
          `CREATE INDEX IF NOT EXISTS idx_${tbl}_project_id ON ${tbl}(project_id)`,
        );
        this.db.exec(
          `CREATE INDEX IF NOT EXISTS idx_${tbl}_symbol_key ON ${tbl}(symbol_key)`,
        );
      }

      this.db.exec('PRAGMA user_version = 7');
    }

    // ── v8: Pin support (is_pinned column + index) ──
    if (current < 8) {
      this.addColumnIfMissing('memories', 'is_pinned', 'INTEGER NOT NULL DEFAULT 0');
      this.db.exec(
        'CREATE INDEX IF NOT EXISTS idx_memories_is_pinned ON memories(is_pinned)',
      );
      this.db.exec('PRAGMA user_version = 8');
    }

    // ── v9: Conflict resolution — memory_relations table ──
    if (current < 9) {
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS memory_relations (
          id TEXT PRIMARY KEY,
          memory_id_a TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
          memory_id_b TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
          relation TEXT NOT NULL CHECK (relation IN ('related','compatible','scoped','conflicts_with','supersedes','not_conflict')),
          confidence REAL,
          reasoning TEXT,
          marked_by TEXT,
          created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_relations_unique ON memory_relations(memory_id_a, memory_id_b, relation);
        CREATE INDEX IF NOT EXISTS idx_memory_relations_a ON memory_relations(memory_id_a);
        CREATE INDEX IF NOT EXISTS idx_memory_relations_b ON memory_relations(memory_id_b);
      `);
      this.db.exec('PRAGMA user_version = 9');
    }

    // ── R3 conflict-flow fix — durable pending drafts (no user_version bump: the v9 contract is pinned by existing suites).
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS conflict_drafts (
        id TEXT PRIMARY KEY, request TEXT NOT NULL, embedding BLOB NOT NULL, created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS conflict_draft_judgments (
        id TEXT PRIMARY KEY, draft_id TEXT NOT NULL REFERENCES conflict_drafts(id) ON DELETE CASCADE,
        candidate_id TEXT NOT NULL, resolution TEXT, confidence REAL, reasoning TEXT, created_at TEXT NOT NULL, resolved_at TEXT
      );
      CREATE UNIQUE INDEX IF NOT EXISTS idx_draft_judgments_pair ON conflict_draft_judgments(draft_id, candidate_id);
    `);
  }

  /**
   * Expose the raw database handle for modules that need direct access
   * (e.g., PromotionRuleEngine).
   */
  getDb(): DatabaseSync {
    return this.db;
  }

  private addColumnIfMissing(table: string, column: string, ddl: string): void {
    const cols = this.db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
    if (cols.some((c) => c.name === column)) return;
    this.db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${ddl}`);
  }

  close(): void {
    try {
      this.db?.close();
    } catch {
      // Already closed or never opened — ignore.
    }
  }

  // ── T-013: Multi-DB cross-project search ───────────────────────

  /**
   * Set sibling stores for cross-project search. When `all_projects=true`
   * is passed to search(), results are merged from this store AND all siblings.
   */
  setSiblingStores(stores: SqliteMemoryStore[]): void {
    this.siblingStores = stores;
  }

  /**
   * Search across this store and all sibling stores, merge by score, apply limit.
   * Called when `all_projects=true` and siblings exist.
   */
  private async searchAllProjects(
    query: string,
    mode: SearchMode,
    filters: SearchFilters,
  ): Promise<SearchResult[]> {
    const limit = filters.limit ?? 10;
    // Disable all_projects on individual searches to prevent recursion
    const localFilters = { ...filters, all_projects: false };

    // Search this store
    const localResults = await this.search(query, mode, localFilters);

    // Search sibling stores (each handles its own access tracking)
    const siblingResults: SearchResult[][] = await Promise.all(
      this.siblingStores.map(s => s.search(query, mode, localFilters)),
    );

    // Merge all results
    const allResults: SearchResult[] = [...localResults];
    for (const sr of siblingResults) {
      allResults.push(...sr);
    }

    // Sort by score descending, apply limit
    allResults.sort((a, b) => b.score - a.score);
    return allResults.slice(0, limit);
  }

  // ── Decay state transitions ───────────────────────────────────────

  /**
   * Evaluate a single memory's decay transition, persist the result, and
   * return the transition outcome. Reads the current `memory_state` and
   * `last_reinforced_at` (falling back to `created_at` when NULL), computes
   * the target stage via `transitionDecayState`, logs the transition to
   * `memory_decay_log`, and updates `memory_state` / `activation_score`.
   */
  transitionDecayStateForMemory(
    memoryId: string,
    config?: ForgettingConfig,
  ): DecayTransitionResult {
    const row = this.db.prepare(
      'SELECT memory_state, last_reinforced_at, created_at, is_pinned, non_forgettable FROM memories WHERE id = ?',
    ).get(memoryId) as { memory_state: string; last_reinforced_at: string | null; created_at: string; is_pinned: number; non_forgettable: number } | undefined;

    if (!row) throw new MemoryNotFoundError(memoryId);

    // Skip pinned memories — they are exempt from decay
    if (row.is_pinned) {
      return {
        newState: (row.memory_state ?? 'active') as DecayState,
        newScore: evaluateActivationScore((row.memory_state ?? 'active') as DecayState),
        reason: 'pinned memory is exempt from decay',
        transitioned: false,
      };
    }

    // Skip non_forgettable memories — they are exempt from decay (T-008)
    if (row.non_forgettable) {
      return {
        newState: (row.memory_state ?? 'active') as DecayState,
        newScore: evaluateActivationScore((row.memory_state ?? 'active') as DecayState),
        reason: 'non_forgettable memory is exempt from decay',
        transitioned: false,
      };
    }

    const currentState = (row.memory_state ?? 'active') as DecayState;
    const lastReinforcedAt = row.last_reinforced_at ?? row.created_at;

    const result = transitionDecayState(
      {
        memoryId,
        currentState,
        lastReinforcedAt,
        config: config ?? DEFAULT_FORGETTING_CONFIG,
      },
      (record) => {
        this.db.prepare(
          `INSERT INTO memory_decay_log (id, memory_id, from_state, to_state, reason, score_before, score_after, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          randomUUID(), record.memory_id, record.from_state, record.to_state,
          record.reason, record.score_before, record.score_after, new Date().toISOString(),
        );
      },
    );

    if (result.transitioned) {
      this.db.prepare(
        'UPDATE memories SET memory_state = ?, activation_score = ? WHERE id = ?',
      ).run(result.newState, result.newScore, memoryId);

      // T-008: Auto-flag for review when transitioning to dimmed/shadow/gist_only
      if (shouldAutoFlagForReview(result.newState, currentState)) {
        this.db.prepare(
          `UPDATE memories SET status = 'needs_review', review_reason = ?, reviewed_at = ? WHERE id = ?`,
        ).run(`decay: ${result.newState}`, new Date().toISOString(), memoryId);
      }
    }

    return result;
  }

  /**
   * Reinforce a memory on access: resets `last_reinforced_at`, reverts
   * cooling/dimmed to active, boosts score for deep stages (shadow/gist_only).
   * Logs reinforcement only when state/score actually changes (silent for
   * active accesses).
   */
  reinforceMemory(memoryId: string): ReinforcementResult {
    const row = this.db.prepare(
      'SELECT memory_state, activation_score, last_reinforced_at, created_at FROM memories WHERE id = ?',
    ).get(memoryId) as {
      memory_state: string;
      activation_score: number;
      last_reinforced_at: string | null;
      created_at: string;
    } | undefined;

    if (!row) throw new MemoryNotFoundError(memoryId);

    const currentState = (row.memory_state ?? 'active') as DecayState;
    const activationScore = row.activation_score ?? 1.0;

    const result = reinforceMemory(
      {
        memoryId,
        currentState,
        activationScore,
        config: DEFAULT_FORGETTING_CONFIG,
      },
      (record) => {
        this.db.prepare(
          `INSERT INTO memory_decay_log (id, memory_id, from_state, to_state, reason, score_before, score_after, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          randomUUID(), record.memory_id, record.from_state, record.to_state,
          record.reason, record.score_before, record.score_after, new Date().toISOString(),
        );
      },
    );

    // Always reset last_reinforced_at on reinforcement (even for silent active accesses).
    const now = new Date().toISOString();
    if (result.transitioned) {
      // Full revert: update state, score, and timer.
      this.db.prepare(
        'UPDATE memories SET memory_state = ?, activation_score = ?, last_reinforced_at = ? WHERE id = ?',
      ).run(result.newState, result.newScore, now, memoryId);
    } else if (!result.silent) {
      // Score boost only (shadow/gist_only): update score and timer, keep state.
      this.db.prepare(
        'UPDATE memories SET activation_score = ?, last_reinforced_at = ? WHERE id = ?',
      ).run(result.newScore, now, memoryId);
    } else {
      // Silent active access: reset timer only.
      this.db.prepare(
        'UPDATE memories SET last_reinforced_at = ? WHERE id = ?',
      ).run(now, memoryId);
    }

    return result;
  }

  /**
   * Batch decay evaluation for scheduled maintenance (sleep mode).
   * Fetches all non-purged memories, evaluates each for decay transitions,
   * and persists all transitions in one atomic transaction with ROLLBACK on error.
   */
  evaluateAllMemoriesForDecay(config?: ForgettingConfig): DecayBatchResult {
    const cfg = config ?? DEFAULT_FORGETTING_CONFIG;

    const rows = this.db.prepare(
      `SELECT id, memory_state, last_reinforced_at, created_at, activation_score
       FROM memories WHERE memory_state != 'purged' AND (is_pinned = 0 OR is_pinned IS NULL) AND (non_forgettable = 0 OR non_forgettable IS NULL)`,
    ).all() as Array<{
      id: string;
      memory_state: string;
      last_reinforced_at: string | null;
      created_at: string;
      activation_score: number;
    }>;

    const batchInput = rows.map(r => ({
      memoryId: r.id,
      currentState: (r.memory_state ?? 'active') as DecayState,
      lastReinforcedAt: r.last_reinforced_at ?? r.created_at,
      activationScore: r.activation_score ?? 1.0,
    }));

    const batchResult = evaluateDecayBatch(batchInput, cfg);

    // Persist all transitions in one atomic transaction.
    // Also create gists for gist_only transitions and prune purged originals.
    this.db.exec('BEGIN');
    try {
      for (const { memoryId, result } of batchResult.results) {
        if (!result.transitioned) continue;
        const fromState = batchInput.find(m => m.memoryId === memoryId)!.currentState;
        // Log the transition
        this.db.prepare(
          `INSERT INTO memory_decay_log (id, memory_id, from_state, to_state, reason, score_before, score_after, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          randomUUID(), memoryId,
          fromState, result.newState, result.reason,
          evaluateActivationScore(fromState),
          result.newScore,
          new Date().toISOString(),
        );
        // Update the memory row
        this.db.prepare(
          'UPDATE memories SET memory_state = ?, activation_score = ? WHERE id = ?',
        ).run(result.newState, result.newScore, memoryId);

        // Create gist when transitioning to gist_only (inline within transaction)
        if (result.newState === 'gist_only' && fromState !== 'gist_only') {
          this.createMemoryGist(memoryId);
        }

        // T-008: Auto-flag for review when transitioning to dimmed/shadow/gist_only.
        // Pinned and non_forgettable memories are already excluded by the query above.
        if (shouldAutoFlagForReview(result.newState, fromState)) {
          this.db.prepare(
            `UPDATE memories SET status = 'needs_review', review_reason = ?, reviewed_at = ? WHERE id = ?`,
          ).run(`decay: ${result.newState}`, new Date().toISOString(), memoryId);
        }
      }
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      return { processed: batchResult.processed, transitioned: 0, results: [] };
    }

    // Timed prune: after commit, remove purged originals not retained by revive gate
    this.prunePurgedMemories();

    return batchResult;
  }

  // ── Gradual forgetting: gist creation, revive gate, timed prune ──

  /**
   * Create a memory_gists row for a memory transitioning to gist_only.
   * Extracts the core fact from the memory content and stores a compact gist
   * that survives the original's eventual purge.
   */
  createMemoryGist(memoryId: string): string {
    const row = this.db.prepare(
      'SELECT id, content, category, tags FROM memories WHERE id = ?',
    ).get(memoryId) as { id: string; content: string; category: string; tags: string } | undefined;

    if (!row) throw new MemoryNotFoundError(memoryId);

    const gistId = randomUUID();
    const now = new Date().toISOString();
    const tags: string[] = JSON.parse(row.tags);

    // Derive gist fields from the memory content.
    // core_fact: first sentence or first 200 chars
    const firstSentence = row.content.match(/^[^.!?]+[.!?]/);
    const coreFact = firstSentence
      ? firstSentence[0].trim()
      : row.content.slice(0, 200).trim();

    // why_it_matters: category-based reasoning
    const whyMap: Record<string, string> = {
      'bug-fix': 'Prevents recurring bugs and regression',
      'architecture': 'Shapes structural decisions across the codebase',
      'security': 'Critical for security posture',
      'performance': 'Impacts system performance characteristics',
      'learning': 'Captures hard-won knowledge',
      'debugging': 'Debugging insight that may recur',
      'code-solution': 'Reusable code pattern',
      'tool-usage': 'Tool configuration or usage pattern',
      'observation': 'Environmental observation',
      'personal': 'Personal workflow preference',
      'relationship': 'Team or collaboration context',
      'other': 'General knowledge',
    };
    const whyItMatters = whyMap[row.category] ?? 'Retained for future reference';

    // revive_triggers: derived from tags + category keywords
    const triggers = [...tags.slice(0, 5), row.category];

    this.db.prepare(
      `INSERT INTO memory_gists (id, original_id, original_source, kind, core_fact, why_it_matters, revive_triggers, non_forgettable, created_at, updated_at, status)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      gistId, memoryId, 'memory', row.category, coreFact, whyItMatters,
      JSON.stringify(triggers), 1, now, now, 'active',
    );

    return gistId;
  }

  /**
   * Revive-before-prune gate: evaluates whether a deep-decay memory should be
   * revived, pruned, or deferred. Records the decision in memory_revive_reviews.
   *
   * Decision logic:
   *   - confidence >= 0.7  → 'revive' (reinforce the memory)
   *   - confidence < 0.7   → 'defer' (leave for next evaluation cycle)
   *   - explicit prune=true → 'prune' (caller explicitly requests removal)
   *
   * Returns the decision and the review record id.
   */
  reviveMemory(memoryId: string, opts?: { confidence?: number; prune?: boolean }): {
    decision: 'revive' | 'prune' | 'defer';
    reviewId: string;
  } {
    const row = this.db.prepare(
      'SELECT id, memory_state, activation_score, content FROM memories WHERE id = ?',
    ).get(memoryId) as {
      id: string; memory_state: string; activation_score: number; content: string;
    } | undefined;

    if (!row) throw new MemoryNotFoundError(memoryId);

    const confidence = opts?.confidence ?? 0.5;
    const reviewId = randomUUID();
    const now = new Date().toISOString();

    let decision: 'revive' | 'prune' | 'defer';
    let reason: string;

    if (opts?.prune) {
      decision = 'prune';
      reason = 'explicit prune requested';
    } else if (confidence >= 0.7) {
      decision = 'revive';
      reason = `confidence ${confidence.toFixed(2)} >= 0.7 threshold`;
    } else {
      decision = 'defer';
      reason = `confidence ${confidence.toFixed(2)} < 0.7 threshold; deferred for next cycle`;
    }

    // Find the associated gist (if any) for traceability
    const gist = this.db.prepare(
      'SELECT id FROM memory_gists WHERE original_id = ? ORDER BY created_at DESC LIMIT 1',
    ).get(memoryId) as { id: string } | undefined;

    this.db.prepare(
      `INSERT INTO memory_revive_reviews (id, memory_id, decision, reason, model, created_at, gist_id, confidence)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(reviewId, memoryId, decision, reason, null, now, gist?.id ?? null, confidence);

    // Execute the decision
    if (decision === 'revive') {
      this.reinforceMemory(memoryId);
    } else if (decision === 'prune') {
      this.purgeMemoryAndEmbedding(memoryId);
    }
    // 'defer' → no action, just the review record

    return { decision, reviewId };
  }

  /**
   * Timed prune: permanently delete the original memory row (and its embedding)
   * for memories that have reached 'purged' state AND are not retained by a
   * recent revive gate. The associated gist is preserved.
   *
   * Returns the count of pruned memories.
   */
  prunePurgedMemories(): number {
    // Find purged memories that have NOT been revived in their last review
    const purged = this.db.prepare(
      `SELECT m.id FROM memories m
       WHERE m.memory_state = 'purged'
         AND NOT EXISTS (
           SELECT 1 FROM memory_revive_reviews r
           WHERE r.memory_id = m.id
             AND r.decision = 'revive'
             AND r.created_at > datetime('now', '-30 days')
         )`,
    ).all() as Array<{ id: string }>;

    let pruned = 0;
    for (const mem of purged) {
      this.purgeMemoryAndEmbedding(mem.id);
      // Mark associated gists as original_pruned_at
      this.db.prepare(
        'UPDATE memory_gists SET original_pruned_at = ?, updated_at = ? WHERE original_id = ?',
      ).run(new Date().toISOString(), new Date().toISOString(), mem.id);
      pruned++;
    }

    return pruned;
  }

  /**
   * Hard-delete a memory row and its embedding. Used by revive gate (prune decision)
   * and timed purge. The associated gist survives.
   */
  private purgeMemoryAndEmbedding(memoryId: string): void {
    this.db.prepare('DELETE FROM memory_embeddings WHERE id = ?').run(memoryId);
    this.db.prepare('DELETE FROM memories WHERE id = ?').run(memoryId);
  }

  // ── Storage ──────────────────────────────────────────────────────

  async store(request: StoreRequest): Promise<Memory | StoreResult> {
    const now = new Date().toISOString();
    const id = request.id ?? randomUUID();
    const tags = JSON.stringify(request.tags ?? []);
    const embedding = await this.embedder.embed(request.content, 'passage');
    const provenance = request.provenance;
    const requestedStatus = request.status; // undefined → leave existing status untouched on conflict

    // ── Collision detection (opt-in) ──
    // W2 fix: detect collisions BEFORE upsert. When candidates exist, return
    // { judgment_required: true, candidates } WITHOUT persisting the new memory.
    // The caller must judge (not_conflict → re-call store) or resolve otherwise.
    if (request.detect_conflicts) {
      const threshold = request.conflict_threshold ?? parseFloat(process.env.MEMORY_CONFLICT_THRESHOLD ?? '0.92');
      const candidates = this.detectCollisions(embedding, threshold, request.content, id);
      if (candidates.length > 0) {
        // R3 fix: durable pending draft OUTSIDE retrieval; judge() resolves it after restart.
        const judgment_ids = this.persistConflictDraft(id, request, now, embedding, candidates);
        return { judgment_required: true, candidates, draft_id: id, judgment_ids };
      }
      const stored = await this._doStore(id, request, now, tags, embedding, provenance, requestedStatus);
      return { stored, judgment_required: false };
    }

    const stored = await this._doStore(id, request, now, tags, embedding, provenance, requestedStatus);
    return stored;
  }

  /**
   * Internal store implementation — performs the actual INSERT/UPDATE.
   * Separated from store() so collision detection can run first.
   */
  private async _doStore(
    id: string,
    request: StoreRequest,
    now: string,
    tags: string,
    embedding: number[],
    provenance: StoreRequest['provenance'],
    requestedStatus: StoreRequest['status'],
  ): Promise<Memory> {

    this.db.exec('BEGIN');
    try {
      // Step 1 — try a fresh INSERT. OR IGNORE leaves the existing row untouched
      // on conflict so we can apply the merge rules explicitly below.
      const insert = this.db.prepare(
        `INSERT OR IGNORE INTO memories
          (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version,
           origin, source_tool, session_id, content_hash, status)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?)`,
      ).run(
        id, request.content, request.category, tags, now, now, 0, now, 1,
        provenance?.origin ?? 'unknown',
        provenance?.sourceTool ?? null,
        provenance?.sessionId ?? null,
        provenance?.contentHash ?? null,
        requestedStatus ?? 'active',
      );

      if (insert.changes === 0) {
        // Step 2 — conflict: upsert with safe merge rules.
        //   • created_at is NOT in the SET clause → preserved.
        //   • version increments (never resets).
        //   • provenance fields use COALESCE(NULL → keep existing).
        //   • status only changes when the caller explicitly provided one.
        this.db.prepare(
          `UPDATE memories SET
             content      = ?,
             category     = ?,
             tags         = ?,
             updated_at   = ?,
             version      = version + 1,
             origin       = COALESCE(?, origin),
             source_tool  = COALESCE(?, source_tool),
             session_id   = COALESCE(?, session_id),
             content_hash = COALESCE(?, content_hash),
             status       = CASE WHEN ? IS NOT NULL THEN ? ELSE status END
           WHERE id = ?`,
        ).run(
          request.content, request.category, tags, now,
          provenance?.origin ?? null,
          provenance?.sourceTool ?? null,
          provenance?.sessionId ?? null,
          provenance?.contentHash ?? null,
          requestedStatus ?? null,
          requestedStatus ?? null,
          id,
        );
      }

      this.db.prepare(
        `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
      ).run(id, embeddingToBlob(embedding));

      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }

    // Auto-create / refresh topic alias when requested (best-effort, post-commit).
    if (request.topicKey) {
      await this.upsertTopicAlias(request.topicKey, id);
    }

    // Auto-link this memory to the active session's most recent prompt
    // (best-effort, fire-and-forget). No-op when no session context is set.
    this.autoLinkToActiveSession(id);

    // Read back the canonical row so created_at (preserved), version
    // (incremented), status, and provenance all reflect truth.
    const row = this.db.prepare(`SELECT * FROM memories WHERE id = ?`).get(id) as unknown as MemoryRow;
    return rowToMemory(row);
  }

  // ── Conflict detection (v9) ────────────────────────────────────────

  /**
   * Detect potential collision candidates for a new memory.
   *
   * Runs cosine-similarity query against same-project memories. For each
   * candidate: if cosine > threshold AND FTS title overlap >= 60%, include
   * in results. Returns candidates sorted by similarity descending. With a
   * prospectiveId, the prospective row and already-judged pairs are excluded.
   */
  private detectCollisions(
    queryEmbedding: number[],
    threshold: number,
    content: string,
    prospectiveId?: string,
  ): CandidateMemory[] {
    // Fetch all existing embeddings + metadata
    const rows = this.db.prepare(
      `SELECT m.id, m.content, m.created_at, e.embedding
       FROM memories m
       JOIN memory_embeddings e ON e.id = m.id
       WHERE m.status != 'archived'
         AND (? IS NULL OR (m.id != ? AND NOT EXISTS (
           SELECT 1 FROM memory_relations r WHERE (r.memory_id_a = ? AND r.memory_id_b = m.id) OR (r.memory_id_b = ? AND r.memory_id_a = m.id))))`,
    ).all(prospectiveId ?? null, prospectiveId ?? null, prospectiveId ?? null, prospectiveId ?? null) as Array<{
      id: string;
      content: string;
      created_at: string;
      embedding: Buffer | Uint8Array;
    }>;

    if (rows.length === 0) return [];

    // Precompute query tokens for FTS overlap check
    const queryTokens = new Set(
      content.toLowerCase().split(/\s+/).filter(t => t.length > 2),
    );

    const candidates: CandidateMemory[] = [];

    for (const row of rows) {
      const similarity = cosineDistance(queryEmbedding, blobToEmbedding(row.embedding));
      if (similarity < threshold) continue;

      // FTS title overlap: >= 60% of query tokens must appear in candidate
      if (queryTokens.size > 0) {
        const candidateLower = row.content.toLowerCase();
        let overlap = 0;
        for (const token of queryTokens) {
          if (candidateLower.includes(token)) overlap++;
        }
        const overlapRatio = overlap / queryTokens.size;
        if (overlapRatio < 0.6) continue;
      }

      candidates.push({
        id: row.id,
        similarity,
        content_preview: row.content.slice(0, 200),
        createdAt: row.created_at,
      });
    }

    // Sort by similarity descending
    candidates.sort((a, b) => b.similarity - a.similarity);
    return candidates;
  }

  // ── Memory relations (v9) ──────────────────────────────────────────

  /**
   * Persist a relation between two memories. Validates the relation enum.
   * Returns the persisted MemoryRelation.
   */
  addMemoryRelation(opts: {
    memoryIdA: string;
    memoryIdB: string;
    relation: RelationType;
    confidence?: number;
    reasoning?: string;
    markedBy?: string;
  }): MemoryRelation {
    if (!VALID_RELATIONS.includes(opts.relation)) {
      throw new Error(
        `Invalid relation '${opts.relation}': must be one of ${VALID_RELATIONS.join(', ')}`,
      );
    }

    const id = randomUUID();
    const now = new Date().toISOString();

    this.db.prepare(
      `INSERT INTO memory_relations (id, memory_id_a, memory_id_b, relation, confidence, reasoning, marked_by, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id, opts.memoryIdA, opts.memoryIdB, opts.relation,
      opts.confidence ?? null, opts.reasoning ?? null, opts.markedBy ?? null, now,
    );

    return {
      id,
      memoryIdA: opts.memoryIdA,
      memoryIdB: opts.memoryIdB,
      relation: opts.relation,
      ...(opts.confidence !== undefined ? { confidence: opts.confidence } : {}),
      ...(opts.reasoning !== undefined ? { reasoning: opts.reasoning } : {}),
      ...(opts.markedBy !== undefined ? { markedBy: opts.markedBy } : {}),
      createdAt: now,
    };
  }

  /**
   * Get a specific relation between two memories (by ordered pair).
   * Returns null if no relation exists.
   */
  getMemoryRelation(memoryIdA: string, memoryIdB: string): MemoryRelation | null {
    const row = this.db.prepare(
      `SELECT id, memory_id_a, memory_id_b, relation, confidence, reasoning, marked_by, created_at
       FROM memory_relations
       WHERE memory_id_a = ? AND memory_id_b = ?
       LIMIT 1`,
    ).get(memoryIdA, memoryIdB) as {
      id: string;
      memory_id_a: string;
      memory_id_b: string;
      relation: string;
      confidence: number | null;
      reasoning: string | null;
      marked_by: string | null;
      created_at: string;
    } | undefined;

    if (!row) return null;

    return {
      id: row.id,
      memoryIdA: row.memory_id_a,
      memoryIdB: row.memory_id_b,
      relation: row.relation as RelationType,
      ...(row.confidence !== null ? { confidence: row.confidence } : {}),
      ...(row.reasoning !== null ? { reasoning: row.reasoning } : {}),
      ...(row.marked_by !== null ? { markedBy: row.marked_by } : {}),
      createdAt: row.created_at,
    };
  }

  /**
   * List all relations involving a memory (either as memory_id_a or memory_id_b).
   */
  listMemoryRelations(memoryId: string): MemoryRelation[] {
    const rows = this.db.prepare(
      `SELECT id, memory_id_a, memory_id_b, relation, confidence, reasoning, marked_by, created_at
       FROM memory_relations
       WHERE memory_id_a = ? OR memory_id_b = ?
       ORDER BY created_at DESC`,
    ).all(memoryId, memoryId) as Array<{
      id: string;
      memory_id_a: string;
      memory_id_b: string;
      relation: string;
      confidence: number | null;
      reasoning: string | null;
      marked_by: string | null;
      created_at: string;
    }>;

    return rows.map(row => ({
      id: row.id,
      memoryIdA: row.memory_id_a,
      memoryIdB: row.memory_id_b,
      relation: row.relation as RelationType,
      ...(row.confidence !== null ? { confidence: row.confidence } : {}),
      ...(row.reasoning !== null ? { reasoning: row.reasoning } : {}),
      ...(row.marked_by !== null ? { markedBy: row.marked_by } : {}),
      createdAt: row.created_at,
    }));
  }

  // ── compare / judge (v9 T-006) ───────────────────────────────────

  /**
   * Compare two memories semantically.
   * Returns similarity (cosine), shared entities, content diff summary, and recommendation.
   */
  async compare(idA: string, idB: string): Promise<CompareResult> {
    const memA = this.db.prepare(
      'SELECT id, content FROM memories WHERE id = ?',
    ).get(idA) as { id: string; content: string } | undefined;
    const memB = this.db.prepare(
      'SELECT id, content FROM memories WHERE id = ?',
    ).get(idB) as { id: string; content: string } | undefined;

    if (!memA) throw new MemoryNotFoundError(idA);
    if (!memB) throw new MemoryNotFoundError(idB);

    const embA = this.db.prepare(
      'SELECT embedding FROM memory_embeddings WHERE id = ?',
    ).get(idA) as { embedding: Buffer | Uint8Array } | undefined;
    const embB = this.db.prepare(
      'SELECT embedding FROM memory_embeddings WHERE id = ?',
    ).get(idB) as { embedding: Buffer | Uint8Array } | undefined;

    const similarity = (embA && embB)
      ? cosineDistance(blobToEmbedding(embA.embedding), blobToEmbedding(embB.embedding))
      : 0;

    const sharedEntities = extractSharedEntities(memA.content, memB.content);

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

  /**
   * Judge a candidate pair: persist a relation, optionally cascade.
   * judgmentId format: "newMemoryId:existingMemoryId"
   * For supersedes: archives the existing (older) memory.
   */
  async judge(judgmentId: string, relation: RelationType, reasoning?: string, confidence?: number): Promise<JudgeResult> {
    const colonIdx = judgmentId.indexOf(':');
    if (colonIdx <= 0 || colonIdx === judgmentId.length - 1) {
      throw new Error(
        `Invalid judgment_id '${judgmentId}': must be in format 'newId:existingId'`,
      );
    }

    if (!VALID_RELATIONS.includes(relation)) {
      throw new Error(
        `Invalid relation '${relation}': must be one of ${VALID_RELATIONS.join(', ')}`,
      );
    }

    const newId = judgmentId.slice(0, colonIdx);
    const existingId = judgmentId.slice(colonIdx + 1);

    // R3 fix: EXACT judgment-row lookup (PK — no ambiguous ':' split for namespaced IDs).
    // ONE transaction: judgment update, pending count, activation, removal — failure
    // rolls back (retryable). conflicts_with keeps the draft NONACTIVE, re-judgeable.
    const j = this.db.prepare('SELECT draft_id FROM conflict_draft_judgments WHERE id = ?').get(judgmentId) as { draft_id: string } | undefined;
    if (j) {
      this.db.exec('BEGIN');
      try {
        this.db.prepare('UPDATE conflict_draft_judgments SET resolution = ?, confidence = ?, reasoning = ?, resolved_at = ? WHERE id = ?')
          .run(relation, confidence ?? null, reasoning ?? null, new Date().toISOString(), judgmentId);
        const { pending, neg } = this.db.prepare(
          `SELECT SUM(resolution IS NULL) AS pending, SUM(resolution = 'conflicts_with') AS neg FROM conflict_draft_judgments WHERE draft_id = ?`,
        ).get(j.draft_id) as { pending: number; neg: number };
        let cascaded = false; let topicKey: string | undefined;
        if (pending === 0 && !neg) ({ cascaded, topicKey } = this.activateDraft(j.draft_id));
        this.db.exec('COMMIT');
        if (topicKey) await this.upsertTopicAlias(topicKey, j.draft_id);
        return { persisted: true, relation, ...(cascaded ? { cascaded: 'archived' as const } : {}) };
      } catch (e) {
        this.db.exec('ROLLBACK');
        throw e;
      }
    }

    // Verify both memories exist
    const newMem = this.db.prepare('SELECT id FROM memories WHERE id = ?').get(newId) as { id: string } | undefined;
    if (!newMem) throw new MemoryNotFoundError(newId);
    const existingMem = this.db.prepare('SELECT id FROM memories WHERE id = ?').get(existingId) as { id: string } | undefined;
    if (!existingMem) throw new MemoryNotFoundError(existingId);

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
      this.updateMemoryStatus(existingId, 'archived', `superseded by ${newId}`);
      cascaded = 'archived';
    }

    return { persisted: true, relation, ...(cascaded ? { cascaded } : {}) };
  }

  /** R3 fix: persist (idempotently) a pending draft + one pending judgment per candidate,
   * in tables OUTSIDE memories/memory_embeddings (invisible to retrieval). */
  private persistConflictDraft(
    id: string, request: StoreRequest, now: string, embedding: number[], candidates: CandidateMemory[],
  ): string[] {
    this.db.exec('BEGIN');
    try {
      this.db.prepare(
        `INSERT INTO conflict_drafts (id, request, embedding, created_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET request = excluded.request, embedding = excluded.embedding`,
      ).run(id, JSON.stringify(request), embeddingToBlob(embedding), now);
      // Deterministic retry: replace any prior pending judgments for this draft.
      this.db.prepare('DELETE FROM conflict_draft_judgments WHERE draft_id = ?').run(id);
      const judgmentIds = candidates.map(c => `${id}:${c.id}`);
      const insertJudgment = this.db.prepare('INSERT INTO conflict_draft_judgments (id, draft_id, candidate_id, created_at) VALUES (?, ?, ?, ?)');
      for (let i = 0; i < candidates.length; i++) insertJudgment.run(judgmentIds[i], id, candidates[i].id, now);
      this.db.exec('COMMIT');
      return judgmentIds;
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  /**
   * R3 fix: activate a fully-judged, non-conflicting draft INSIDE the caller's transaction.
   * Upsert keeps legacy _doStore row semantics (created_at kept; version++, COALESCE provenance,
   * explicit-status-only for pre-existing IDs); topicKey alias applied post-commit by the caller.
   */
  private activateDraft(draftId: string): { cascaded: boolean; topicKey?: string } {
    const d = this.db.prepare('SELECT request, embedding, created_at FROM conflict_drafts WHERE id = ?')
      .get(draftId) as { request: string; embedding: Buffer; created_at: string } | undefined;
    if (!d) throw new Error(`No pending draft '${draftId}'`);
    const req = JSON.parse(d.request) as StoreRequest;
    const tags = JSON.stringify(req.tags ?? []); const p = req.provenance;
    this.db.prepare(
      `INSERT INTO memories (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version, origin, source_tool, session_id, content_hash, status)
       VALUES (?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET content = excluded.content, category = excluded.category, tags = excluded.tags,
         updated_at = excluded.updated_at, version = memories.version + 1, origin = COALESCE(excluded.origin, memories.origin),
         source_tool = COALESCE(excluded.source_tool, memories.source_tool), session_id = COALESCE(excluded.session_id, memories.session_id),
         content_hash = COALESCE(excluded.content_hash, memories.content_hash), status = CASE WHEN ? IS NOT NULL THEN ? ELSE memories.status END`,
    ).run(
      draftId, req.content, req.category, tags, d.created_at, d.created_at, d.created_at,
      p?.origin ?? 'unknown', p?.sourceTool ?? null, p?.sessionId ?? null, p?.contentHash ?? null,
      req.status ?? 'active', req.status ?? null, req.status ?? null,
    );
    this.db.prepare('INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)').run(draftId, d.embedding);
    let cascaded = false;
    for (const j of this.db.prepare(
      'SELECT candidate_id, resolution, confidence, reasoning FROM conflict_draft_judgments WHERE draft_id = ? AND resolution IS NOT NULL',
    ).all(draftId) as Array<{ candidate_id: string; resolution: RelationType; confidence: number | null; reasoning: string | null }>) {
      this.addMemoryRelation({
        memoryIdA: draftId, memoryIdB: j.candidate_id, relation: j.resolution,
        confidence: j.confidence ?? undefined, reasoning: j.reasoning ?? undefined, markedBy: 'judge',
      });
      if (j.resolution === 'supersedes') {
        const at = new Date().toISOString();
        this.db.prepare(
          `UPDATE memories SET status = 'archived', review_reason = ?, reviewed_at = ?, updated_at = ?, version = version + 1 WHERE id = ?`,
        ).run(`superseded by ${draftId}`, at, at, at, j.candidate_id);
        cascaded = true;
      }
    }
    this.db.prepare('DELETE FROM conflict_drafts WHERE id = ?').run(draftId);
    return { cascaded, ...(req.topicKey ? { topicKey: req.topicKey } : {}) };
  }

  // ── Review queue (v9 T-008) ─────────────────────────────────────

  /**
   * List memories with status='needs_review', ordered by reviewed_at ASC
   * (oldest review request first). Supports optional project filter and limit.
   */
  async listReviewQueue(opts?: { projectId?: string; limit?: number }): Promise<{ count: number; items: Memory[] }> {
    const limit = opts?.limit ?? 50;
    const conditions: string[] = ["status = 'needs_review'"];
    const args: (string | number)[] = [];

    if (opts?.projectId) {
      // Memories don't have project_id directly — project scope is via DB isolation.
      // This parameter is reserved for future multi-project unified search.
      // For now, it's a no-op filter on the memory table.
    }

    // LIMIT is a SQL clause appended after ORDER BY, never a WHERE predicate.
    const rows = this.db.prepare(
      `SELECT * FROM memories WHERE ${conditions.join(' AND ')} ORDER BY reviewed_at ASC, created_at ASC LIMIT ?`,
    ).all(...args, limit) as unknown as MemoryRow[];

    return { count: rows.length, items: rows.map(rowToMemory) };
  }

  async storeBatch(memories: StoreRequest[]): Promise<Memory[]> {
    if (memories.length === 0) return [];

    // 1. Embed all memories first (async, before transaction)
    const embedded = await Promise.all(
      memories.map(async m => ({ m, vec: await this.embedder.embed(m.content, 'passage') }))
    );

    // 2. Write all in one transaction (synchronous db calls)
    const now = new Date().toISOString();
    const results: Memory[] = [];

    this.db.exec('BEGIN');
    try {
      for (const { m, vec } of embedded) {
        const id = (m as any).id ?? randomUUID();
        const tags = JSON.stringify(m.tags ?? []);
        const createdAt = (m as any).created_at ?? (m as any).createdAt ?? now;
        const updatedAt = (m as any).updated_at ?? (m as any).updatedAt ?? now;
        const accessCount = (m as any).access_count ?? 0;
        const lastAccessedAt = (m as any).last_accessed_at ?? now;
        const version = (m as any).version ?? 1;
        const provenance = m.provenance;

        this.db.prepare(
          `INSERT INTO memories
            (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version,
             origin, source_tool, session_id, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             content        = excluded.content,
             category       = excluded.category,
             tags           = excluded.tags,
             updated_at     = excluded.updated_at`,
        ).run(
          id, m.content, m.category, tags, createdAt, updatedAt, accessCount, lastAccessedAt, version,
          provenance?.origin ?? 'unknown',
          provenance?.sourceTool ?? null,
          provenance?.sessionId ?? null,
          provenance?.contentHash ?? null,
        );

        this.db.prepare(
          `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
        ).run(id, embeddingToBlob(vec));

        results.push({
          id,
          content: m.content,
          category: m.category,
          tags: m.tags ?? [],
          createdAt,
          updatedAt,
          version,
          ...(provenance ? { provenance } : {}),
        });
      }
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }

    return results;
  }

  // ── Access tracking ──────────────────────────────────────────────

  touchAccessed(id: string): void {
    // best-effort: no error if id doesn't exist
    this.db.prepare(
      'UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?',
    ).run(new Date().toISOString(), id);
  }

  /**
   * Bump access_count + last_accessed_at for session-ledger rows read through
   * supported retrieval paths (getSession). One UPDATE per (table, id) so only
   * the exact rows returned are touched. Best-effort: never throws.
   */
  private touchSessionAccess(
    promptIds: string[],
    responseIds: string[],
    toolCallIds: string[],
    summaryIds: string[],
  ): void {
    const now = new Date().toISOString();
    const tables: Array<[string, string[]]> = [
      ['session_prompts', promptIds],
      ['session_responses', responseIds],
      ['session_tool_calls', toolCallIds],
      ['session_summaries', summaryIds],
    ];
    try {
      for (const [table, ids] of tables) {
        if (ids.length === 0) continue;
        const placeholders = ids.map(() => '?').join(',');
        this.db.prepare(
          `UPDATE ${table} SET access_count = access_count + 1, last_accessed_at = ? WHERE id IN (${placeholders})`,
        ).run(now, ...ids);
      }
    } catch {
      // Access tracking must never fail a read.
    }
  }

  // ── Search ────────────────────────────────────────────────────────

  async search(
    query: string,
    mode: SearchMode = 'hybrid',
    filters: SearchFilters = {},
  ): Promise<SearchResult[]> {
    // T-013: When all_projects=true and sibling stores are configured,
    // delegate to searchAllProjects for multi-DB enumeration.
    if (filters.all_projects && this.siblingStores.length > 0) {
      return this.searchAllProjects(query, mode, filters);
    }

    const limit = filters.limit ?? 10;

    switch (mode) {
      case 'semantic':
        return this.semanticSearch(query, filters, limit);
      case 'keyword':
        return this.keywordSearch(query, filters, limit);
      case 'hybrid':
        return this.hybridSearch(query, filters, limit);
    }
  }

  private keywordSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
    trackAccess: boolean = true,
    rawScores?: Map<string, number>,
  ): SearchResult[] {
    if (!query.trim()) return [];

    try {
      const escaped = escapeFtsQuery(query);
      // bm25() returns negative values: lower (more negative) = better match
      // We negate to get positive score where higher = better
      type FtsRow = MemoryRow & { fts_score: number };
      const rows = this.db.prepare(`
        SELECT m.*, -bm25(memories_fts) AS fts_score
        FROM memories_fts
        JOIN memories m ON m.id = memories_fts.id
        WHERE memories_fts MATCH ?
        ORDER BY bm25(memories_fts)
        LIMIT ?
      `).all(escaped, limit * 3) as unknown as FtsRow[];

      const ftsScoreMap = new Map(rows.map(r => [r.id, r.fts_score]));
      if (rawScores) {
        for (const [id, score] of ftsScoreMap) rawScores.set(id, score);
      }
      return this.applyFiltersAndDecay(rows, filters, (r) => ftsScoreMap.get(r.id) ?? 1, limit, trackAccess);
    } catch {
      return [];
    }
  }

  private async semanticSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
    trackAccess: boolean = true,
    rawScores?: Map<string, number>,
  ): Promise<SearchResult[]> {
    const queryVec = await this.embedder.embed(query, 'query');
    return this.vectorSearch(queryVec, filters, limit, null, trackAccess, rawScores);
  }

  private vectorSearch(
    queryVec: number[],
    filters: SearchFilters,
    limit: number,
    excludeId: string | null,
    trackAccess: boolean = true,
    rawScores?: Map<string, number>,
  ): SearchResult[] {
    // Fetch all embeddings + row data and compute cosine in-process
    const rows = this.db.prepare(
      `SELECT m.*, e.embedding
       FROM memories m
       JOIN memory_embeddings e ON e.id = m.id`,
    ).all() as unknown as (MemoryRow & { embedding: Buffer | Uint8Array })[];

    const scored = rows
      .filter(r => r.id !== excludeId)
      .map(r => ({
        row: r as MemoryRow,
        rawScore: cosineDistance(queryVec, blobToEmbedding(r.embedding)),
      }))
      .sort((a, b) => b.rawScore - a.rawScore);

    // Build scoreMap once (O(N)) before calling applyFiltersAndDecay
    const scoreMap = new Map(scored.map(s => [s.row.id, s.rawScore]));
    if (rawScores) {
      for (const [id, score] of scoreMap) rawScores.set(id, score);
    }
    return this.applyFiltersAndDecay(
      scored.map(s => s.row),
      filters,
      (r) => scoreMap.get(r.id) ?? 0,
      limit,
      trackAccess,
    );
  }

  private async hybridSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
  ): Promise<SearchResult[]> {
    return (await this._hybridSearchCore(query, filters, limit)).results;
  }

  private async _hybridSearchCore(
    query: string,
    filters: SearchFilters,
    limit: number,
  ): Promise<{ results: SearchResult[]; summary?: ExplainSummary }> {
    const explain = filters.explain ?? false;
    const kwRaw = explain ? new Map<string, number>() : undefined;
    const semRaw = explain ? new Map<string, number>() : undefined;

    // Pass trackAccess=false to avoid double-incrementing access_count
    const [keywordResults, semanticResults] = await Promise.all([
      Promise.resolve(this.keywordSearch(query, filters, limit * 2, false, kwRaw)),
      this.semanticSearch(query, filters, limit * 2, false, semRaw),
    ]);

    if (keywordResults.length === 0 && semanticResults.length === 0) return { results: [] };

    // Build merged result set via RRF
    const kwList = keywordResults.map(r => ({ id: r.memory.id, score: r.score }));
    const semList = semanticResults.map(r => ({ id: r.memory.id, score: r.score }));

    // Capture per-id rank before fusion (0-based)
    const kwRank = new Map(kwList.map(({ id }, i) => [id, i]));
    const semRank = new Map(semList.map(({ id }, i) => [id, i]));

    const merged = rrf([kwList, semList]);
    const idToMemory = new Map<string, Memory>();
    const idToRow = new Map<string, MemoryRow>();
    for (const r of [...keywordResults, ...semanticResults]) {
      idToMemory.set(r.memory.id, r.memory);
    }
    if (explain) {
      // Re-fetch rows for decay factor computation (explain path only)
      const ids = [...idToMemory.keys()];
      if (ids.length > 0) {
        const placeholders = ids.map(() => '?').join(',');
        const rows = this.db.prepare(
          `SELECT * FROM memories WHERE id IN (${placeholders})`,
        ).all(...ids) as unknown as MemoryRow[];
        for (const row of rows) idToRow.set(row.id, row);
      }
    }

    const finalResults = merged
      .slice(0, limit)
      .filter(({ id }) => idToMemory.has(id))
      .map(({ id, score: rrfScore }) => {
        const memory = idToMemory.get(id)!;
        const result: SearchResult = { memory, score: rrfScore };
        if (explain) {
          const row = idToRow.get(id);
          const { factor } = row
            ? applyDecayWithFactor(row, 1)
            : { factor: 1 };
          const kwIdx = kwRank.get(id);
          const semIdx = semRank.get(id);
          result.explain = {
            keyword: kwIdx !== undefined
              ? { rank: kwIdx, bm25: kwRaw?.get(id) ?? 0 }
              : null,
            semantic: semIdx !== undefined
              ? { rank: semIdx, cosine: semRaw?.get(id) ?? 0 }
              : null,
            rrfScore,
            decayFactor: factor,
            finalScore: rrfScore,
          };
        }
        return result;
      });

    // Touch access exactly once per final result
    for (const { memory } of finalResults) {
      this.touchAccessed(memory.id);
      // Reinforce on access (PR 2b)
      try { this.reinforceMemory(memory.id); } catch { /* best-effort */ }
    }

    // Fallback: if one list was empty, fall back to the other (already tracked above)
    if (finalResults.length === 0) {
      const fallback = keywordResults.length > 0 ? keywordResults : semanticResults;
      for (const { memory } of fallback.slice(0, limit)) {
        this.touchAccessed(memory.id);
        try { this.reinforceMemory(memory.id); } catch { /* best-effort */ }
      }
      return { results: fallback.slice(0, limit) };
    }

    const summary: ExplainSummary | undefined = explain
      ? {
          keywordCandidates: keywordResults.length,
          semanticCandidates: semanticResults.length,
          fusedCount: finalResults.length,
          query: { fts: escapeFtsQuery(query) },
        }
      : undefined;

    return { results: finalResults, summary };
  }

  /**
   * Like search(), but when filters.explain is true (and mode === 'hybrid')
   * returns a per-call ExplainSummary alongside the results.
   * Only available on SqliteMemoryStore directly (not on MemoryStore interface).
   */
  async searchWithSummary(
    query: string,
    mode: SearchMode,
    filters: SearchFilters,
  ): Promise<{ results: SearchResult[]; summary?: ExplainSummary }> {
    if (!filters.explain || mode !== 'hybrid') {
      return { results: await this.search(query, mode, filters) };
    }
    const limit = filters.limit ?? 10;
    return this._hybridSearchCore(query, filters, limit);
  }

  // ── Unified session search (PR 1) ─────────────────────────────────

  /**
   * Compacted-first unified search over the session ledger.
   *
   * Searches compacted summaries first; raw prompt/response/tool-call rows are
   * opt-in via `includeRawSessionItems`. Each candidate is scored by a keyword
   * signal (token overlap) and a semantic signal (cosine over the stored
   * embedding), fused with RRF. With `sourceBalancing` on, results are
   * interleaved round-robin across sources so verbose raw rows cannot crowd out
   * dense summaries. PR 1 fixes sourceMultiplier and accessMultiplier at 1 (no
   * decay scoring yet — that is PR 2).
   */
  async searchSessions(query: UnifiedSearchQuery): Promise<SessionSearchResult[]> {
    const q = parseSearchQuery(query);

    // Resolve which session sources to scan: summaries always (compacted-first);
    // raw rows only when opted in. Honor an explicit `sources` filter too.
    const requested = new Set(q.sources);
    const sources: UnifiedSearchSource[] = [];
    for (const s of COMPACTED_SESSION_SOURCES) {
      if (requested.has(s)) sources.push(s);
    }
    if (q.includeRawSessionItems) {
      for (const s of RAW_SESSION_SOURCES) {
        if (requested.has(s)) sources.push(s);
      }
    }
    if (sources.length === 0) return [];

    const queryVec = await this.embedder.embed(q.query, 'query');
    const tokens = q.query.toLowerCase().split(/\s+/).filter(Boolean);

    // Collect scored candidates per source so balancing can interleave them.
    const perSource = new Map<UnifiedSearchSource, ScoredSessionCandidate[]>();
    for (const source of sources) {
      perSource.set(source, this.scoreSessionSource(source, q, tokens, queryVec));
    }

    const ordered = q.sourceBalancing
      ? interleaveBalanced(sources, perSource)
      : flattenByScore(sources, perSource);

    const results = ordered
      .slice(0, q.limit)
      .map(c => this.toSessionSearchResult(c, q));

    // PR 2b: reinforce memories referenced by compacted summaries.
    // Only summaries carry memory_ids in the PR 1 schema. We query the
    // memory_ids column for the surfaced summary IDs (not fetched by
    // scoreSessionSource which only selects content/embedding/status).
    const summaryIds = ordered
      .slice(0, q.limit)
      .filter(c => c.source === 'session_summary')
      .map(c => c.row.id);
    if (summaryIds.length > 0) {
      try {
        const placeholders = summaryIds.map(() => '?').join(',');
        const rows = this.db.prepare(
          `SELECT memory_ids FROM session_summaries WHERE id IN (${placeholders})`,
        ).all(...summaryIds) as { memory_ids: string | null }[];
        for (const row of rows) {
          if (!row.memory_ids) continue;
          try {
            const ids: string[] = JSON.parse(row.memory_ids);
            for (const memId of ids) {
              try { this.reinforceMemory(memId); } catch { /* best-effort */ }
            }
          } catch { /* malformed JSON → skip */ }
        }
      } catch { /* best-effort */ }
    }

    return results;
  }

  /**
   * Score every active row of one session source against the query. Returns the
   * candidates sorted best-first (RRF over keyword + semantic ranks).
   */
  private scoreSessionSource(
    source: UnifiedSearchSource,
    q: ResolvedSearchQuery,
    tokens: string[],
    queryVec: number[],
  ): ScoredSessionCandidate[] {
    const meta = SESSION_SOURCE_META[source];
    if (!meta) return [];

    const where: string[] = [];
    const args: string[] = [];
    // Status: 'all' disables the filter; otherwise match the requested status.
    if (q.status !== 'all') {
      where.push('status = ?');
      args.push(q.status);
    }
    if (q.sessionId) {
      where.push('session_id = ?');
      args.push(q.sessionId);
    }
    // v7: project-scoped session search (default-deny).
    // When projectId is set, exclude rows from OTHER projects always.
    // Legacy rows (NULL project_id) are excluded by default and only
    // included when includeLegacy is explicitly true.
    if (q.projectId) {
      if (q.includeLegacy) {
        where.push('(project_id = ? OR project_id IS NULL)');
        args.push(q.projectId);
      } else {
        where.push('project_id = ?');
        args.push(q.projectId);
      }
    }
    if (q.after) {
      where.push('created_at >= ?');
      args.push(q.after);
    }
    if (q.before) {
      where.push('created_at <= ?');
      args.push(q.before);
    }
    const whereSql = where.length ? `WHERE ${where.join(' AND ')}` : '';

    const rows = this.db.prepare(
      `SELECT id, session_id, ${meta.contentCol} AS content, created_at, status, embedding
       FROM ${meta.table} ${whereSql}`,
    ).all(...args) as unknown as SessionCandidateRow[];

    // Compute keyword (token overlap) and semantic (cosine) scores.
    const scored = rows.map(row => {
      const content = row.content ?? '';
      const lower = content.toLowerCase();
      const keywordScore = tokens.length
        ? tokens.reduce((n, t) => (lower.includes(t) ? n + 1 : n), 0) / tokens.length
        : 0;
      const semanticScore = row.embedding
        ? cosineDistance(queryVec, blobToEmbedding(row.embedding))
        : 0;
      return { row, source, content, keywordScore, semanticScore };
    });

    // Keep only rows with some signal so empty-keyword + zero-vector noise drops.
    const candidates = scored.filter(s => s.keywordScore > 0 || s.semanticScore > 0);

    const kwRanked = [...candidates].sort((a, b) => b.keywordScore - a.keywordScore);
    const semRanked = [...candidates].sort((a, b) => b.semanticScore - a.semanticScore);
    const kwRank = new Map(kwRanked.map((c, i) => [c.row.id, i]));
    const semRank = new Map(semRanked.map((c, i) => [c.row.id, i]));

    const fused = rrf([
      kwRanked.map(c => ({ id: c.row.id, score: c.keywordScore })),
      semRanked.map(c => ({ id: c.row.id, score: c.semanticScore })),
    ]);
    const byId = new Map(candidates.map(c => [c.row.id, c]));

    return fused
      .map(({ id, score }) => {
        const c = byId.get(id)!;
        return {
          ...c,
          score,
          keywordRank: kwRank.get(id) ?? -1,
          semanticRank: semRank.get(id) ?? -1,
        } as ScoredSessionCandidate;
      })
      .filter(Boolean);
  }

  /** Map a scored candidate to the public SessionSearchResult shape. */
  private toSessionSearchResult(
    c: ScoredSessionCandidate,
    q: ResolvedSearchQuery,
  ): SessionSearchResult {
    const meta = SESSION_SOURCE_META[c.source]!;
    const decayFactor = scoreSessionRow({
      updated_at: c.row.created_at,
      access_count: 0,
      last_accessed_at: c.row.created_at,
    });
    const hint = generateDecisionHint(c.content, c.score, q.maxContentChars);

    const result: SessionSearchResult = {
      id: c.row.id,
      source: c.source,
      contentRef: `${c.source}:${c.row.id}`,
      decisionHint: hint,
      score: c.score,
      status: c.row.status ?? 'active',
      createdAt: c.row.created_at,
      sessionId: c.row.session_id,
      isCompact: meta.isCompact,
    };
    if (q.includeContent) {
      result.content = c.content.length > q.maxContentChars
        ? c.content.slice(0, q.maxContentChars)
        : c.content;
    }
    if (q.explain) {
      // PR 2b: activation_score multiplier for explain finalScore.
      // Session ranking `score` is left un-decayed (PR 1 stability), but
      // the explain's `finalScore` reflects the full decay picture.
      let activationScore = 1.0;
      try {
        const memRow = this.db.prepare(
          'SELECT activation_score FROM memories WHERE id = ?',
        ).get(c.row.id) as { activation_score: number | undefined } | undefined;
        activationScore = memRow?.activation_score ?? 1.0;
      } catch { /* best-effort */ }
      const explain: UnifiedSearchExplain = {
        keywordRank: c.keywordRank,
        semanticCosine: c.semanticScore,
        sourceMultiplier: 1,
        decayFactor: decayFactor * activationScore,
        accessMultiplier: 1,
        finalScore: c.score * decayFactor * activationScore,
      };
      result.explain = explain;
    }
    return result;
  }

  private applyFiltersAndDecay(
    rows: MemoryRow[],
    filters: SearchFilters,
    scoreFor: (r: MemoryRow) => number,
    limit: number,
    trackAccess: boolean = true,
  ): SearchResult[] {
    let filtered = rows;

    if (filters.category) {
      filtered = filtered.filter(r => r.category === filters.category);
    }
    if (filters.after) {
      filtered = filtered.filter(r => r.created_at >= filters.after!);
    }
    if (filters.before) {
      filtered = filtered.filter(r => r.created_at <= filters.before!);
    }
    if (filters.tags && filters.tags.length > 0) {
      filtered = filtered.filter(r => {
        const tags: string[] = JSON.parse(r.tags);
        return filters.tags!.some(t => tags.includes(t));
      }
      );
    }

    // Status filter: 'all' = no filter; a specific status = match only that;
    // undefined (default) = exclude 'archived'.
    if (filters.status === 'all') {
      // no filtering
    } else if (filters.status) {
      filtered = filtered.filter(r => (r.status ?? 'active') === filters.status);
    } else {
      filtered = filtered.filter(r => (r.status ?? 'active') !== 'archived');
    }

    const results = filtered
      .map(r => ({
        memory: rowToMemory(r),
        score: applyDecay(r, scoreFor(r)),
        _pinned: r.is_pinned ?? 0,
      }))
      .sort((a, b) => {
        // Pinned memories always sort first
        if (a._pinned !== b._pinned) return b._pinned - a._pinned;
        return b.score - a.score;
      })
      .slice(0, limit)
      .map(({ memory, score }) => ({ memory, score }));

    // Track access (best-effort); skip when caller handles it (e.g. hybridSearch)
    if (trackAccess) {
      for (const { memory } of results) {
        this.touchAccessed(memory.id);
        // Reinforce on access (PR 2b): resets timer, reverts early stages.
        try { this.reinforceMemory(memory.id); } catch { /* best-effort */ }
      }
    }

    return results;
  }

  // ── findRelated() ─────────────────────────────────────────────────

  async findRelated(memoryId: string, limit: number = 5): Promise<SearchResult[]> {
    const embRow = this.db.prepare(
      `SELECT embedding FROM memory_embeddings WHERE id = ?`,
    ).get(memoryId) as { embedding: Buffer | Uint8Array } | undefined;

    if (!embRow) throw new MemoryNotFoundError(memoryId);

    const queryVec = blobToEmbedding(embRow.embedding);
    return this.vectorSearch(queryVec, {}, limit, memoryId);
  }

  // ── listRecent() ──────────────────────────────────────────────────

  async listRecent(limit: number = 10, category?: MemoryCategory): Promise<Memory[]> {
    // Spec (memory-lifecycle.md): list_recent MUST exclude archived by default.
    // There is no `status` filter parameter on this method, so we always
    // hide archived rows here — callers wanting archived memories must use
    // search() with status: 'all' or status: 'archived'.
    let rows: MemoryRow[];
    if (category) {
      rows = this.db.prepare(
        `SELECT * FROM memories WHERE category = ? AND (status IS NULL OR status != 'archived') ORDER BY created_at DESC LIMIT ?`,
      ).all(category, limit) as unknown as MemoryRow[];
    } else {
      rows = this.db.prepare(
        `SELECT * FROM memories WHERE (status IS NULL OR status != 'archived') ORDER BY created_at DESC LIMIT ?`,
      ).all(limit) as unknown as MemoryRow[];
    }
    return rows.map(rowToMemory);
  }

  // ── update() with OCC ─────────────────────────────────────────────

  /**
   * Update a memory by id. Reads the current version and uses it as the
   * expected version for the OCC check. Use `updateWithVersion` to provide
   * an explicit expected version for testing or true concurrent update scenarios.
   */
  async update(id: string, updates: UpdateRequest): Promise<Memory> {
    const existing = this.db.prepare(
      'SELECT version FROM memories WHERE id = ?',
    ).get(id) as { version: number } | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    return this._updateWithVersion(id, updates, existing.version);
  }

  /**
   * Update with an explicit expected version for OCC. Throws ConcurrentUpdateError
   * if the current version does not match expectedVersion.
   */
  private async _updateWithVersion(
    id: string,
    updates: UpdateRequest,
    expectedVersion: number,
  ): Promise<Memory> {
    const existing = this.db.prepare(
      `SELECT * FROM memories WHERE id = ?`,
    ).get(id) as MemoryRow | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    const now = new Date().toISOString();
    const newContent = updates.content ?? existing.content;
    const newCategory = updates.category ?? existing.category;
    const newTags = updates.tags ? JSON.stringify(updates.tags) : existing.tags;
    const embedding = updates.content !== undefined
      ? await this.embedder.embed(newContent, 'passage')
      : null;

    this.db.exec('BEGIN');
    try {
      const result = this.db.prepare(
        `UPDATE memories
         SET content = ?, category = ?, tags = ?, updated_at = ?, version = version + 1
         WHERE id = ? AND version = ?`,
      ).run(newContent, newCategory, newTags, now, id, expectedVersion) as { changes: number };

      if (result.changes === 0) {
        const current = this.db.prepare(
          `SELECT version FROM memories WHERE id = ?`,
        ).get(id) as { version: number } | undefined;

        if (!current) throw new MemoryNotFoundError(id);
        throw new ConcurrentUpdateError(id, expectedVersion, current.version);
      }

      if (embedding) {
        this.db.prepare(
          `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
        ).run(id, embeddingToBlob(embedding));
      }

      const updated = this.db.prepare(
        `SELECT * FROM memories WHERE id = ?`,
      ).get(id) as unknown as MemoryRow;

      this.db.exec('COMMIT');

      // Auto-link the updated memory to the active session's most recent prompt
      // (best-effort, fire-and-forget). No-op when no session context is set.
      try {
        this.autoLinkToActiveSession(id);
      } catch {
        // Swallow: auto-link must never fail the update.
      }

      return rowToMemory(updated);
    } catch (e) {
      try {
        this.db.exec('ROLLBACK');
      } catch {
        // Ignore rollback failure so the original error is preserved.
      }
      throw e;
    }
  }

  // ── delete() ─────────────────────────────────────────────────────

  async delete(id: string): Promise<void> {
    const result = this.db.prepare(
      `DELETE FROM memories WHERE id = ?`,
    ).run(id) as { changes: number };

    if (result.changes === 0) throw new MemoryNotFoundError(id);
  }

  // ── stats() ──────────────────────────────────────────────────────

  async stats(): Promise<MemoryStats> {
    const rows = this.db.prepare(
      `SELECT * FROM memories`,
    ).all() as unknown as MemoryRow[];

    if (rows.length === 0) {
      return {
        totalMemories: 0,
        byCategory: {},
        oldestMemory: null,
        newestMemory: null,
        neverAccessed: 0,
        belowPruneThreshold: 0,
        avgAccessCount: 0,
        mostAccessed: [],
      };
    }

    const byCategory: Record<string, number> = {};
    let neverAccessed = 0;
    let belowPruneThreshold = 0;
    let totalAccessCount = 0;
    const accessCounts: { id: string; content: string; count: number }[] = [];

    for (const row of rows) {
      byCategory[row.category] = (byCategory[row.category] ?? 0) + 1;

      const ac = row.access_count ?? 0;
      totalAccessCount += ac;
      if (ac === 0) neverAccessed++;

      accessCounts.push({
        id: row.id,
        content: row.content.slice(0, 80),
        count: ac,
      });

      if (!isEvergreen(row)) {
        const effHL = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
        const ageDays = Math.max(
          0,
          (Date.now() - new Date(row.updated_at).getTime()) / MS_PER_DAY,
        );
        const strength = effHL > 0 ? Math.pow(0.5, ageDays / effHL) : 0;
        if (strength < 0.05 || (ac === 0 && ageDays > 90)) {
          belowPruneThreshold++;
        }
      }
    }

    const sorted = [...rows].sort((a, b) => a.created_at.localeCompare(b.created_at));
    accessCounts.sort((a, b) => b.count - a.count);

    return {
      totalMemories: rows.length,
      byCategory,
      oldestMemory: sorted[0]?.created_at ?? null,
      newestMemory: sorted.at(-1)?.created_at ?? null,
      neverAccessed,
      belowPruneThreshold,
      avgAccessCount: rows.length > 0
        ? Math.round((totalAccessCount / rows.length) * 10) / 10
        : 0,
      mostAccessed: accessCounts.slice(0, 5),
    };
  }

  // ── prune() ──────────────────────────────────────────────────────

  async prune(options: PruneOptions = {}): Promise<PruneResult> {
    const { dryRun = true, minStrength = 0.05, maxDormantDays = 90 } = options;

    const rows = this.db.prepare(`SELECT * FROM memories`).all() as unknown as MemoryRow[];
    const candidates: PruneResult['candidates'] = [];

    for (const row of rows) {
      if (isEvergreen(row)) continue;

      const accessCount = row.access_count ?? 0;
      const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
      const ageMs = Date.now() - new Date(row.updated_at).getTime();
      const ageDays = Math.max(0, ageMs / MS_PER_DAY);
      const strength = effectiveHalfLife > 0
        ? Math.pow(0.5, ageDays / effectiveHalfLife)
        : 0;

      let reason = '';
      if (strength < minStrength) {
        reason = `strength ${strength.toFixed(4)} < ${minStrength}`;
      } else if (accessCount === 0 && ageDays > maxDormantDays) {
        reason = `never accessed, ${Math.floor(ageDays)} days old > ${maxDormantDays}`;
      }

      if (reason) {
        candidates.push({
          id: row.id,
          content: row.content.slice(0, 120),
          strength,
          reason,
        });
      }
    }

    let pruned = 0;
    if (!dryRun && candidates.length > 0) {
      for (const c of candidates) {
        await this.delete(c.id);
        pruned++;
      }
    }

    return {
      pruned,
      inspected: rows.length,
      dryRun,
      candidates,
    };
  }

  // ── Topic aliases (PR-1: implemented; needed by store() topicKey wiring) ──

  async upsertTopicAlias(topicKey: string, memoryId: string): Promise<void> {
    const now = new Date().toISOString();
    // The memory_topics table ships with a composite PRIMARY KEY
    // (topic_key, memory_id) from PR-1's migration v2. Spec semantics
    // require that re-aliasing an existing topic_key to a NEW memory
    // replaces the previous target (resolveTopic returns one memory).
    // Composite-PK ON CONFLICT cannot do that in-place, so we
    // DELETE-then-INSERT inside an explicit transaction. This is
    // idempotent and avoids introducing a v3 migration.
    this.db.exec('BEGIN');
    try {
      this.db.prepare(`DELETE FROM memory_topics WHERE topic_key = ?`).run(topicKey);
      this.db.prepare(
        `INSERT INTO memory_topics (topic_key, memory_id, created_at, updated_at)
         VALUES (?, ?, ?, ?)`,
      ).run(topicKey, memoryId, now, now);
      this.db.exec('COMMIT');
    } catch (err) {
      try { this.db.exec('ROLLBACK'); } catch { /* ignore */ }
      throw err;
    }
  }

  async resolveTopic(topicKey: string): Promise<Memory | null> {
    const row = this.db.prepare(
      `SELECT m.*
       FROM memory_topics t
       JOIN memories m ON m.id = t.memory_id
       WHERE t.topic_key = ?
       ORDER BY t.updated_at DESC
       LIMIT 1`,
    ).get(topicKey) as unknown as MemoryRow | undefined;

    return row ? rowToMemory(row) : null;
  }

  // ── Session context (in-memory; PR-1 foundation) ──────────────────

  setSessionContext(sessionId: string, projectId: string): void {
    this.sessionContext = { sessionId, projectId };
    // v7: backfill project_id on existing ledger rows for this session that
    // still carry a NULL project_id (legacy rows created before the v7
    // migration or before a session context was established). Only touches
    // NULL rows so an explicit re-scope call never clobbers an existing value.
    // Synchronous and best-effort: a few rows per session, no fire-and-forget.
    const sessionTables = [
      'session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries',
    ];
    try {
      for (const tbl of sessionTables) {
        this.db.prepare(
          `UPDATE ${tbl} SET project_id = ? WHERE session_id = ? AND project_id IS NULL`,
        ).run(projectId, sessionId);
      }
    } catch {
      // Backfill must never fail the session-context call.
    }
  }

  getSessionContext(): { sessionId: string; projectId: string } | null {
    return this.sessionContext;
  }

  // ── Lifecycle status transitions (T-003 / PR-2) ──────────────────────

  async updateMemoryStatus(id: string, status: MemoryStatus, reviewReason?: string): Promise<Memory> {
    const existing = this.db.prepare(
      `SELECT status FROM memories WHERE id = ?`,
    ).get(id) as { status: string | null } | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    const current = (existing.status ?? 'active') as MemoryStatus;
    this.validateStatusTransition(current, status, reviewReason);

    const now = new Date().toISOString();
    this.db.prepare(
      `UPDATE memories
       SET status = ?, review_reason = ?, reviewed_at = ?, updated_at = ?, version = version + 1
       WHERE id = ?`,
    ).run(status, reviewReason ?? null, now, now, id);

    const row = this.db.prepare(`SELECT * FROM memories WHERE id = ?`).get(id) as unknown as MemoryRow;
    return rowToMemory(row);
  }

  /**
   * Validate a lifecycle transition.
   *
   * Rules:
   *   • active → needs_review      : always allowed
   *   • active → archived          : requires reviewReason
   *   • needs_review → active      : requires reviewReason
   *   • needs_review → archived    : always allowed
   *   • archived → *               : NOT allowed (use store() with new content)
   *   • same → same                : allowed (updates reason/reviewed_at)
   *
   * Throws a descriptive Error on any illegal transition.
   */
  private validateStatusTransition(from: MemoryStatus, to: MemoryStatus, reason?: string): void {
    if (from === to) return; // idempotent / metadata-only update

    if (from === 'archived') {
      throw new Error(
        `Invalid status transition: 'archived' is terminal. ` +
        `Cannot move to '${to}'. Use store() with new content to create a fresh memory.`,
      );
    }

    const needsReason = (from === 'active' && to === 'archived') ||
                        (from === 'needs_review' && to === 'active');
    if (needsReason && !reason) {
      throw new Error(
        `Invalid status transition: '${from}' → '${to}' requires a reviewReason.`,
      );
    }

    // All remaining transitions (active→needs_review, needs_review→archived) are valid.
  }

  // ── Session ledger store methods (T-004 / PR-2) ──────────────────────

  async storeSessionPrompt(sessionId: string, rawPrompt: string): Promise<SessionPrompt> {
    const id = randomUUID();
    const now = new Date().toISOString();
    const embedding = await this.embedder.embed(rawPrompt, 'passage');

    this.db.prepare(
      `INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, updated_at, access_count, last_accessed_at, status, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(id, sessionId, rawPrompt, now, now, 0, now, 'active', embeddingToBlobOrNull(embedding));

    return { id, sessionId, rawPrompt, createdAt: now, updatedAt: now, accessCount: 0, lastAccessedAt: now, status: 'active' };
  }

  async storeSessionResponse(sessionId: string, promptId: string, fullResponse: string): Promise<SessionResponse> {
    // Validate the prompt exists — gives a clear error instead of a raw FK violation.
    const prompt = this.db.prepare(
      `SELECT 1 AS ok FROM session_prompts WHERE id = ?`,
    ).get(promptId);
    if (!prompt) {
      throw new Error(`Prompt not found: ${promptId} (cannot store response)`);
    }

    const id = randomUUID();
    const now = new Date().toISOString();
    const embedding = await this.embedder.embed(fullResponse, 'passage');

    this.db.prepare(
      `INSERT INTO session_responses (id, session_id, prompt_id, full_response, created_at, updated_at, access_count, last_accessed_at, status, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(id, sessionId, promptId, fullResponse, now, now, 0, now, 'active', embeddingToBlobOrNull(embedding));

    return { id, sessionId, promptId, fullResponse, createdAt: now, updatedAt: now, accessCount: 0, lastAccessedAt: now, status: 'active' };
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
        `Invalid importance '${params.importance}': must be 'low', 'normal', or 'high'`,
      );
    }

    const id = randomUUID();
    const now = new Date().toISOString();
    // Embed from the richest available text for downstream semantic lookups.
    const embedText = params.resultSummary || params.toolName;
    const embedding = await this.embedder.embed(embedText, 'passage');

    this.db.prepare(
      `INSERT INTO session_tool_calls
         (id, session_id, prompt_id, response_id, tool_name, args_safe, result_summary, importance, created_at, updated_at, access_count, last_accessed_at, status, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id, params.sessionId, params.promptId ?? null, params.responseId ?? null,
      params.toolName, params.argsSafe ?? null, params.resultSummary ?? null,
      params.importance ?? null, now, now, 0, now, 'active', embeddingToBlobOrNull(embedding),
    );

    const result: SessionToolCall = {
      id,
      sessionId: params.sessionId,
      toolName: params.toolName,
      createdAt: now,
      updatedAt: now,
      accessCount: 0,
      lastAccessedAt: now,
      status: 'active',
    };
    if (params.promptId) result.promptId = params.promptId;
    if (params.responseId) result.responseId = params.responseId;
    if (params.argsSafe) result.argsSafe = params.argsSafe;
    if (params.resultSummary) result.resultSummary = params.resultSummary;
    if (params.importance) result.importance = params.importance;
    return result;
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
    const id = randomUUID();
    const now = new Date().toISOString();
    const decisions = params.decisions ?? [];
    // Embed from summary + decisions so the summary is retrievable by either signal.
    const embedText = `${params.summary} ${decisions.join(' ')}`.trim();
    const embedding = await this.embedder.embed(embedText, 'passage');

    this.db.prepare(
      `INSERT INTO session_summaries
         (id, session_id, prompt_id, response_id, summary, decisions, key_titles, memory_ids, artifact_refs, created_at, updated_at, access_count, last_accessed_at, status, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id, params.sessionId, params.promptId ?? null, params.responseId ?? null,
      params.summary,
      params.decisions ? JSON.stringify(params.decisions) : null,
      params.keyTitles ? JSON.stringify(params.keyTitles) : null,
      params.memoryIds ? JSON.stringify(params.memoryIds) : null,
      params.artifactRefs ? JSON.stringify(params.artifactRefs) : null,
      now, now, 0, now, 'active', embeddingToBlobOrNull(embedding),
    );

    const result: SessionSummary = {
      id,
      sessionId: params.sessionId,
      summary: params.summary,
      createdAt: now,
      updatedAt: now,
      accessCount: 0,
      lastAccessedAt: now,
      status: 'active',
    };
    if (params.promptId) result.promptId = params.promptId;
    if (params.responseId) result.responseId = params.responseId;
    if (params.decisions) result.decisions = params.decisions;
    if (params.keyTitles) result.keyTitles = params.keyTitles;
    if (params.memoryIds) result.memoryIds = params.memoryIds;
    if (params.artifactRefs) result.artifactRefs = params.artifactRefs;
    return result;
  }

  async getSession(
    sessionId: string,
    opts?: { limit?: number; includeSummary?: boolean; includeForgotten?: boolean },
  ): Promise<SessionLedger> {
    const limit = opts?.limit;
    const limitClause = limit !== undefined ? ' LIMIT ?' : '';
    const limitArgs = limit !== undefined ? [limit] : [];
    // Spec (session-ledger.md): forgotten rows are excluded by default and
    // only surfaced when includeForgotten is explicitly true.
    const statusFilter = opts?.includeForgotten
      ? ''
      : `AND (status IS NULL OR status != 'forgotten')`;

    const promptRows = this.db.prepare(
      `SELECT id, session_id, raw_prompt, created_at, updated_at, access_count, last_accessed_at, status, project_id, symbol_key FROM session_prompts
       WHERE session_id = ? ${statusFilter} ORDER BY created_at ASC${limitClause}`,
    ).all(sessionId, ...limitArgs) as unknown as SessionPromptRow[];

    const responseRows = this.db.prepare(
      `SELECT id, session_id, prompt_id, full_response, created_at, updated_at, access_count, last_accessed_at, status, project_id, symbol_key FROM session_responses
       WHERE session_id = ? ${statusFilter} ORDER BY created_at ASC${limitClause}`,
    ).all(sessionId, ...limitArgs) as unknown as SessionResponseRow[];

    const toolCallRows = this.db.prepare(
      `SELECT id, session_id, prompt_id, response_id, tool_name, args_safe, result_summary, importance, created_at, updated_at, access_count, last_accessed_at, status, project_id, symbol_key
       FROM session_tool_calls WHERE session_id = ? ${statusFilter} ORDER BY created_at ASC${limitClause}`,
    ).all(sessionId, ...limitArgs) as unknown as SessionToolCallRow[];

    // includeSummary defaults to true; skip the summary query only when explicitly false.
    const summaryRows: SessionSummaryRow[] = opts?.includeSummary === false
      ? []
      : this.db.prepare(
          `SELECT id, session_id, prompt_id, response_id, summary, decisions, key_titles, memory_ids, artifact_refs, created_at, updated_at, access_count, last_accessed_at, status, project_id, symbol_key
           FROM session_summaries WHERE session_id = ? ${statusFilter} ORDER BY created_at ASC${limitClause}`,
        ).all(sessionId, ...limitArgs) as unknown as SessionSummaryRow[];

    // Spec (session-ledger-decay.md): supported reads MUST update access_count
    // and last_accessed_at. We bump every returned row once per call
    // (best-effort; never throws on the read path).
    this.touchSessionAccess(
      promptRows.map(r => r.id),
      responseRows.map(r => r.id),
      toolCallRows.map(r => r.id),
      summaryRows.map(r => r.id),
    );

    return {
      sessionId,
      prompts: promptRows.map(rowToSessionPrompt),
      responses: responseRows.map(rowToSessionResponse),
      toolCalls: toolCallRows.map(rowToSessionToolCall),
      summaries: summaryRows.map(rowToSessionSummary),
    };
  }

  async listSessions(limit?: number): Promise<SessionMeta[]> {
    const limitClause = limit !== undefined ? ' LIMIT ?' : '';
    const limitArgs = limit !== undefined ? [limit] : [];

    // Derive distinct sessions across all four ledger tables, with the most
    // recent activity timestamp. Join against session_prompts to compute the
    // per-session prompt count (SessionMeta.promptCount).
    const rows = this.db.prepare(
      `SELECT
         activity.session_id AS session_id,
         COALESCE(p.prompt_count, 0) AS prompt_count,
         activity.last_activity AS last_activity
       FROM (
         SELECT session_id, MAX(created_at) AS last_activity
         FROM (
           SELECT session_id, created_at FROM session_prompts
           UNION ALL
           SELECT session_id, created_at FROM session_responses
           UNION ALL
           SELECT session_id, created_at FROM session_tool_calls
           UNION ALL
           SELECT session_id, created_at FROM session_summaries
         )
         GROUP BY session_id
       ) activity
       LEFT JOIN (
         SELECT session_id, COUNT(*) AS prompt_count
         FROM session_prompts
         GROUP BY session_id
       ) p ON p.session_id = activity.session_id
       ORDER BY activity.last_activity DESC${limitClause}`,
    ).all(...limitArgs) as Array<{ session_id: string; prompt_count: number; last_activity: string }>;

    return rows.map(r => ({
      sessionId: r.session_id,
      promptCount: r.prompt_count,
      lastActivity: r.last_activity,
    }));
  }

  async linkPromptToMemory(
    promptId: string,
    memoryId: string,
    relation: string,
    responseId?: string,
  ): Promise<void> {
    const now = new Date().toISOString();
    const rid = responseId ?? null;
    // Use INSERT ... WHERE NOT EXISTS with NULL-safe IS comparison so that
    // repeated calls (including with a NULL responseId) do not create
    // duplicate rows — SQLite treats NULL ≠ NULL inside a PRIMARY KEY, so a
    // naive INSERT OR IGNORE would not dedup NULL response_id links.
    this.db.prepare(
      `INSERT INTO session_memory_links (prompt_id, response_id, memory_id, relation, created_at)
       SELECT ?, ?, ?, ?, ?
       WHERE NOT EXISTS (
         SELECT 1 FROM session_memory_links
         WHERE prompt_id = ? AND memory_id = ? AND relation = ? AND response_id IS ?
       )`,
    ).run(
      promptId, rid, memoryId, relation, now,
      promptId, memoryId, relation, rid,
    );
  }

  // ── Session decay: forget / purge by id ─────────────────────────────

  /**
   * Detect which session-ledger table holds a given id. Returns the entity
   * type ('prompt' | 'response' | 'tool_call' | 'summary') plus the table
   * name, or null when the id matches no session entity.
   *
   * Throws when the same id appears in more than one table (ambiguous): the
   * caller must refuse to mutate because the target row cannot be uniquely
   * identified. In practice ids are randomUUID() so collisions are
   * astronomically unlikely, but the guard keeps the contract safe.
   */
  private detectSessionEntity(id: string): { type: SessionItemType; table: string } {
    const checks: Array<{ type: SessionItemType; table: string }> = [
      { type: 'prompt', table: 'session_prompts' },
      { type: 'response', table: 'session_responses' },
      { type: 'tool_call', table: 'session_tool_calls' },
      { type: 'summary', table: 'session_summaries' },
    ];
    const hits: Array<{ type: SessionItemType; table: string }> = [];
    for (const c of checks) {
      const row = this.db.prepare(`SELECT 1 AS ok FROM ${c.table} WHERE id = ?`).get(id);
      if (row) hits.push(c);
    }
    if (hits.length === 0) {
      throw new Error(`NOT_FOUND: no session entity with id ${id}`);
    }
    if (hits.length > 1) {
      throw new Error(
        `AMBIGUOUS: id ${id} matches multiple session tables (${hits.map(h => h.type).join(', ')}); refusing to delete`,
      );
    }
    return hits[0];
  }

  /**
   * Soft-forget a session-ledger record by id (sets status='forgotten').
   * Spec (session-forget.md): returns { type, success }. Forgotten rows are
   * hidden from default getSession/search retrieval but remain recoverable
   * until explicitly purged.
   */
  async forgetSessionItem(id: string): Promise<SessionForgetResult> {
    const { type } = this.detectSessionEntity(id);
    const tables: Record<SessionItemType, string> = {
      prompt: 'session_prompts',
      response: 'session_responses',
      tool_call: 'session_tool_calls',
      summary: 'session_summaries',
    };
    const now = new Date().toISOString();
    const res = this.db.prepare(
      `UPDATE ${tables[type]} SET status = 'forgotten', updated_at = ? WHERE id = ?`,
    ).run(now, id) as { changes: number };
    return { type, success: res.changes > 0 };
  }

  /**
   * Hard-purge a session-ledger record by id — permanently deletes the row
   * (including its embedding column). Spec (session-forget.md): explicit
   * destructive path; never runs automatically through decay/maintenance.
   * Returns { type, success }.
   *
   * Inter-table references between session rows are non-cascading by design,
   * so to honour "delete only the intended item" the explicit destructive
   * path relaxes FK enforcement for this single DELETE. Dependents are left
   * in place (they become orphans rather than being cascade-removed).
   */
  async purgeSessionItem(id: string): Promise<SessionPurgeResult> {
    const { type, table } = this.detectSessionEntity(id);
    const fkWasOn =
      (this.db.prepare('PRAGMA foreign_keys').get() as { foreign_keys: number }).foreign_keys === 1;
    if (fkWasOn) this.db.exec('PRAGMA foreign_keys = OFF');
    try {
      const res = this.db.prepare(`DELETE FROM ${table} WHERE id = ?`).run(id) as { changes: number };
      return { type, success: res.changes > 0 };
    } finally {
      if (fkWasOn) this.db.exec('PRAGMA foreign_keys = ON');
    }
  }

  // ── Symbol-linked memory context (v7) ──────────────────────────────

  /**
   * Attach a symbol key to an existing memory or session-ledger record.
   * Upserts — overwrites the previous symbol_key on the same record.
   * Throws MemoryNotFoundError if the target record does not exist.
   */
  async attachSymbol(opts: { symbolKey: string; memoryId?: string; sessionItemId?: string }): Promise<{ id: string; symbolKey: string }> {
    const { symbolKey, memoryId, sessionItemId } = opts;

    if (memoryId) {
      // Verify the memory exists.
      const mem = this.db.prepare('SELECT id FROM memories WHERE id = ?').get(memoryId) as { id: string } | undefined;
      if (!mem) throw new MemoryNotFoundError(memoryId);
      this.db.prepare('UPDATE memories SET symbol_key = ? WHERE id = ?').run(symbolKey, memoryId);
      return { id: memoryId, symbolKey };
    }

    if (sessionItemId) {
      // Detect which session table the id belongs to.
      const { table } = this.detectSessionEntity(sessionItemId);
      this.db.prepare(`UPDATE ${table} SET symbol_key = ? WHERE id = ?`).run(symbolKey, sessionItemId);
      return { id: sessionItemId, symbolKey };
    }

    throw new Error('Either memoryId or sessionItemId must be provided');
  }

  /**
   * Return all memories and session items linked to a symbol key.
   * Sources default to ['memory', 'session_summary']. Soft-miss: returns
   * empty array for unknown keys (never throws).
   */
  async getBySymbol(symbolKey: string, opts?: { sources?: string[]; limit?: number }): Promise<import('./types.js').SymbolLinkedResult[]> {
    const sources = opts?.sources ?? ['memory', 'session_summary'];
    const limit = opts?.limit ?? 50;
    const results: import('./types.js').SymbolLinkedResult[] = [];

    if (sources.includes('memory')) {
      const rows = this.db.prepare(
        `SELECT id, content, created_at FROM memories WHERE symbol_key = ? LIMIT ?`,
      ).all(symbolKey, limit) as Array<{ id: string; content: string; created_at: string }>;
      for (const row of rows) {
        results.push({
          id: row.id,
          source: 'memory',
          symbolKey,
          content: row.content,
          createdAt: row.created_at,
        });
      }
    }

    const sessionSourceMap: Record<string, { table: string; contentCol: string }> = {
      session_summary: { table: 'session_summaries', contentCol: 'summary' },
      session_prompt: { table: 'session_prompts', contentCol: 'raw_prompt' },
      session_response: { table: 'session_responses', contentCol: 'full_response' },
      session_tool_call: { table: 'session_tool_calls', contentCol: 'result_summary' },
    };

    for (const source of sources) {
      const meta = sessionSourceMap[source];
      if (!meta) continue;
      const rows = this.db.prepare(
        `SELECT id, ${meta.contentCol} as content, created_at FROM ${meta.table} WHERE symbol_key = ? LIMIT ?`,
      ).all(symbolKey, limit) as Array<{ id: string; content: string; created_at: string }>;
      for (const row of rows) {
        results.push({
          id: row.id,
          source: source as import('./types.js').SymbolLinkedResult['source'],
          symbolKey,
          content: row.content ?? '',
          createdAt: row.created_at,
        });
      }
    }

    return results.slice(0, limit);
  }

  // ── Pin / Unpin (v8) ───────────────────────────────────────────────

  async pin(id: string): Promise<Memory> {
    const existing = this.db.prepare(
      'SELECT id FROM memories WHERE id = ?',
    ).get(id) as { id: string } | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    this.db.prepare(
      'UPDATE memories SET is_pinned = 1 WHERE id = ?',
    ).run(id);

    const row = this.db.prepare('SELECT * FROM memories WHERE id = ?').get(id) as unknown as MemoryRow;
    return rowToMemory(row);
  }

  async unpin(id: string): Promise<Memory> {
    const existing = this.db.prepare(
      'SELECT id FROM memories WHERE id = ?',
    ).get(id) as { id: string } | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    this.db.prepare(
      'UPDATE memories SET is_pinned = 0 WHERE id = ?',
    ).run(id);

    const row = this.db.prepare('SELECT * FROM memories WHERE id = ?').get(id) as unknown as MemoryRow;
    return rowToMemory(row);
  }

  // ── Auto-link helper (T-004 / PR-2) ──────────────────────────────────

  /**
   * Fire-and-forget: when a session context is active, link the just-stored /
   * just-updated memory to the most recent prompt (and its most recent
   * response, if any) in that session. Never throws — callers (store/update)
   * must not fail because of auto-linking.
   */
  private autoLinkToActiveSession(memoryId: string): void {
    const ctx = this.sessionContext;
    if (!ctx) return;

    try {
      // ORDER BY created_at DESC, rowid DESC: deterministic tiebreaker when
      // multiple prompts share a millisecond timestamp (MockEmbedder is sub-ms).
      // rowid is monotonic with insertion order, so the last-inserted row wins.
      const prompt = this.db.prepare(
        `SELECT id FROM session_prompts WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1`,
      ).get(ctx.sessionId) as { id: string } | undefined;

      if (!prompt) return; // nothing to link to

      const response = this.db.prepare(
        `SELECT id FROM session_responses WHERE prompt_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1`,
      ).get(prompt.id) as { id: string } | undefined;

      // linkPromptToMemory is sync internally; call it directly (not via the
      // async interface wrapper) since we are already inside a sync helper.
      const now = new Date().toISOString();
      const rid = response?.id ?? null;
      this.db.prepare(
        `INSERT INTO session_memory_links (prompt_id, response_id, memory_id, relation, created_at)
         SELECT ?, ?, ?, ?, ?
         WHERE NOT EXISTS (
           SELECT 1 FROM session_memory_links
           WHERE prompt_id = ? AND memory_id = ? AND relation = ? AND response_id IS ?
         )`,
      ).run(
        prompt.id, rid, memoryId, 'created_from', now,
        prompt.id, memoryId, 'created_from', rid,
      );
    } catch {
      // Best-effort: swallow so store()/update() callers are unaffected.
    }
  }
}
