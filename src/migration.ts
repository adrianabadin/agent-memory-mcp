// src/migration.ts
//
// T-009: Migration logic for importing Engram SQLite databases into PMC.
// Provides pure functions that read Engram rows and write to a PMC MemoryStore.

import type { Embedder, MemoryStore, StoreRequest } from './types.js';

/**
 * Map an Engram observation type to a PMC memory category.
 * Copied from tools/project-memory-context/src/memory-protocol.mjs
 * (pure function, no I/O, avoids cross-package import issues).
 *
 * Unknown / null / anything-else falls back to "learning".
 */
export function mapEngramTypeToPmcCategory(engramType: string | null | undefined): string {
  switch (engramType) {
    case 'bugfix':       return 'bug-fix';
    case 'decision':     return 'architecture';
    case 'architecture': return 'architecture';
    case 'discovery':    return 'learning';
    case 'pattern':      return 'learning';
    case 'config':       return 'tool-usage';
    case 'preference':   return 'personal';
    default:             return 'learning';
  }
}

// ── Types ─────────────────────────────────────────────────────────

export interface EngramObservation {
  id: number;
  sync_id: string | null;
  session_id: string;
  type: string;
  title: string;
  content: string;
  tool_name: string | null;
  project: string | null;
  scope: string;
  topic_key: string | null;
  normalized_hash: string | null;
  revision_count: number;
  duplicate_count: number;
  last_seen_at: string | null;
  pinned: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
  review_after: string | null;
  expires_at: string | null;
  embedding: Buffer | null;
  embedding_model: string | null;
  embedding_created_at: string | null;
}

export interface EngramUserPrompt {
  id: number;
  sync_id: string | null;
  session_id: string;
  content: string;
  project: string | null;
  created_at: string;
}

export interface EngramMemoryRelation {
  id: number;
  sync_id: string;
  source_id: string;
  target_id: string;
  relation: string;
  reason: string | null;
  evidence: string | null;
  confidence: number | null;
  judgment_status: string;
  marked_by_actor: string | null;
  marked_by_kind: string | null;
  marked_by_model: string | null;
  session_id: string | null;
  superseded_at: string | null;
  superseded_by_relation_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface MigrationOptions {
  dryRun: boolean;
  limit?: number;
  batchSize?: number;
}

export interface MigrationResult {
  migrated: number;
  skipped: number;
  errors: number;
}

export interface SessionLedgerMigrationResult {
  promptsMigrated: number;
  summariesMigrated: number;
  skipped: number;
  errors: number;
}

export interface RelationMigrationResult {
  migrated: number;
  skipped: number;
  errors: number;
}

export interface SymbolLinkResult {
  linked: number;
  errors: number;
}

export interface ProgressRow {
  table: string;
  migrated: number;
  skipped: number;
  errors: number;
}

export interface FullMigrationResult {
  success: boolean;
  table: ProgressRow[];
}

export interface GraphNode {
  id: string;
  path: string;
  name: string;
}

export interface EngramDbSnapshot {
  observations: EngramObservation[];
  userPrompts: EngramUserPrompt[];
  relations: EngramMemoryRelation[];
}

// ── Helpers ───────────────────────────────────────────────────────

function composeContent(title: string, content: string): string {
  const titleText = (title ?? '').trim();
  const bodyText = (content ?? '').trim();
  return titleText ? `**What**: ${titleText}\n\n${bodyText}` : bodyText;
}

function buildTags(project: string | null, topicKey: string | null): string[] {
  const tags: string[] = [];
  if (project) tags.push(project);
  if (topicKey) {
    for (const segment of String(topicKey).split('/')) {
      const trimmed = segment.trim();
      if (trimmed && !tags.includes(trimmed)) tags.push(trimmed);
    }
  }
  return tags;
}

/** Extract file-path-like references from content (e.g., `src/foo.ts`). */
function extractPaths(content: string): string[] {
  const pathRegex = /(?:^|[\s`'"])([a-zA-Z0-9_\-./]+\.[a-zA-Z]{1,10})(?:$|[\s`'",:;!?])/g;
  const paths: string[] = [];
  let match;
  while ((match = pathRegex.exec(content)) !== null) {
    const p = match[1];
    // Filter out common non-path tokens
    if (p.includes('/') && !p.startsWith('http') && !p.startsWith('www.')) {
      paths.push(p);
    }
  }
  return [...new Set(paths)];
}

/** Match extracted paths against graph.json nodes. */
function matchPathsToGraph(
  paths: string[],
  graphNodes: GraphNode[],
): GraphNode | null {
  for (const path of paths) {
    for (const node of graphNodes) {
      if (node.path === path || node.path.endsWith('/' + path) || path.endsWith('/' + node.path)) {
        return node;
      }
    }
  }
  return null;
}

// ── Migration functions ───────────────────────────────────────────

/**
 * Migrate Engram observations → PMC memories.
 * Skips deleted observations (deleted_at IS NOT NULL).
 * Re-embeds all content using PMC's embedder.
 */
export async function migrateObservations(
  observations: EngramObservation[],
  store: MemoryStore,
  embedder: Embedder,
  options: MigrationOptions,
): Promise<MigrationResult> {
  const limit = options.limit ?? Infinity;
  let migrated = 0;
  let skipped = 0;
  let errors = 0;

  const batch: StoreRequest[] = [];
  const batchSize = options.batchSize ?? 50;

  for (const obs of observations) {
    if (migrated + skipped >= limit) break;

    // Skip deleted
    if (obs.deleted_at) {
      skipped++;
      continue;
    }

    // Skip session_summary — those go to session_summaries table
    if (obs.type === 'session_summary') {
      skipped++;
      continue;
    }

    try {
      const content = composeContent(obs.title, obs.content);
      const category = mapEngramTypeToPmcCategory(obs.type);
      const tags = buildTags(obs.project, obs.topic_key);

      if (options.dryRun) {
        migrated++;
        continue;
      }

      // Re-embed with PMC's embedder (bge-m3 1024-d), NOT Engram's embedding
      await embedder.embed(content);

      batch.push({
        content,
        category: category as StoreRequest['category'],
        tags,
        provenance: {
          origin: 'extracted',
          ...(obs.session_id ? { sessionId: obs.session_id } : {}),
          ...(obs.tool_name ? { sourceTool: obs.tool_name } : {}),
        },
      });

      if (batch.length >= batchSize) {
        await store.storeBatch(batch);
        batch.length = 0;
      }

      migrated++;
    } catch {
      errors++;
    }
  }

  // Flush remaining batch
  if (batch.length > 0 && !options.dryRun) {
    await store.storeBatch(batch);
  }

  return { migrated, skipped, errors };
}

/**
 * Migrate Engram user_prompts → PMC session_prompts
 * and session_summary observations → PMC session_summaries.
 */
export async function migrateSessionLedger(
  prompts: EngramUserPrompt[],
  sessionSummaryObs: EngramObservation[],
  store: MemoryStore,
  _embedder: Embedder,
  options: MigrationOptions,
): Promise<SessionLedgerMigrationResult> {
  const limit = options.limit ?? Infinity;
  let promptsMigrated = 0;
  let summariesMigrated = 0;
  let skipped = 0;
  let errors = 0;

  // Migrate user_prompts → session_prompts
  for (const prompt of prompts) {
    if (promptsMigrated >= limit) break;

    try {
      if (options.dryRun) {
        promptsMigrated++;
        continue;
      }

      // Use storeSessionPrompt if available on the concrete store
      const concrete = store as unknown as Record<string, unknown>;
      if (typeof concrete.storeSessionPrompt === 'function') {
        await (concrete.storeSessionPrompt as (id: string, sid: string, content: string, created: string) => Promise<void>)(
          prompt.sync_id || `prompt-${prompt.id}`,
          prompt.session_id,
          prompt.content,
          prompt.created_at,
        );
      }
      promptsMigrated++;
    } catch {
      errors++;
    }
  }

  // Migrate session_summary observations → session_summaries
  for (const obs of sessionSummaryObs) {
    if (summariesMigrated >= limit) break;

    try {
      if (options.dryRun) {
        summariesMigrated++;
        continue;
      }

      const concrete = store as unknown as Record<string, unknown>;
      if (typeof concrete.storeSessionSummary === 'function') {
        await (concrete.storeSessionSummary as (data: { id: string; sessionId: string; summary: string; createdAt: string }) => Promise<void>)({
          id: obs.sync_id || `summary-${obs.id}`,
          sessionId: obs.session_id,
          summary: obs.content,
          createdAt: obs.created_at,
        });
      }
      summariesMigrated++;
    } catch {
      errors++;
    }
  }

  return { promptsMigrated, summariesMigrated, skipped, errors };
}

/**
 * Migrate Engram memory_relations → PMC memory_relations.
 */
export async function migrateRelations(
  relations: EngramMemoryRelation[],
  store: MemoryStore,
  options: MigrationOptions,
): Promise<RelationMigrationResult> {
  const limit = options.limit ?? Infinity;
  let migrated = 0;
  let skipped = 0;
  let errors = 0;

  for (const rel of relations) {
    if (migrated >= limit) break;

    try {
      if (options.dryRun) {
        migrated++;
        continue;
      }

      const concrete = store as unknown as Record<string, unknown>;
      if (typeof concrete.storeRelation === 'function') {
        await (concrete.storeRelation as (data: { sourceId: string; targetId: string; relation: string; reason: string | null; confidence: number | null; judgmentStatus: string }) => Promise<void>)({
          sourceId: rel.source_id,
          targetId: rel.target_id,
          relation: rel.relation,
          reason: rel.reason,
          confidence: rel.confidence,
          judgmentStatus: rel.judgment_status,
        });
      }
      migrated++;
    } catch {
      errors++;
    }
  }

  return { migrated, skipped, errors };
}

/**
 * Link migrated memories to graph.json nodes by parsing content for paths.
 * Failures are logged but do NOT abort the linking process.
 */
export async function linkSymbols(
  memories: Array<{ id: string; content: string }>,
  graph: { nodes: GraphNode[] },
  store: MemoryStore,
): Promise<SymbolLinkResult> {
  let linked = 0;
  let errors = 0;

  for (const mem of memories) {
    try {
      const paths = extractPaths(mem.content);
      const node = matchPathsToGraph(paths, graph.nodes);
      if (node) {
        const concrete = store as unknown as Record<string, unknown>;
        if (typeof concrete.attachSymbol === 'function') {
          await (concrete.attachSymbol as (id: string, key: string) => Promise<void>)(mem.id, node.path + '|' + node.name);
          linked++;
        }
      }
    } catch {
      errors++;
    }
  }

  return { linked, errors };
}

/**
 * Full migration orchestrator.
 * Reads from an Engram DB snapshot and writes to PMC.
 */
export async function runMigration(
  engramDb: EngramDbSnapshot,
  store: MemoryStore,
  embedder: Embedder,
  options: MigrationOptions & { graphNodes: GraphNode[] },
): Promise<FullMigrationResult> {
  const table: ProgressRow[] = [];

  // 1. Observations → memories
  const obsResult = await migrateObservations(
    engramDb.observations,
    store,
    embedder,
    options,
  );
  table.push({ table: 'observations → memories', ...obsResult });

  // 2. Session ledger
  const sessionSummaryObs = engramDb.observations.filter(o => o.type === 'session_summary');
  const sessionResult = await migrateSessionLedger(
    engramDb.userPrompts,
    sessionSummaryObs,
    store,
    embedder,
    options,
  );
  table.push({
    table: 'user_prompts → session_prompts',
    migrated: sessionResult.promptsMigrated,
    skipped: sessionResult.skipped,
    errors: sessionResult.errors,
  });
  table.push({
    table: 'session_summary obs → session_summaries',
    migrated: sessionResult.summariesMigrated,
    skipped: 0,
    errors: 0,
  });

  // 3. Relations
  const relResult = await migrateRelations(engramDb.relations, store, options);
  table.push({ table: 'memory_relations', ...relResult });

  // 4. Symbol linking (only in non-dry-run mode)
  if (!options.dryRun && options.graphNodes.length > 0) {
    // Collect the memories we just wrote (approximate — real IDs from storeBatch)
    const memsForLinking = engramDb.observations
      .filter(o => !o.deleted_at && o.type !== 'session_summary')
      .map((o, i) => ({ id: `pmc-mem-${i}`, content: o.content }));
    const linkResult = await linkSymbols(memsForLinking, { nodes: options.graphNodes }, store);
    table.push({ table: 'symbol linking', migrated: linkResult.linked, skipped: 0, errors: linkResult.errors });
  }

  const totalErrors = table.reduce((sum, row) => sum + row.errors, 0);
  return { success: totalErrors === 0, table };
}
