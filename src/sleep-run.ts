// agent-memory-mcp/src/sleep-run.ts
//
// Sleep-mode one-shot maintenance pipeline. Runs decay evaluation, promotion
// passes, and symbol enrichment in a single atomic run with checkpoint/restore
// capability. Designed for background execution during idle periods.
//
// Pipeline phases:
//   1. keep-awake lease acquire
//   2. refresh context (per-project if multi-project)
//   3. decay evaluation (evaluateAllMemoriesForDecay)
//   4. enrich phase (local-only enforcement, symbol-boundary pause/yield)
//   5. promotion pass (PromotionRuleEngine for eligible memories)
//   6. sync context (per-project if multi-project)
//   7. checkpoint report persistence
//   8. keep-awake lease release

import { randomUUID } from 'node:crypto';
import { writeFile, readFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { DatabaseSync } from 'node:sqlite';
import type { SqliteMemoryStore } from './sqlite-store.js';
import { PromotionRuleEngine } from './global/global-promotion.js';
import type { PromotionResult } from './global/global-promotion.js';
import { DEFAULT_FORGETTING_CONFIG } from './decay.js';
import type { ForgettingConfig } from './decay.js';

// ── Enrichment provider types ───────────────────────────────────────

/** Enrichment provider identifier. Only 'local-model' is allowed for sleep runs. */
export type EnrichmentProviderId = 'local-model';

/** Result of an enrichment attempt. */
export interface EnrichResult {
  enriched: number;
  deferred: number;
  provider: EnrichmentProviderId;
  reason?: string;
}

/** Enrichment provider function signature. */
export type EnrichmentProvider = (
  symbols: string[],
  provider: EnrichmentProviderId,
) => Promise<EnrichResult>;

// ── Report JSON Schema ──────────────────────────────────────────────

export interface SleepRunPhaseResult {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startedAt: string | null;
  completedAt: string | null;
  result?: unknown;
  error?: string;
}

export interface SleepRunSummary {
  decayEvaluated: number;
  decayTransitioned: number;
  promotionEvaluated: number;
  promotionPromoted: number;
  symbolsEnriched: number;
  symbolsDeferred: number;
  /** Number of projects processed (multi-project mode). */
  projectsProcessed: number;
  /** Whether the run was paused due to activity detection. */
  pausedByActivity: boolean;
}

export interface SleepRunReport {
  runId: string;
  startedAt: string;
  completedAt: string | null;
  status: 'running' | 'completed' | 'failed' | 'paused';
  phases: SleepRunPhaseResult[];
  summary: SleepRunSummary;
  errors: string[];
  /** Multi-project: list of project IDs processed in this run. */
  projects?: string[];
}

// ── Keep-Awake Lease ────────────────────────────────────────────────

export interface KeepAwakeLease {
  acquiredAt: string;
  expiresAt: string;
  holder: string;
}

export interface KeepAwakeLeaseManager {
  acquire(holder: string, ttlMs?: number): boolean;
  renew(holder: string, ttlMs?: number): boolean;
  release(holder: string): boolean;
  isActive(): boolean;
  getLease(): KeepAwakeLease | null;
}

// ── Sleep Run Pipeline ──────────────────────────────────────────────

export interface SleepRunConfig {
  /** Path to persist the checkpoint report JSON. */
  checkpointPath: string;
  /** Forgetting config for decay evaluation. Defaults to DEFAULT_FORGETTING_CONFIG. */
  forgettingConfig?: ForgettingConfig;
  /** TTL for keep-awake lease in ms. Defaults to 300_000 (5 minutes). */
  leaseTTLMs?: number;
  /** Holder identifier for the keep-awake lease. */
  leaseHolder?: string;
  /** Maximum memories to evaluate for promotion per run. Defaults to 100. */
  promotionBatchSize?: number;
  /** Multi-project: list of project IDs to iterate over. If empty, runs single-project. */
  projectIds?: string[];
  /** Enrichment provider ID. Must be 'local-model' for sleep runs. */
  enrichmentProvider?: EnrichmentProviderId;
}

export interface SleepRunDeps {
  store: SqliteMemoryStore;
  db: DatabaseSync;
  /** Enrichment provider. Required for enrich phase; enforces local-only. */
  enrichmentProvider?: EnrichmentProvider;
  /** Optional external refresh-context callback. */
  refreshContext?: () => Promise<void>;
  /** Optional external sync-context callback. */
  syncContext?: () => Promise<void>;
  /**
   * Activity detector: returns true if user activity or CPU pressure is detected
   * and the run should pause at the next safe boundary. Checked after each symbol
   * during the enrich phase.
   */
  shouldPause?: () => boolean;
}

// ── In-memory lease manager ─────────────────────────────────────────

export class InMemoryKeepAwakeLeaseManager implements KeepAwakeLeaseManager {
  private lease: KeepAwakeLease | null = null;

  acquire(holder: string, ttlMs = 300_000): boolean {
    if (this.lease && new Date(this.lease.expiresAt).getTime() > Date.now()) {
      return false; // active lease exists
    }
    const now = new Date();
    this.lease = {
      acquiredAt: now.toISOString(),
      expiresAt: new Date(now.getTime() + ttlMs).toISOString(),
      holder,
    };
    return true;
  }

  renew(holder: string, ttlMs = 300_000): boolean {
    if (!this.lease || this.lease.holder !== holder) {
      return false;
    }
    if (new Date(this.lease.expiresAt).getTime() <= Date.now()) {
      return false; // lease expired
    }
    const now = new Date();
    this.lease.expiresAt = new Date(now.getTime() + ttlMs).toISOString();
    return true;
  }

  release(holder: string): boolean {
    if (!this.lease || this.lease.holder !== holder) {
      return false;
    }
    this.lease = null;
    return true;
  }

  isActive(): boolean {
    return this.lease !== null && new Date(this.lease.expiresAt).getTime() > Date.now();
  }

  getLease(): KeepAwakeLease | null {
    if (!this.lease) return null;
    if (new Date(this.lease.expiresAt).getTime() <= Date.now()) {
      this.lease = null;
      return null;
    }
    return { ...this.lease };
  }
}

// ── Report builder ──────────────────────────────────────────────────

function createReport(runId: string): SleepRunReport {
  return {
    runId,
    startedAt: new Date().toISOString(),
    completedAt: null,
    status: 'running',
    phases: [
      { name: 'lease_acquire', status: 'pending', startedAt: null, completedAt: null },
      { name: 'refresh_context', status: 'pending', startedAt: null, completedAt: null },
      { name: 'decay_evaluation', status: 'pending', startedAt: null, completedAt: null },
      { name: 'enrich', status: 'pending', startedAt: null, completedAt: null },
      { name: 'promotion_pass', status: 'pending', startedAt: null, completedAt: null },
      { name: 'sync_context', status: 'pending', startedAt: null, completedAt: null },
      { name: 'lease_release', status: 'pending', startedAt: null, completedAt: null },
    ],
    summary: {
      decayEvaluated: 0,
      decayTransitioned: 0,
      promotionEvaluated: 0,
      promotionPromoted: 0,
      symbolsEnriched: 0,
      symbolsDeferred: 0,
      projectsProcessed: 0,
      pausedByActivity: false,
    },
    errors: [],
  };
}

function startPhase(report: SleepRunReport, phaseName: string): void {
  const phase = report.phases.find(p => p.name === phaseName);
  if (phase) {
    phase.status = 'running';
    phase.startedAt = new Date().toISOString();
  }
}

function completePhase(report: SleepRunReport, phaseName: string, result?: unknown): void {
  const phase = report.phases.find(p => p.name === phaseName);
  if (phase) {
    phase.status = 'completed';
    phase.completedAt = new Date().toISOString();
    if (result !== undefined) phase.result = result;
  }
}

function failPhase(report: SleepRunReport, phaseName: string, error: string): void {
  const phase = report.phases.find(p => p.name === phaseName);
  if (phase) {
    phase.status = 'failed';
    phase.completedAt = new Date().toISOString();
    phase.error = error;
  }
  report.errors.push(`[${phaseName}] ${error}`);
}

// ── One-shot pipeline ───────────────────────────────────────────────

/**
 * Execute a single sleep-run maintenance pipeline. Returns the completed
 * report with all phase outcomes and summary statistics.
 *
 * The pipeline is idempotent: re-running with the same database state
 * produces the same decay/promotion results (no double-processing).
 *
 * Supports multi-project iteration: when config.projectIds is non-empty,
 * the pipeline iterates over each project running refresh/enrich/sync per project.
 */
export async function executeSleepRun(
  deps: SleepRunDeps,
  config: SleepRunConfig,
): Promise<SleepRunReport> {
  const runId = randomUUID();
  const report = createReport(runId);
  const leaseManager = new InMemoryKeepAwakeLeaseManager();
  const leaseHolder = config.leaseHolder ?? `sleep-run-${runId}`;
  const leaseTTL = config.leaseTTLMs ?? 300_000;
  const promotionBatchSize = config.promotionBatchSize ?? 100;
  const projectIds = config.projectIds ?? [];

  // Track multi-project IDs in report
  if (projectIds.length > 0) {
    report.projects = projectIds;
  }

  try {
    // Phase 1: Acquire keep-awake lease
    startPhase(report, 'lease_acquire');
    const acquired = leaseManager.acquire(leaseHolder, leaseTTL);
    if (!acquired) {
      failPhase(report, 'lease_acquire', 'Could not acquire keep-awake lease');
      report.status = 'failed';
      report.completedAt = new Date().toISOString();
      return report;
    }
    completePhase(report, 'lease_acquire', { holder: leaseHolder, ttlMs: leaseTTL });

    // Phase 2: Refresh context (per-project if multi-project)
    startPhase(report, 'refresh_context');
    if (deps.refreshContext) {
      try {
        if (projectIds.length > 0) {
          // Multi-project: refresh each project
          for (const _projectId of projectIds) {
            await deps.refreshContext();
            report.summary.projectsProcessed++;
          }
        } else {
          await deps.refreshContext();
        }
        completePhase(report, 'refresh_context');
      } catch (e) {
        failPhase(report, 'refresh_context', String(e));
        // Non-fatal: continue pipeline
      }
    } else {
      completePhase(report, 'refresh_context', { skipped: true, reason: 'no refreshContext callback' });
    }

    // Phase 3: Decay evaluation
    startPhase(report, 'decay_evaluation');
    try {
      const decayResult = deps.store.evaluateAllMemoriesForDecay(
        config.forgettingConfig ?? DEFAULT_FORGETTING_CONFIG,
      );
      report.summary.decayEvaluated = decayResult.processed;
      report.summary.decayTransitioned = decayResult.transitioned;
      completePhase(report, 'decay_evaluation', {
        processed: decayResult.processed,
        transitioned: decayResult.transitioned,
      });
    } catch (e) {
      failPhase(report, 'decay_evaluation', String(e));
      // Decay failure is fatal for the pipeline
      report.status = 'failed';
      report.completedAt = new Date().toISOString();
      return report;
    }

    // Phase 4: Enrich (local-only enforcement, symbol-boundary pause/yield)
    startPhase(report, 'enrich');
    const enrichResult = await runEnrichPhase(deps, config, report);
    if (enrichResult.paused) {
      // Activity detected — flush checkpoint and return paused
      completePhase(report, 'enrich', { ...enrichResult });
      startPhase(report, 'lease_release');
      leaseManager.release(leaseHolder);
      completePhase(report, 'lease_release');
      report.status = 'paused';
      report.completedAt = new Date().toISOString();
      report.summary.pausedByActivity = true;
      try { await persistReport(config.checkpointPath, report); } catch { /* non-fatal */ }
      return report;
    }

    // Phase 5: Promotion pass
    startPhase(report, 'promotion_pass');
    try {
      const promotionResult = runPromotionPass(deps.db, promotionBatchSize);
      report.summary.promotionEvaluated = promotionResult.evaluated;
      report.summary.promotionPromoted = promotionResult.promoted;
      completePhase(report, 'promotion_pass', promotionResult);
    } catch (e) {
      failPhase(report, 'promotion_pass', String(e));
      // Non-fatal: continue pipeline
    }

    // Phase 6: Sync context (per-project if multi-project)
    startPhase(report, 'sync_context');
    if (deps.syncContext) {
      try {
        if (projectIds.length > 0) {
          for (const _projectId of projectIds) {
            await deps.syncContext();
          }
        } else {
          await deps.syncContext();
        }
        completePhase(report, 'sync_context');
      } catch (e) {
        failPhase(report, 'sync_context', String(e));
        // Non-fatal: continue pipeline
      }
    } else {
      completePhase(report, 'sync_context', { skipped: true, reason: 'no syncContext callback' });
    }

    // Phase 7: Release keep-awake lease
    startPhase(report, 'lease_release');
    leaseManager.release(leaseHolder);
    completePhase(report, 'lease_release');

    report.status = 'completed';
    report.completedAt = new Date().toISOString();
  } catch (e) {
    report.errors.push(`[unexpected] ${String(e)}`);
    report.status = 'failed';
    report.completedAt = new Date().toISOString();
    // Best-effort lease release
    leaseManager.release(leaseHolder);
  }

  // Persist checkpoint report
  try {
    await persistReport(config.checkpointPath, report);
  } catch {
    // Checkpoint persistence failure is non-fatal
  }

  return report;
}

// ── Enrich phase with local-only enforcement ────────────────────────

interface EnrichPhaseResult {
  enriched: number;
  deferred: number;
  paused: boolean;
  reason?: string;
}

/**
 * Run the enrich phase with local-only enforcement and symbol-boundary pause/yield.
 *
 * - Validates that the enrichment provider is 'local-model'
 * - If provider unavailable, marks deferred with local_model_unavailable
 * - Processes symbols one at a time, checking shouldPause after each
 * - Flushes checkpoint at safe boundaries when pause detected
 */
async function runEnrichPhase(
  deps: SleepRunDeps,
  config: SleepRunConfig,
  report: SleepRunReport,
): Promise<EnrichPhaseResult> {
  const providerId = config.enrichmentProvider ?? 'local-model';

  // Local-only enforcement: only 'local-model' provider is allowed
  if (providerId !== 'local-model') {
    report.summary.symbolsDeferred++;
    completePhase(report, 'enrich', {
      skipped: true,
      reason: `provider '${providerId}' not allowed; only 'local-model' permitted`,
      local_model_unavailable: true,
    });
    return { enriched: 0, deferred: 1, paused: false, reason: 'local_model_unavailable' };
  }

  if (!deps.enrichmentProvider) {
    report.summary.symbolsDeferred++;
    completePhase(report, 'enrich', {
      skipped: true,
      reason: 'no enrichment provider configured',
      local_model_unavailable: true,
    });
    return { enriched: 0, deferred: 1, paused: false, reason: 'local_model_unavailable' };
  }

  // Fetch symbols to enrich (non-purged memories with content)
  const symbols = deps.db.prepare(
    `SELECT id FROM memories WHERE memory_state != 'purged' ORDER BY updated_at DESC LIMIT 500`,
  ).all() as Array<{ id: string }>;

  if (symbols.length === 0) {
    completePhase(report, 'enrich', { skipped: true, reason: 'no symbols to enrich' });
    return { enriched: 0, deferred: 0, paused: false };
  }

  let enriched = 0;
  let deferred = 0;

  // Symbol-boundary pause/yield: process one symbol at a time
  for (const sym of symbols) {
    // Check for activity detection before each symbol
    if (deps.shouldPause?.()) {
      report.summary.symbolsEnriched += enriched;
      report.summary.symbolsDeferred += deferred;
      // Flush checkpoint at safe boundary
      try {
        await persistReport(config.checkpointPath, report);
      } catch { /* non-fatal */ }
      return { enriched, deferred, paused: true, reason: 'user_activity_detected' };
    }

    try {
      const result = await deps.enrichmentProvider([sym.id], providerId);
      enriched += result.enriched;
      deferred += result.deferred;
    } catch (e) {
      // Per-symbol failure is non-fatal; continue with next symbol
      deferred++;
    }
  }

  report.summary.symbolsEnriched += enriched;
  report.summary.symbolsDeferred += deferred;
  completePhase(report, 'enrich', { enriched, deferred, provider: providerId });

  return { enriched, deferred, paused: false };
}

// ── Promotion pass ──────────────────────────────────────────────────

/**
 * Run promotion evaluation on eligible memories. Returns aggregated results.
 * Memories eligible for promotion are those with content_hash matching or
 * critical categories with 2+ occurrences.
 */
function runPromotionPass(
  db: DatabaseSync,
  batchSize: number,
): PromotionResult {
  const engine = new PromotionRuleEngine(db);

  // Find candidate memories: those with content_hash set or critical categories
  const candidates = db.prepare(
    `SELECT id FROM memories
     WHERE non_forgettable = 0
       AND (content_hash IS NOT NULL OR category IN ('bug-fix', 'architecture', 'security', 'performance'))
     LIMIT ?`,
  ).all(batchSize) as Array<{ id: string }>;

  const aggregate: PromotionResult = {
    evaluated: 0,
    promoted: 0,
    skipped: 0,
    errors: [],
    promotedMemoryIds: [],
  };

  for (const candidate of candidates) {
    try {
      const result = engine.evaluateForPromotion(candidate.id);
      aggregate.evaluated += result.evaluated;
      aggregate.promoted += result.promoted;
      aggregate.skipped += result.skipped;
      aggregate.errors.push(...result.errors);
      aggregate.promotedMemoryIds.push(...result.promotedMemoryIds);
    } catch (e) {
      aggregate.errors.push(`[${candidate.id}] ${String(e)}`);
    }
  }

  return aggregate;
}

// ── Checkpoint persistence ──────────────────────────────────────────

async function persistReport(path: string, report: SleepRunReport): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(report, null, 2), 'utf-8');
}

/**
 * Load a previously persisted checkpoint report. Returns null if the file
 * does not exist or is malformed.
 */
export async function loadCheckpoint(path: string): Promise<SleepRunReport | null> {
  try {
    const raw = await readFile(path, 'utf-8');
    return JSON.parse(raw) as SleepRunReport;
  } catch {
    return null;
  }
}

// ── Schema validation ───────────────────────────────────────────────

/**
 * Validate that an object conforms to the SleepRunReport schema.
 * Returns an array of validation error messages (empty = valid).
 */
export function validateReport(data: unknown): string[] {
  const errors: string[] = [];

  if (!data || typeof data !== 'object') {
    return ['Report must be an object'];
  }

  const report = data as Record<string, unknown>;

  if (typeof report.runId !== 'string') errors.push('runId must be a string');
  if (typeof report.startedAt !== 'string') errors.push('startedAt must be a string');
  if (report.completedAt !== null && typeof report.completedAt !== 'string') {
    errors.push('completedAt must be null or a string');
  }
  if (!['running', 'completed', 'failed', 'paused'].includes(report.status as string)) {
    errors.push(`status must be one of running|completed|failed|paused, got: ${report.status}`);
  }
  if (!Array.isArray(report.phases)) {
    errors.push('phases must be an array');
  } else {
    for (const phase of report.phases as Record<string, unknown>[]) {
      if (typeof phase.name !== 'string') errors.push('phase.name must be a string');
      if (!['pending', 'running', 'completed', 'failed', 'skipped'].includes(phase.status as string)) {
        errors.push(`phase.status must be one of pending|running|completed|failed|skipped, got: ${phase.status}`);
      }
    }
  }
  if (!report.summary || typeof report.summary !== 'object') {
    errors.push('summary must be an object');
  } else {
    const s = report.summary as Record<string, unknown>;
    if (typeof s.decayEvaluated !== 'number') errors.push('summary.decayEvaluated must be a number');
    if (typeof s.decayTransitioned !== 'number') errors.push('summary.decayTransitioned must be a number');
    if (typeof s.promotionEvaluated !== 'number') errors.push('summary.promotionEvaluated must be a number');
    if (typeof s.promotionPromoted !== 'number') errors.push('summary.promotionPromoted must be a number');
    if (typeof s.symbolsEnriched !== 'number') errors.push('summary.symbolsEnriched must be a number');
    if (typeof s.symbolsDeferred !== 'number') errors.push('summary.symbolsDeferred must be a number');
  }
  if (!Array.isArray(report.errors)) {
    errors.push('errors must be an array');
  }

  return errors;
}
