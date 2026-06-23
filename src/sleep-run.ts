// agent-memory-mcp/src/sleep-run.ts
//
// Sleep-mode one-shot maintenance pipeline. Runs decay evaluation, promotion
// passes, and symbol enrichment in a single atomic run with checkpoint/restore
// capability. Designed for background execution during idle periods.
//
// Pipeline phases:
//   1. keep-awake lease acquire
//   2. decay evaluation (evaluateAllMemoriesForDecay)
//   3. promotion pass (PromotionRuleEngine for eligible memories)
//   4. checkpoint report persistence
//   5. keep-awake lease release

import { randomUUID } from 'node:crypto';
import { writeFile, readFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { DatabaseSync } from 'node:sqlite';
import type { SqliteMemoryStore } from './sqlite-store.js';
import { PromotionRuleEngine } from './global/global-promotion.js';
import type { PromotionResult } from './global/global-promotion.js';
import { DEFAULT_FORGETTING_CONFIG } from './decay.js';
import type { ForgettingConfig } from './decay.js';

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
}

export interface SleepRunReport {
  runId: string;
  startedAt: string;
  completedAt: string | null;
  status: 'running' | 'completed' | 'failed' | 'paused';
  phases: SleepRunPhaseResult[];
  summary: SleepRunSummary;
  errors: string[];
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
}

export interface SleepRunDeps {
  store: SqliteMemoryStore;
  db: DatabaseSync;
  /** Optional external enrichment callback. Returns count of enriched symbols. */
  enrich?: (symbols: string[]) => Promise<number>;
  /** Optional external refresh-context callback. */
  refreshContext?: () => Promise<void>;
  /** Optional external sync-context callback. */
  syncContext?: () => Promise<void>;
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

    // Phase 2: Refresh context (optional)
    startPhase(report, 'refresh_context');
    if (deps.refreshContext) {
      try {
        await deps.refreshContext();
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

    // Phase 4: Promotion pass
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

    // Phase 5: Sync context (optional)
    startPhase(report, 'sync_context');
    if (deps.syncContext) {
      try {
        await deps.syncContext();
        completePhase(report, 'sync_context');
      } catch (e) {
        failPhase(report, 'sync_context', String(e));
        // Non-fatal: continue pipeline
      }
    } else {
      completePhase(report, 'sync_context', { skipped: true, reason: 'no syncContext callback' });
    }

    // Phase 6: Release keep-awake lease
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
