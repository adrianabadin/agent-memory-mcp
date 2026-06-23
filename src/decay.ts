// agent-memory-mcp/src/decay.ts

export const EVERGREEN_TAGS = new Set(['evergreen', 'never-forget']);

const MS_PER_DAY = 86_400_000;

/**
 * Parse the MEMORY_DECAY_HALF_LIFE env var into a number of days.
 * Returns 30 (default) if undefined; 0 if non-numeric or explicitly "0" (disables decay).
 */
export function parseDecayHalfLife(value: string | undefined): number {
  if (value === undefined) return 30;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? 0 : Math.max(0, parsed);
}

/**
 * Exponential decay factor based on how old `updatedAt` is.
 * Returns 1.0 if halfLifeDays <= 0 (decay disabled).
 */
export function computeDecayFactor(updatedAt: string, halfLifeDays: number): number {
  if (halfLifeDays <= 0) return 1.0;
  const ageMs = Date.now() - new Date(updatedAt).getTime();
  const ageDays = Math.max(0, ageMs / MS_PER_DAY);
  return Math.pow(0.5, ageDays / halfLifeDays);
}

/**
 * Multiplier applied to effective half-life based on access frequency and recency.
 * High-access and recently-accessed memories decay more slowly.
 * Range: 1.0 (never accessed, old) to 3.0 (max access, very recent).
 */
export function importanceMultiplier(row: Record<string, unknown>): number {
  const accessCount = (row['access_count'] as number) ?? 0;
  const lastAccessedAt = (row['last_accessed_at'] as string) ?? (row['updated_at'] as string);

  // Access frequency boost: linear ramp 1.0 → 2.0, capped at 20 accesses
  const accessBoost = 1 + Math.min(accessCount, 20) / 20;

  // Recency of access: recently-accessed memories are clearly still useful
  const daysSinceAccess = Math.max(
    0,
    (Date.now() - new Date(lastAccessedAt).getTime()) / MS_PER_DAY,
  );
  const recencyBoost = daysSinceAccess < 7 ? 1.5 : daysSinceAccess < 30 ? 1.2 : 1.0;

  return accessBoost * recencyBoost; // range: 1.0 to 3.0
}

// ── Gradual forgetting state machine (PR 2a) ───────────────────────
//
// Staged decay: a memory advances sequentially
//   active → cooling → dimmed → shadow → gist_only → purged
// based on the time elapsed since its last reinforcement. Each stage lowers
// the activation score by ~0.2, so ranking can down-weight stale memories
// before any content is pruned. The original temporal-decay functions above
// (computeDecayFactor / importanceMultiplier) remain unchanged and are still
// used by the live search ranking path; the stage machine here drives the
// scheduled, durable `memory_state` / `activation_score` columns and the
// `memory_decay_log` audit trail.
//
// NOTE: this module intentionally has NO SQLite dependency. DB persistence is
// the store's job — `transitionDecayState` accepts an optional `writeLog`
// callback so the store can inject a `memory_decay_log` INSERT without
// coupling the state machine to the database.

/** Ordered decay states, most-recalled → fully-forgotten. */
export const DECAY_STAGE_ORDER = [
  'active',
  'cooling',
  'dimmed',
  'shadow',
  'gist_only',
  'purged',
] as const;

export type DecayState = (typeof DECAY_STAGE_ORDER)[number];

/** Day thresholds for each stage boundary. All intervals are additive: the
 *  cumulative time since `last_reinforced_at` determines the target stage. */
export interface ForgettingConfig {
  /** active → cooling boundary (days). */
  coolingAfterDays: number;
  /** additional cooling → dimmed (days). */
  dimmedAfterDays: number;
  /** additional dimmed → shadow (days). */
  shadowAfterDays: number;
  /** additional shadow → gist_only (days). */
  gistCandidateAfterDays: number;
  /** additional gist_only → purged (days). */
  gistOnlyRetentionDays: number;
}

/** Defaults mirror the sleep-config schema in design.md. */
export const DEFAULT_FORGETTING_CONFIG: ForgettingConfig = {
  coolingAfterDays: 30,
  dimmedAfterDays: 90,
  shadowAfterDays: 180,
  gistCandidateAfterDays: 365,
  gistOnlyRetentionDays: 90,
};

/** Activation score per stage (1.0 → 0.8 → 0.6 → 0.4 → 0.2 → 0.0). */
const STAGE_SCORES: Record<DecayState, number> = {
  active: 1.0,
  cooling: 0.8,
  dimmed: 0.6,
  shadow: 0.4,
  gist_only: 0.2,
  purged: 0.0,
};

/**
 * Return the activation score for a decay stage. Scores decrease by ~0.2 per
 * stage so stale memories rank lower in retrieval without being deleted.
 */
export function evaluateActivationScore(state: DecayState): number {
  return STAGE_SCORES[state];
}

export interface DecayTransitionInput {
  memoryId: string;
  currentState: DecayState;
  /** ISO timestamp of the most recent reinforcement. Drives all boundaries. */
  lastReinforcedAt: string;
  config: ForgettingConfig;
  /** Epoch ms; injectable for deterministic tests. Defaults to Date.now(). */
  now?: number;
}

export interface DecayTransitionResult {
  newState: DecayState;
  newScore: number;
  /** Human-readable reason for the (lack of) transition. */
  reason: string;
  /** True only when `newState !== currentState`. */
  transitioned: boolean;
}

export interface DecayLogRecord {
  memory_id: string;
  from_state: DecayState;
  to_state: DecayState;
  reason: string;
  score_before: number;
  score_after: number;
}

/** Injected by the store to persist a transition into `memory_decay_log`. */
export type DecayLogWriter = (record: DecayLogRecord) => void;

/**
 * Pure decision core: given a memory's current state and the elapsed time since
 * its last reinforcement, compute the target decay stage (jumping multiple
 * stages if a lot of time passed) and never regressing below the current state.
 */
export function computeDecayTransition(input: DecayTransitionInput): DecayTransitionResult {
  const { currentState, lastReinforcedAt, config, now = Date.now() } = input;
  const elapsedDays = Math.max(0, (now - new Date(lastReinforcedAt).getTime()) / MS_PER_DAY);

  // Cumulative day boundaries measured from lastReinforcedAt.
  const boundaries: Array<{ state: DecayState; threshold: number }> = [
    { state: 'cooling', threshold: config.coolingAfterDays },
    { state: 'dimmed', threshold: config.coolingAfterDays + config.dimmedAfterDays },
    {
      state: 'shadow',
      threshold: config.coolingAfterDays + config.dimmedAfterDays + config.shadowAfterDays,
    },
    {
      state: 'gist_only',
      threshold:
        config.coolingAfterDays +
        config.dimmedAfterDays +
        config.shadowAfterDays +
        config.gistCandidateAfterDays,
    },
    {
      state: 'purged',
      threshold:
        config.coolingAfterDays +
        config.dimmedAfterDays +
        config.shadowAfterDays +
        config.gistCandidateAfterDays +
        config.gistOnlyRetentionDays,
    },
  ];

  // Walk boundaries in order; the furthest crossed wins.
  let target: DecayState = 'active';
  let crossedThreshold = 0;
  for (const b of boundaries) {
    if (elapsedDays >= b.threshold) {
      target = b.state;
      crossedThreshold = b.threshold;
    }
  }

  // Never regress: reinforcement (PR 2b) is what moves memories back toward
  // active; the scheduled evaluator only advances or holds.
  const currentOrder = DECAY_STAGE_ORDER.indexOf(currentState);
  const targetOrder = DECAY_STAGE_ORDER.indexOf(target);
  if (targetOrder < currentOrder) {
    return {
      newState: currentState,
      newScore: evaluateActivationScore(currentState),
      reason: `already past ${target} (no regression); held in ${currentState}`,
      transitioned: false,
    };
  }

  const transitioned = targetOrder > currentOrder;
  const reason = transitioned
    ? `elapsed ${elapsedDays.toFixed(1)}d >= ${crossedThreshold}d → ${target}`
    : `within ${currentState} window (${elapsedDays.toFixed(1)}d elapsed)`;

  return {
    newState: target,
    newScore: evaluateActivationScore(target),
    reason,
    transitioned,
  };
}

/**
 * Evaluate a single memory's decay transition. Computes the new state/score and,
 * when provided, persists a transition record via `writeLog` (the store binds
 * this to a `memory_decay_log` INSERT). The caller is responsible for writing
 * the resulting `memory_state` / `activation_score` back to the row.
 */
export function transitionDecayState(
  input: DecayTransitionInput,
  writeLog?: DecayLogWriter,
): DecayTransitionResult {
  const result = computeDecayTransition(input);
  if (result.transitioned && writeLog) {
    writeLog({
      memory_id: input.memoryId,
      from_state: input.currentState,
      to_state: result.newState,
      reason: result.reason,
      score_before: evaluateActivationScore(input.currentState),
      score_after: result.newScore,
    });
  }
  return result;
}

// ── Reinforcement (PR 2b) ──────────────────────────────────────────

/** Score boost applied when a deep-stage (shadow/gist_only) memory is accessed. */
export const REINFORCEMENT_SCORE_BOOST = 0.7;

/** Maximum score a reinforced memory can reach without full state revert. */
export const REINFORCED_SCORE_CAP = 0.95;

export interface ReinforcementInput {
  memoryId: string;
  currentState: DecayState;
  activationScore: number;
  config: ForgettingConfig;
  /** Epoch ms; injectable for deterministic tests. Defaults to Date.now(). */
  now?: number;
}

export interface ReinforcementResult {
  newState: DecayState;
  newScore: number;
  /** True when state reverted (cooling/dimmed → active). */
  transitioned: boolean;
  /** True when active access was a no-op (no log should be emitted). */
  silent: boolean;
  reason: string;
}

export interface ReinforcementLogRecord {
  memory_id: string;
  from_state: DecayState;
  to_state: DecayState;
  reason: string;
  score_before: number;
  score_after: number;
}

/** Injected by the store to persist a reinforcement record into `memory_decay_log`. */
export type ReinforcementLogWriter = (record: ReinforcementLogRecord) => void;

/**
 * Pure decision core for memory reinforcement on access.
 *
 * Rules:
 *   • active → silent no-op (score stays 1.0, no log)
 *   • purged → no-op (score stays 0.0)
 *   • cooling / dimmed → revert to active (score = 1.0)
 *   • shadow / gist_only → keep state, boost score by REINFORCEMENT_SCORE_BOOST
 *     (capped at REINFORCED_SCORE_CAP = 0.95)
 */
export function reinforceMemory(
  input: ReinforcementInput,
  writeLog?: ReinforcementLogWriter,
): ReinforcementResult {
  const { currentState, activationScore } = input;

  // Active: silent no-op
  if (currentState === 'active') {
    return {
      newState: 'active',
      newScore: 1.0,
      transitioned: false,
      silent: true,
      reason: 'already active; no reinforcement needed',
    };
  }

  // Purged: no-op
  if (currentState === 'purged') {
    return {
      newState: 'purged',
      newScore: 0.0,
      transitioned: false,
      silent: true,
      reason: 'purged memories cannot be reinforced',
    };
  }

  // Cooling / dimmed: revert to active
  if (currentState === 'cooling' || currentState === 'dimmed') {
    const result: ReinforcementResult = {
      newState: 'active',
      newScore: 1.0,
      transitioned: true,
      silent: false,
      reason: `reinforced from ${currentState} back to active`,
    };
    if (writeLog) {
      writeLog({
        memory_id: input.memoryId,
        from_state: currentState,
        to_state: 'active',
        reason: result.reason,
        score_before: activationScore,
        score_after: 1.0,
      });
    }
    return result;
  }

  // Shadow / gist_only: keep state, boost score
  const boosted = Math.min(activationScore + REINFORCEMENT_SCORE_BOOST, REINFORCED_SCORE_CAP);
  const result: ReinforcementResult = {
    newState: currentState,
    newScore: boosted,
    transitioned: false,
    silent: false,
    reason: `reinforced in ${currentState}; score ${activationScore.toFixed(2)} → ${boosted.toFixed(2)}`,
  };
  if (writeLog) {
    writeLog({
      memory_id: input.memoryId,
      from_state: currentState,
      to_state: currentState,
      reason: result.reason,
      score_before: activationScore,
      score_after: boosted,
    });
  }
  return result;
}

// ── Batch decay evaluation (PR 2b) ─────────────────────────────────

export interface DecayBatchInput {
  memoryId: string;
  currentState: DecayState;
  lastReinforcedAt: string;
  activationScore: number;
}

export interface DecayBatchResult {
  processed: number;
  transitioned: number;
  results: Array<{ memoryId: string; result: DecayTransitionResult }>;
}

/**
 * Pure batch aggregation: evaluate a collection of memories for decay transitions.
 * The store fetches rows and calls this, then persists all transitions in one
 * atomic transaction.
 */
export function evaluateDecayBatch(
  memories: DecayBatchInput[],
  config: ForgettingConfig,
  now: number = Date.now(),
): DecayBatchResult {
  const results: Array<{ memoryId: string; result: DecayTransitionResult }> = [];
  let transitioned = 0;

  for (const mem of memories) {
    const result = computeDecayTransition({
      memoryId: mem.memoryId,
      currentState: mem.currentState,
      lastReinforcedAt: mem.lastReinforcedAt,
      config,
      now,
    });
    if (result.transitioned) transitioned++;
    results.push({ memoryId: mem.memoryId, result });
  }

  return { processed: memories.length, transitioned, results };
}
