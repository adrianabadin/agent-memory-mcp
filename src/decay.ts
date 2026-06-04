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
