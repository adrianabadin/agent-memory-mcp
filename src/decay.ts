// agent-memory-mcp/src/decay.ts

export const EVERGREEN_TAGS = new Set(['evergreen', 'never-forget']);

/**
 * Parse the MEMORY_DECAY_HALF_LIFE env var into a number of days.
 * Returns 30 (default) if undefined; 0 if explicitly "0" (disables decay).
 */
export function parseDecayHalfLife(value: string | undefined): number {
  if (value === undefined) return 30;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? 30 : Math.max(0, parsed);
}

/**
 * Exponential decay factor based on how old `updatedAt` is.
 * Returns 1.0 if halfLifeDays <= 0 (decay disabled).
 */
export function computeDecayFactor(updatedAt: string, halfLifeDays: number): number {
  if (halfLifeDays <= 0) return 1.0;
  const ageMs = Date.now() - new Date(updatedAt).getTime();
  const ageDays = Math.max(0, ageMs / 86_400_000);
  return Math.pow(0.5, ageDays / halfLifeDays);
}

/**
 * Multiplier applied to half-life based on memory importance signals.
 * High-importance memories decay more slowly.
 */
export function importanceMultiplier(row: Record<string, unknown>): number {
  const tags: string[] = (() => {
    try {
      const raw = row['tags'];
      return Array.isArray(raw) ? raw : JSON.parse(String(raw ?? '[]'));
    } catch {
      return [];
    }
  })();

  if (tags.some(t => EVERGREEN_TAGS.has(t))) return Infinity;
  if (tags.includes('important') || tags.includes('critical')) return 4;
  if (tags.includes('architecture') || tags.includes('decision')) return 2;
  return 1;
}
