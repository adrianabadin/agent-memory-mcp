// agent-memory-mcp/src/sleep-config.ts
//
// Sleep mode configuration parser and validator. Reads global config from
// ~/.config/opencode/project-memory-context.json and applies defaults.
//
// Config schema:
// {
//   "sleep": { idleMinMinutes, cpuHighPercent, cpuResumeBelowPercent, ... },
//   "forgetting": { coolingAfterDays, dimmedAfterDays, ... }
// }

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { ForgettingConfig } from './decay.js';
import { DEFAULT_FORGETTING_CONFIG } from './decay.js';

// ── Sleep Config Types ──────────────────────────────────────────────

export interface SleepConfig {
  /** Minutes of idle time before sleep-run is eligible. Default: 10 */
  idleMinMinutes: number;
  /** CPU usage % above which sleep-run pauses. Default: 65 */
  cpuHighPercent: number;
  /** CPU usage % below which sleep-run resumes. Default: 45 */
  cpuResumeBelowPercent: number;
  /** Maximum hours a sleep-run can execute. Default: 6 */
  maxRunHours: number;
  /** Keep-awake lease duration in minutes. Default: 30 */
  keepAwakeLeaseMinutes: number;
  /** Only run when machine is plugged in (laptops). Default: true */
  onlyWhenPluggedIn: boolean;
  /** Pause after completing current symbol (not mid-symbol). Default: true */
  pauseAfterCurrentSymbol: boolean;
}

export interface FullSleepConfig {
  sleep: SleepConfig;
  forgetting: ForgettingConfig;
}

// ── Defaults ────────────────────────────────────────────────────────

export const DEFAULT_SLEEP_CONFIG: SleepConfig = {
  idleMinMinutes: 10,
  cpuHighPercent: 65,
  cpuResumeBelowPercent: 45,
  maxRunHours: 6,
  keepAwakeLeaseMinutes: 30,
  onlyWhenPluggedIn: true,
  pauseAfterCurrentSymbol: true,
};

// ── Config Path ─────────────────────────────────────────────────────

export function getDefaultConfigPath(): string {
  return join(homedir(), '.config', 'opencode', 'project-memory-context.json');
}

// ── Validation ──────────────────────────────────────────────────────

export function validateSleepConfig(config: Partial<SleepConfig>): string[] {
  const errors: string[] = [];

  if (config.idleMinMinutes !== undefined && config.idleMinMinutes <= 0) {
    errors.push('idleMinMinutes must be > 0');
  }
  if (config.cpuHighPercent !== undefined && (config.cpuHighPercent < 0 || config.cpuHighPercent > 100)) {
    errors.push('cpuHighPercent must be 0-100');
  }
  if (config.cpuResumeBelowPercent !== undefined && (config.cpuResumeBelowPercent < 0 || config.cpuResumeBelowPercent > 100)) {
    errors.push('cpuResumeBelowPercent must be 0-100');
  }
  if (config.cpuHighPercent !== undefined && config.cpuResumeBelowPercent !== undefined) {
    if (config.cpuResumeBelowPercent >= config.cpuHighPercent) {
      errors.push('cpuResumeBelowPercent must be < cpuHighPercent');
    }
  }
  if (config.keepAwakeLeaseMinutes !== undefined && config.keepAwakeLeaseMinutes < 10) {
    errors.push('keepAwakeLeaseMinutes must be >= 10');
  }
  if (config.maxRunHours !== undefined && config.maxRunHours <= 0) {
    errors.push('maxRunHours must be > 0');
  }

  return errors;
}

// ── Config Loading ──────────────────────────────────────────────────

/**
 * Load sleep configuration from the global config file.
 * Returns defaults for any missing keys. Creates the file if it doesn't exist.
 */
export async function loadSleepConfig(configPath?: string): Promise<FullSleepConfig> {
  const path = configPath ?? getDefaultConfigPath();

  let raw: string;
  try {
    raw = await readFile(path, 'utf-8');
  } catch {
    // File doesn't exist — return defaults
    return {
      sleep: { ...DEFAULT_SLEEP_CONFIG },
      forgetting: { ...DEFAULT_FORGETTING_CONFIG },
    };
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    // Malformed JSON — return defaults
    return {
      sleep: { ...DEFAULT_SLEEP_CONFIG },
      forgetting: { ...DEFAULT_FORGETTING_CONFIG },
    };
  }

  const sleepRaw = (parsed['sleep'] ?? {}) as Record<string, unknown>;
  const forgettingRaw = (parsed['forgetting'] ?? {}) as Record<string, unknown>;

  const sleep: SleepConfig = {
    idleMinMinutes: typeof sleepRaw['idleMinMinutes'] === 'number' ? sleepRaw['idleMinMinutes'] : DEFAULT_SLEEP_CONFIG.idleMinMinutes,
    cpuHighPercent: typeof sleepRaw['cpuHighPercent'] === 'number' ? sleepRaw['cpuHighPercent'] : DEFAULT_SLEEP_CONFIG.cpuHighPercent,
    cpuResumeBelowPercent: typeof sleepRaw['cpuResumeBelowPercent'] === 'number' ? sleepRaw['cpuResumeBelowPercent'] : DEFAULT_SLEEP_CONFIG.cpuResumeBelowPercent,
    maxRunHours: typeof sleepRaw['maxRunHours'] === 'number' ? sleepRaw['maxRunHours'] : DEFAULT_SLEEP_CONFIG.maxRunHours,
    keepAwakeLeaseMinutes: typeof sleepRaw['keepAwakeLeaseMinutes'] === 'number' ? sleepRaw['keepAwakeLeaseMinutes'] : DEFAULT_SLEEP_CONFIG.keepAwakeLeaseMinutes,
    onlyWhenPluggedIn: typeof sleepRaw['onlyWhenPluggedIn'] === 'boolean' ? sleepRaw['onlyWhenPluggedIn'] : DEFAULT_SLEEP_CONFIG.onlyWhenPluggedIn,
    pauseAfterCurrentSymbol: typeof sleepRaw['pauseAfterCurrentSymbol'] === 'boolean' ? sleepRaw['pauseAfterCurrentSymbol'] : DEFAULT_SLEEP_CONFIG.pauseAfterCurrentSymbol,
  };

  const forgetting: ForgettingConfig = {
    coolingAfterDays: typeof forgettingRaw['coolingAfterDays'] === 'number' ? forgettingRaw['coolingAfterDays'] : DEFAULT_FORGETTING_CONFIG.coolingAfterDays,
    dimmedAfterDays: typeof forgettingRaw['dimmedAfterDays'] === 'number' ? forgettingRaw['dimmedAfterDays'] : DEFAULT_FORGETTING_CONFIG.dimmedAfterDays,
    shadowAfterDays: typeof forgettingRaw['shadowAfterDays'] === 'number' ? forgettingRaw['shadowAfterDays'] : DEFAULT_FORGETTING_CONFIG.shadowAfterDays,
    gistCandidateAfterDays: typeof forgettingRaw['gistCandidateAfterDays'] === 'number' ? forgettingRaw['gistCandidateAfterDays'] : DEFAULT_FORGETTING_CONFIG.gistCandidateAfterDays,
    gistOnlyRetentionDays: typeof forgettingRaw['gistOnlyRetentionDays'] === 'number' ? forgettingRaw['gistOnlyRetentionDays'] : DEFAULT_FORGETTING_CONFIG.gistOnlyRetentionDays,
  };

  // Validate
  const errors = validateSleepConfig(sleep);
  if (errors.length > 0) {
    throw new Error(`Invalid sleep config: ${errors.join('; ')}`);
  }

  return { sleep, forgetting };
}

/**
 * Save sleep configuration to the global config file.
 * Merges with existing config (preserves other keys).
 */
export async function saveSleepConfig(config: FullSleepConfig, configPath?: string): Promise<void> {
  const path = configPath ?? getDefaultConfigPath();

  let existing: Record<string, unknown> = {};
  try {
    const raw = await readFile(path, 'utf-8');
    existing = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    // File doesn't exist or malformed — start fresh
  }

  existing['sleep'] = config.sleep;
  existing['forgetting'] = config.forgetting;

  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(existing, null, 2), 'utf-8');
}
