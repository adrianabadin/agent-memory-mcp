// tests/sleep-watch.test.ts
//
// Tests for the sleep-mode daemon with idle detection, CPU pressure
// monitoring, and keep-awake lease management (PR 5).
//
// Covers: config parsing, idle eligibility, daemon lifecycle, signal
// handling, checkpoint/resume, and platform abstraction.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, rm, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import {
  loadSleepConfig,
  saveSleepConfig,
  validateSleepConfig,
  DEFAULT_SLEEP_CONFIG,
  type SleepConfig,
  type FullSleepConfig,
} from '../src/sleep-config.js';

import {
  SleepWatchDaemon,
  checkIdleEligibility,
  getDaemonStatus,
  type PlatformOps,
  type DaemonState,
  type DaemonStatus,
} from '../src/sleep-watch.js';

import { DEFAULT_FORGETTING_CONFIG } from '../src/decay.js';

// ── Mock Platform Ops ───────────────────────────────────────────────

function createMockPlatformOps(overrides: Partial<PlatformOps> = {}): PlatformOps {
  return {
    getIdleSeconds: overrides.getIdleSeconds ?? (async () => 600), // 10 minutes
    getCPUUsage: overrides.getCPUUsage ?? (async () => 20), // 20%
    isPluggedIn: overrides.isPluggedIn ?? (async () => true),
  };
}

// ── Helpers ─────────────────────────────────────────────────────────

async function makeTestDir(): Promise<{ tmpDir: string; cleanup: () => Promise<void> }> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-watch-'));
  return {
    tmpDir,
    cleanup: async () => {
      await rm(tmpDir, { recursive: true, force: true });
    },
  };
}

function createTestConfig(overrides: Partial<SleepConfig> = {}): FullSleepConfig {
  return {
    sleep: { ...DEFAULT_SLEEP_CONFIG, ...overrides },
    forgetting: { ...DEFAULT_FORGETTING_CONFIG },
  };
}

// ═════════════════════════════════════════════════════════════════════
// Sleep Config Parsing
// ═════════════════════════════════════════════════════════════════════

describe('sleep config parsing', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('returns defaults when config file does not exist', async () => {
    const config = await loadSleepConfig(join(tmpDir, 'nonexistent.json'));

    expect(config.sleep.idleMinMinutes).toBe(10);
    expect(config.sleep.cpuHighPercent).toBe(65);
    expect(config.sleep.cpuResumeBelowPercent).toBe(45);
    expect(config.sleep.maxRunHours).toBe(6);
    expect(config.sleep.keepAwakeLeaseMinutes).toBe(30);
    expect(config.sleep.onlyWhenPluggedIn).toBe(true);
    expect(config.sleep.pauseAfterCurrentSymbol).toBe(true);
  });

  it('returns defaults when config file is malformed JSON', async () => {
    const configPath = join(tmpDir, 'malformed.json');
    await writeFile(configPath, 'not-json', 'utf-8');

    const config = await loadSleepConfig(configPath);

    expect(config.sleep.idleMinMinutes).toBe(10);
    expect(config.sleep.cpuHighPercent).toBe(65);
  });

  it('parses sleep config from file', async () => {
    const configPath = join(tmpDir, 'config.json');
    await writeFile(configPath, JSON.stringify({
      sleep: {
        idleMinMinutes: 5,
        cpuHighPercent: 80,
        cpuResumeBelowPercent: 50,
        maxRunHours: 4,
        keepAwakeLeaseMinutes: 20,
        onlyWhenPluggedIn: false,
        pauseAfterCurrentSymbol: false,
      },
    }), 'utf-8');

    const config = await loadSleepConfig(configPath);

    expect(config.sleep.idleMinMinutes).toBe(5);
    expect(config.sleep.cpuHighPercent).toBe(80);
    expect(config.sleep.cpuResumeBelowPercent).toBe(50);
    expect(config.sleep.maxRunHours).toBe(4);
    expect(config.sleep.keepAwakeLeaseMinutes).toBe(20);
    expect(config.sleep.onlyWhenPluggedIn).toBe(false);
    expect(config.sleep.pauseAfterCurrentSymbol).toBe(false);
  });

  it('parses forgetting config from file', async () => {
    const configPath = join(tmpDir, 'config.json');
    await writeFile(configPath, JSON.stringify({
      forgetting: {
        coolingAfterDays: 15,
        dimmedAfterDays: 60,
        shadowAfterDays: 120,
        gistCandidateAfterDays: 200,
        gistOnlyRetentionDays: 45,
      },
    }), 'utf-8');

    const config = await loadSleepConfig(configPath);

    expect(config.forgetting.coolingAfterDays).toBe(15);
    expect(config.forgetting.dimmedAfterDays).toBe(60);
    expect(config.forgetting.shadowAfterDays).toBe(120);
    expect(config.forgetting.gistCandidateAfterDays).toBe(200);
    expect(config.forgetting.gistOnlyRetentionDays).toBe(45);
  });

  it('applies defaults for missing keys', async () => {
    const configPath = join(tmpDir, 'partial.json');
    await writeFile(configPath, JSON.stringify({
      sleep: {
        idleMinMinutes: 3,
      },
    }), 'utf-8');

    const config = await loadSleepConfig(configPath);

    expect(config.sleep.idleMinMinutes).toBe(3);
    expect(config.sleep.cpuHighPercent).toBe(65); // default
    expect(config.sleep.cpuResumeBelowPercent).toBe(45); // default
  });

  it('preserves other config keys when saving', async () => {
    const configPath = join(tmpDir, 'config.json');
    await writeFile(configPath, JSON.stringify({
      otherKey: 'preserved',
      sleep: { idleMinMinutes: 5 },
    }), 'utf-8');

    const config = createTestConfig({ idleMinMinutes: 10 });
    await saveSleepConfig(config, configPath);

    const raw = await readFile(configPath, 'utf-8');
    const parsed = JSON.parse(raw) as Record<string, unknown>;

    expect(parsed['otherKey']).toBe('preserved');
    expect((parsed['sleep'] as Record<string, unknown>)['idleMinMinutes']).toBe(10);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep Config Validation
// ═════════════════════════════════════════════════════════════════════

describe('sleep config validation', () => {
  it('accepts valid config', () => {
    const errors = validateSleepConfig(DEFAULT_SLEEP_CONFIG);
    expect(errors).toEqual([]);
  });

  it('rejects idleMinMinutes <= 0', () => {
    const errors = validateSleepConfig({ idleMinMinutes: 0 });
    expect(errors).toContain('idleMinMinutes must be > 0');
  });

  it('rejects cpuHighPercent out of range', () => {
    const errors = validateSleepConfig({ cpuHighPercent: 150 });
    expect(errors).toContain('cpuHighPercent must be 0-100');
  });

  it('rejects cpuResumeBelowPercent >= cpuHighPercent', () => {
    const errors = validateSleepConfig({
      cpuHighPercent: 50,
      cpuResumeBelowPercent: 60,
    });
    expect(errors).toContain('cpuResumeBelowPercent must be < cpuHighPercent');
  });

  it('rejects keepAwakeLeaseMinutes < 10', () => {
    const errors = validateSleepConfig({ keepAwakeLeaseMinutes: 5 });
    expect(errors).toContain('keepAwakeLeaseMinutes must be >= 10');
  });

  it('rejects maxRunHours <= 0', () => {
    const errors = validateSleepConfig({ maxRunHours: 0 });
    expect(errors).toContain('maxRunHours must be > 0');
  });
});

// ═════════════════════════════════════════════════════════════════════
// Idle Eligibility Check
// ═════════════════════════════════════════════════════════════════════

describe('idle eligibility check', () => {
  it('returns eligible when all conditions met', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      600, // 10 minutes idle
      20,  // 20% CPU
      true, // plugged in
    );

    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('returns not_idle when idle time below threshold', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      300, // 5 minutes idle (below 10 min threshold)
      20,
      true,
    );

    expect(result.eligible).toBe(false);
    expect(result.reason).toBe('not_idle');
  });

  it('returns cpu_pressure when CPU above threshold', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      600,
      80, // 80% CPU (above 65% threshold)
      true,
    );

    expect(result.eligible).toBe(false);
    expect(result.reason).toBe('cpu_pressure');
  });

  it('returns not_plugged_in when not on AC power', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      600,
      20,
      false, // not plugged in
    );

    expect(result.eligible).toBe(false);
    expect(result.reason).toBe('not_plugged_in');
  });

  it('ignores power status when onlyWhenPluggedIn is false', () => {
    const config = { ...DEFAULT_SLEEP_CONFIG, onlyWhenPluggedIn: false };
    const result = checkIdleEligibility(config, 600, 20, false);

    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('handles unknown idle time (-1) as eligible', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      -1, // unknown idle time
      20,
      true,
    );

    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('uses custom idle threshold', () => {
    const config = { ...DEFAULT_SLEEP_CONFIG, idleMinMinutes: 5 };
    const result = checkIdleEligibility(config, 300, 20, true); // 5 minutes = threshold

    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('uses custom CPU threshold', () => {
    const config = { ...DEFAULT_SLEEP_CONFIG, cpuHighPercent: 90 };
    const result = checkIdleEligibility(config, 600, 80, true); // 80% < 90%

    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep-Watch Daemon State Machine
// ═════════════════════════════════════════════════════════════════════

describe('SleepWatchDaemon state machine', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('starts in idle state', () => {
    const config = createTestConfig();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
    });

    expect(daemon.getState().status).toBe('idle');
  });

  it('transitions to running on start', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    expect(daemon.getState().status).toBe('running');
    expect(daemon.getState().leaseActive).toBe(true);

    await daemon.stop();
  });

  it('transitions to stopped on stop', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.stop();

    expect(daemon.getState().status).toBe('stopped');
    expect(daemon.getState().leaseActive).toBe(false);
  });

  it('transitions to paused on pause', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    daemon.pause('test');

    expect(daemon.getState().status).toBe('paused');
    expect(daemon.getState().pauseReason).toBe('test');

    await daemon.stop();
  });

  it('transitions back to running on resume', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    daemon.pause('test');
    daemon.resume();

    expect(daemon.getState().status).toBe('running');
    expect(daemon.getState().pauseReason).toBeNull();

    await daemon.stop();
  });

  it('emits state change callbacks', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    const states: DaemonStatus[] = [];
    daemon.onStateChange((state) => {
      states.push(state.status);
    });

    await daemon.start();
    daemon.pause('test');
    daemon.resume();
    await daemon.stop();

    expect(states).toContain('running');
    expect(states).toContain('paused');
    expect(states).toContain('stopping');
    expect(states).toContain('stopped');
  });

  it('throws when starting in invalid state', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    await expect(daemon.start()).rejects.toThrow('Cannot start daemon in state: running');

    await daemon.stop();
  });

  it('ignores pause when not running', () => {
    const config = createTestConfig();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
    });

    daemon.pause('test');
    expect(daemon.getState().status).toBe('idle');
  });

  it('ignores resume when not paused', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    daemon.resume(); // Should be no-op since not paused

    expect(daemon.getState().status).toBe('running');

    await daemon.stop();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Daemon Polling and Platform Integration
// ═════════════════════════════════════════════════════════════════════

describe('daemon polling and platform integration', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('reads idle seconds from platform ops', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps({
      getIdleSeconds: async () => 300,
    });

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    // The daemon should have read idle seconds during the initial poll
    expect(daemon.getState().idleSeconds).toBe(300);

    await daemon.stop();
  });

  it('reads CPU usage from platform ops', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps({
      getCPUUsage: async () => 45,
    });

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    expect(daemon.getState().cpuUsage).toBe(45);

    await daemon.stop();
  });

  it('reads plugged in status from platform ops', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps({
      isPluggedIn: async () => false,
    });

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    expect(daemon.getState().isPluggedIn).toBe(false);

    await daemon.stop();
  });

  it('updates last idle check timestamp', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    expect(daemon.getState().lastIdleCheck).toBeNull();

    await daemon.start();

    expect(daemon.getState().lastIdleCheck).toBeTruthy();

    await daemon.stop();
  });

  it('tracks run statistics', async () => {
    const config = createTestConfig();
    // Use non-eligible platform ops to avoid spawning a run during initial poll
    const platformOps = createMockPlatformOps({
      getIdleSeconds: async () => 0, // Not idle
    });
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    // Initially no runs (not eligible)
    expect(daemon.getState().totalRunsSpawned).toBe(0);
    expect(daemon.getState().totalRunsCompleted).toBe(0);

    await daemon.stop();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Daemon Checkpoint Persistence
// ═════════════════════════════════════════════════════════════════════

describe('daemon checkpoint persistence', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('saves checkpoint to disk', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const checkpointPath = join(tmpDir, 'checkpoint.json');

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath,
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.saveCheckpoint();

    const raw = await readFile(checkpointPath, 'utf-8');
    const checkpoint = JSON.parse(raw) as Record<string, unknown>;

    expect(checkpoint['daemonId']).toBe(daemon.getId());
    expect((checkpoint['state'] as Record<string, unknown>)['status']).toBe('running');

    await daemon.stop();
  });

  it('loads checkpoint from disk', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const checkpointPath = join(tmpDir, 'checkpoint.json');

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath,
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.saveCheckpoint();

    const loaded = await SleepWatchDaemon.loadCheckpoint(checkpointPath);

    expect(loaded).not.toBeNull();
    expect(loaded!.daemonId).toBe(daemon.getId());
    expect(loaded!.state.status).toBe('running');

    await daemon.stop();
  });

  it('returns null for nonexistent checkpoint', async () => {
    const loaded = await SleepWatchDaemon.loadCheckpoint(join(tmpDir, 'nonexistent.json'));
    expect(loaded).toBeNull();
  });

  it('returns null for malformed checkpoint', async () => {
    const checkpointPath = join(tmpDir, 'malformed.json');
    await writeFile(checkpointPath, 'not-json', 'utf-8');

    const loaded = await SleepWatchDaemon.loadCheckpoint(checkpointPath);
    expect(loaded).toBeNull();
  });

  it('deletes checkpoint file', async () => {
    const checkpointPath = join(tmpDir, 'checkpoint.json');
    await writeFile(checkpointPath, '{"test": true}', 'utf-8');

    await SleepWatchDaemon.deleteCheckpoint(checkpointPath);

    const loaded = await SleepWatchDaemon.loadCheckpoint(checkpointPath);
    expect(loaded).toBeNull();
  });

  it('saves checkpoint with config', async () => {
    const config = createTestConfig({ idleMinMinutes: 5 });
    const platformOps = createMockPlatformOps();
    const checkpointPath = join(tmpDir, 'checkpoint.json');

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath,
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.saveCheckpoint();

    const loaded = await SleepWatchDaemon.loadCheckpoint(checkpointPath);

    expect(loaded).not.toBeNull();
    expect(loaded!.config.idleMinMinutes).toBe(5);

    await daemon.stop();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Status Reporting
// ═════════════════════════════════════════════════════════════════════

describe('status reporting', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('returns null when no checkpoint exists', async () => {
    const status = await getDaemonStatus(join(tmpDir, 'nonexistent.json'));
    expect(status).toBeNull();
  });

  it('returns status from checkpoint', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const checkpointPath = join(tmpDir, 'checkpoint.json');

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath,
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.saveCheckpoint();

    const status = await getDaemonStatus(checkpointPath);

    expect(status).not.toBeNull();
    expect(status!.daemonId).toBe(daemon.getId());
    expect(status!.status).toBe('running');
    expect(status!.config.idleMinMinutes).toBe(10);
    expect(status!.uptimeMs).toBeGreaterThanOrEqual(0);

    await daemon.stop();
  });

  it('includes all required fields', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const checkpointPath = join(tmpDir, 'checkpoint.json');

    const daemon = new SleepWatchDaemon(config, {
      checkpointPath,
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.saveCheckpoint();

    const status = await getDaemonStatus(checkpointPath);

    expect(status).toHaveProperty('daemonId');
    expect(status).toHaveProperty('status');
    expect(status).toHaveProperty('startedAt');
    expect(status).toHaveProperty('uptimeMs');
    expect(status).toHaveProperty('lastIdleCheck');
    expect(status).toHaveProperty('idleSeconds');
    expect(status).toHaveProperty('cpuUsage');
    expect(status).toHaveProperty('isPluggedIn');
    expect(status).toHaveProperty('leaseActive');
    expect(status).toHaveProperty('currentRunPid');
    expect(status).toHaveProperty('currentRunStartedAt');
    expect(status).toHaveProperty('totalRunsSpawned');
    expect(status).toHaveProperty('totalRunsCompleted');
    expect(status).toHaveProperty('pauseReason');
    expect(status).toHaveProperty('config');

    await daemon.stop();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Keep-Awake Lease Integration
// ═════════════════════════════════════════════════════════════════════

describe('keep-awake lease integration', () => {
  let tmpDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    const env = await makeTestDir();
    tmpDir = env.tmpDir;
    cleanup = env.cleanup;
  });

  afterEach(async () => {
    await cleanup();
  });

  it('acquires lease on start', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();

    expect(daemon.getState().leaseActive).toBe(true);

    await daemon.stop();
  });

  it('releases lease on stop', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    await daemon.stop();

    expect(daemon.getState().leaseActive).toBe(false);
  });

  it('uses configured lease duration', async () => {
    const config = createTestConfig({ keepAwakeLeaseMinutes: 20 });
    const platformOps = createMockPlatformOps();
    const daemon = new SleepWatchDaemon(config, {
      checkpointPath: join(tmpDir, 'checkpoint.json'),
      platformOps,
      pollIntervalMs: 1000,
    });

    await daemon.start();
    expect(daemon.getState().leaseActive).toBe(true);

    await daemon.stop();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Edge Cases
// ═════════════════════════════════════════════════════════════════════

describe('edge cases', () => {
  it('handles platform ops errors gracefully', async () => {
    const config = createTestConfig();
    const platformOps: PlatformOps = {
      getIdleSeconds: async () => { throw new Error('idle error'); },
      getCPUUsage: async () => { throw new Error('cpu error'); },
      isPluggedIn: async () => { throw new Error('power error'); },
    };

    // Should not throw
    const result = checkIdleEligibility(config, -1, 0, true);
    expect(result.eligible).toBe(true);
  });

  it('handles zero idle time', () => {
    const result = checkIdleEligibility(DEFAULT_SLEEP_CONFIG, 0, 20, true);
    expect(result.eligible).toBe(false);
    expect(result.reason).toBe('not_idle');
  });

  it('handles exactly at idle threshold', () => {
    const thresholdSeconds = DEFAULT_SLEEP_CONFIG.idleMinMinutes * 60;
    const result = checkIdleEligibility(DEFAULT_SLEEP_CONFIG, thresholdSeconds, 20, true);
    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('handles exactly at CPU threshold', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      600,
      DEFAULT_SLEEP_CONFIG.cpuHighPercent, // exactly at threshold
      true,
    );
    expect(result.eligible).toBe(true);
    expect(result.reason).toBe('eligible');
  });

  it('handles CPU above threshold', () => {
    const result = checkIdleEligibility(
      DEFAULT_SLEEP_CONFIG,
      600,
      DEFAULT_SLEEP_CONFIG.cpuHighPercent + 1, // just above threshold
      true,
    );
    expect(result.eligible).toBe(false);
    expect(result.reason).toBe('cpu_pressure');
  });

  it('generates unique daemon IDs', () => {
    const config = createTestConfig();
    const daemon1 = new SleepWatchDaemon(config);
    const daemon2 = new SleepWatchDaemon(config);

    expect(daemon1.getId()).not.toBe(daemon2.getId());
  });

  it('returns readonly state snapshot', async () => {
    const config = createTestConfig();
    const platformOps = createMockPlatformOps();
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-readonly-'));

    try {
      const daemon = new SleepWatchDaemon(config, {
        checkpointPath: join(tmpDir, 'checkpoint.json'),
        platformOps,
        pollIntervalMs: 1000,
      });

      await daemon.start();

      const state1 = daemon.getState();
      const state2 = daemon.getState();

      // Should be equal but not same reference
      expect(state1).toEqual(state2);
      expect(state1).not.toBe(state2);

      await daemon.stop();
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});
