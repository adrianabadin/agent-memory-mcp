// tests/sleep-run.test.ts
//
// Tests for the sleep-mode one-shot maintenance pipeline (PR 4).
// Covers: checkpoint schema, keep-awake lease, decay integration,
// promotion pass, pipeline orchestration, and error handling.

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import {
  executeSleepRun,
  loadCheckpoint,
  validateReport,
  InMemoryKeepAwakeLeaseManager,
  type SleepRunReport,
  type SleepRunConfig,
  type SleepRunDeps,
  type KeepAwakeLease,
} from '../src/sleep-run.js';
import { DEFAULT_FORGETTING_CONFIG } from '../src/decay.js';

const MS_PER_DAY = 86_400_000;

// ── Helpers ─────────────────────────────────────────────────────────

async function makeTestEnv(): Promise<{
  store: SqliteMemoryStore;
  db: DatabaseSync;
  dbPath: string;
  tmpDir: string;
  cleanup: () => Promise<void>;
}> {
  const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-run-'));
  const dbPath = join(tmpDir, 'test.db');
  const embedder = new MockEmbedder();
  const store = new SqliteMemoryStore(dbPath, embedder);
  await store.initialize();
  const db = store.getDb();
  return {
    store,
    db,
    dbPath,
    tmpDir,
    cleanup: async () => {
      store.close();
      await rm(tmpDir, { recursive: true, force: true });
    },
  };
}

// ═════════════════════════════════════════════════════════════════════
// Keep-Awake Lease Manager
// ═════════════════════════════════════════════════════════════════════

describe('InMemoryKeepAwakeLeaseManager', () => {
  let manager: InMemoryKeepAwakeLeaseManager;

  beforeEach(() => {
    manager = new InMemoryKeepAwakeLeaseManager();
  });

  it('acquires a lease when none exists', () => {
    expect(manager.acquire('holder-1')).toBe(true);
    expect(manager.isActive()).toBe(true);
    expect(manager.getLease()?.holder).toBe('holder-1');
  });

  it('rejects acquisition when active lease exists', () => {
    manager.acquire('holder-1');
    expect(manager.acquire('holder-2')).toBe(false);
    expect(manager.getLease()?.holder).toBe('holder-1');
  });

  it('allows acquisition after lease expires', () => {
    // Acquire with 1ms TTL
    manager.acquire('holder-1', 1);
    // Wait for expiry
    const start = Date.now();
    while (Date.now() - start < 10) { /* spin */ }

    expect(manager.acquire('holder-2')).toBe(true);
    expect(manager.getLease()?.holder).toBe('holder-2');
  });

  it('renews an active lease held by the same holder', () => {
    manager.acquire('holder-1', 1000);
    expect(manager.renew('holder-1', 60_000)).toBe(true);
    expect(manager.isActive()).toBe(true);
  });

  it('rejects renewal from a different holder', () => {
    manager.acquire('holder-1', 1000);
    expect(manager.renew('holder-2', 60_000)).toBe(false);
  });

  it('rejects renewal when lease has expired', () => {
    manager.acquire('holder-1', 1);
    const start = Date.now();
    while (Date.now() - start < 10) { /* spin */ }

    expect(manager.renew('holder-1', 60_000)).toBe(false);
  });

  it('releases a lease held by the same holder', () => {
    manager.acquire('holder-1');
    expect(manager.release('holder-1')).toBe(true);
    expect(manager.isActive()).toBe(false);
    expect(manager.getLease()).toBeNull();
  });

  it('rejects release from a different holder', () => {
    manager.acquire('holder-1');
    expect(manager.release('holder-2')).toBe(false);
    expect(manager.isActive()).toBe(true);
  });

  it('returns null for getLease when no lease exists', () => {
    expect(manager.getLease()).toBeNull();
  });

  it('returns null for getLease when lease has expired', () => {
    manager.acquire('holder-1', 1);
    const start = Date.now();
    while (Date.now() - start < 10) { /* spin */ }

    expect(manager.getLease()).toBeNull();
    expect(manager.isActive()).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Checkpoint Report Schema Validation
// ═════════════════════════════════════════════════════════════════════

describe('SleepRunReport schema validation', () => {
  it('validates a complete valid report', () => {
    const report: SleepRunReport = {
      runId: randomUUID(),
      startedAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      status: 'completed',
      phases: [
        { name: 'decay_evaluation', status: 'completed', startedAt: new Date().toISOString(), completedAt: new Date().toISOString() },
      ],
      summary: {
        decayEvaluated: 10,
        decayTransitioned: 3,
        promotionEvaluated: 5,
        promotionPromoted: 1,
        symbolsEnriched: 0,
        symbolsDeferred: 0,
      },
      errors: [],
    };
    expect(validateReport(report)).toEqual([]);
  });

  it('rejects non-object input', () => {
    expect(validateReport(null)).toContain('Report must be an object');
    expect(validateReport('string')).toContain('Report must be an object');
  });

  it('requires runId to be a string', () => {
    const errors = validateReport({ runId: 123 });
    expect(errors).toContain('runId must be a string');
  });

  it('requires valid status enum', () => {
    const errors = validateReport({
      runId: 'x',
      startedAt: '2024-01-01',
      completedAt: null,
      status: 'invalid',
      phases: [],
      summary: { decayEvaluated: 0, decayTransitioned: 0, promotionEvaluated: 0, promotionPromoted: 0, symbolsEnriched: 0, symbolsDeferred: 0 },
      errors: [],
    });
    expect(errors.some(e => e.includes('status'))).toBe(true);
  });

  it('requires phases to be an array', () => {
    const errors = validateReport({
      runId: 'x',
      startedAt: '2024-01-01',
      completedAt: null,
      status: 'running',
      phases: 'not-array',
      summary: { decayEvaluated: 0, decayTransitioned: 0, promotionEvaluated: 0, promotionPromoted: 0, symbolsEnriched: 0, symbolsDeferred: 0 },
      errors: [],
    });
    expect(errors).toContain('phases must be an array');
  });

  it('validates phase status enum', () => {
    const errors = validateReport({
      runId: 'x',
      startedAt: '2024-01-01',
      completedAt: null,
      status: 'running',
      phases: [{ name: 'test', status: 'invalid', startedAt: null, completedAt: null }],
      summary: { decayEvaluated: 0, decayTransitioned: 0, promotionEvaluated: 0, promotionPromoted: 0, symbolsEnriched: 0, symbolsDeferred: 0 },
      errors: [],
    });
    expect(errors.some(e => e.includes('phase.status'))).toBe(true);
  });

  it('requires summary with numeric fields', () => {
    const errors = validateReport({
      runId: 'x',
      startedAt: '2024-01-01',
      completedAt: null,
      status: 'running',
      phases: [],
      summary: { decayEvaluated: 'not-number' },
      errors: [],
    });
    expect(errors.some(e => e.includes('decayEvaluated'))).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════
// Checkpoint Persistence
// ═════════════════════════════════════════════════════════════════════

describe('checkpoint persistence', () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'checkpoint-'));
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('persists and loads a checkpoint report', async () => {
    const checkpointPath = join(tmpDir, 'checkpoint.json');
    const env = await makeTestEnv();

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      const loaded = await loadCheckpoint(checkpointPath);
      expect(loaded).not.toBeNull();
      expect(loaded!.runId).toBe(report.runId);
      expect(loaded!.status).toBe('completed');
    } finally {
      await env.cleanup();
    }
  });

  it('returns null for nonexistent checkpoint', async () => {
    const loaded = await loadCheckpoint(join(tmpDir, 'nonexistent.json'));
    expect(loaded).toBeNull();
  });

  it('returns null for malformed checkpoint', async () => {
    const path = join(tmpDir, 'malformed.json');
    const { writeFile } = await import('node:fs/promises');
    await writeFile(path, 'not-json', 'utf-8');
    const loaded = await loadCheckpoint(path);
    expect(loaded).toBeNull();
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep-Run Pipeline: Decay Integration
// ═════════════════════════════════════════════════════════════════════

describe('sleep-run pipeline: decay evaluation', () => {
  let env: Awaited<ReturnType<typeof makeTestEnv>>;

  beforeEach(async () => { env = await makeTestEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('evaluates decay on all non-purged memories', async () => {
    // Create memories with different ages
    const m1 = await env.store.store({ content: 'fresh', category: 'learning', tags: [] });
    const m2 = await env.store.store({ content: 'stale', category: 'learning', tags: [] });
    const m3 = await env.store.store({ content: 'old', category: 'learning', tags: [] });

    // Make m2 and m3 old enough to transition
    const old35 = new Date(Date.now() - 35 * MS_PER_DAY).toISOString();
    const old121 = new Date(Date.now() - 121 * MS_PER_DAY).toISOString();
    env.db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old35, m2.id);
    env.db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old121, m3.id);

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-decay-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.status).toBe('completed');
      expect(report.summary.decayEvaluated).toBe(3);
      expect(report.summary.decayTransitioned).toBeGreaterThanOrEqual(2);

      // Verify m2 transitioned to cooling
      const row2 = env.db.prepare('SELECT memory_state FROM memories WHERE id = ?').get(m2.id) as { memory_state: string };
      expect(row2.memory_state).toBe('cooling');

      // Verify m3 transitioned to dimmed
      const row3 = env.db.prepare('SELECT memory_state FROM memories WHERE id = ?').get(m3.id) as { memory_state: string };
      expect(row3.memory_state).toBe('dimmed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('excludes purged memories from evaluation', async () => {
    const mem = await env.store.store({ content: 'purged', category: 'learning', tags: [] });
    env.db.prepare("UPDATE memories SET memory_state = 'purged' WHERE id = ?").run(mem.id);

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-purged-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.summary.decayEvaluated).toBe(0);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('respects custom forgetting config', async () => {
    const mem = await env.store.store({ content: 'custom config', category: 'learning', tags: [] });
    const old15 = new Date(Date.now() - 15 * MS_PER_DAY).toISOString();
    env.db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old15, mem.id);

    // Custom config: cooling after 10 days (instead of default 30)
    const customConfig = { ...DEFAULT_FORGETTING_CONFIG, coolingAfterDays: 10 };

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-custom-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath, forgettingConfig: customConfig },
      );

      expect(report.summary.decayTransitioned).toBe(1);
      const row = env.db.prepare('SELECT memory_state FROM memories WHERE id = ?').get(mem.id) as { memory_state: string };
      expect(row.memory_state).toBe('cooling');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep-Run Pipeline: Promotion Pass
// ═════════════════════════════════════════════════════════════════════

describe('sleep-run pipeline: promotion pass', () => {
  let env: Awaited<ReturnType<typeof makeTestEnv>>;

  beforeEach(async () => { env = await makeTestEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('runs promotion on eligible memories', async () => {
    // Create memories eligible for promotion (critical category, 2+ in same project)
    const m1 = await env.store.store({ content: 'security issue', category: 'security', tags: [] });
    const m2 = await env.store.store({ content: 'security issue 2', category: 'security', tags: [] });
    env.db.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id IN (?, ?)").run(m1.id, m2.id);

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-promo-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath, promotionBatchSize: 100 },
      );

      expect(report.status).toBe('completed');
      expect(report.summary.promotionEvaluated).toBeGreaterThanOrEqual(2);
      // At least one should be promoted (critical category rule)
      expect(report.summary.promotionPromoted).toBeGreaterThanOrEqual(1);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('skips non-eligible memories', async () => {
    // Single memory in non-critical category → no promotion
    await env.store.store({ content: 'learning note', category: 'learning', tags: [] });

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-promo-skip-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.summary.promotionPromoted).toBe(0);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('respects promotionBatchSize limit', async () => {
    // Create 10 eligible memories
    for (let i = 0; i < 10; i++) {
      const m = await env.store.store({ content: `perf issue ${i}`, category: 'performance', tags: [] });
      env.db.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m.id);
    }

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-promo-batch-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath, promotionBatchSize: 5 },
      );

      // Should evaluate at most 5 candidates
      expect(report.summary.promotionEvaluated).toBeLessThanOrEqual(5);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep-Run Pipeline: Full Orchestration
// ═════════════════════════════════════════════════════════════════════

describe('sleep-run pipeline: full orchestration', () => {
  let env: Awaited<ReturnType<typeof makeTestEnv>>;

  beforeEach(async () => { env = await makeTestEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('executes all phases in order', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-full-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.status).toBe('completed');
      expect(report.phases).toHaveLength(6);
      expect(report.phases[0].name).toBe('lease_acquire');
      expect(report.phases[0].status).toBe('completed');
      expect(report.phases[1].name).toBe('refresh_context');
      expect(report.phases[2].name).toBe('decay_evaluation');
      expect(report.phases[3].name).toBe('promotion_pass');
      expect(report.phases[4].name).toBe('sync_context');
      expect(report.phases[5].name).toBe('lease_release');
      expect(report.phases[5].status).toBe('completed');

      // All phases should have timestamps
      for (const phase of report.phases) {
        expect(phase.startedAt).toBeTruthy();
        expect(phase.completedAt).toBeTruthy();
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('calls refreshContext callback when provided', async () => {
    let refreshCalled = false;
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-refresh-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        {
          store: env.store,
          db: env.db,
          refreshContext: async () => { refreshCalled = true; },
        },
        { checkpointPath },
      );

      expect(refreshCalled).toBe(true);
      const refreshPhase = report.phases.find(p => p.name === 'refresh_context');
      expect(refreshPhase?.status).toBe('completed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('calls syncContext callback when provided', async () => {
    let syncCalled = false;
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-sync-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        {
          store: env.store,
          db: env.db,
          syncContext: async () => { syncCalled = true; },
        },
        { checkpointPath },
      );

      expect(syncCalled).toBe(true);
      const syncPhase = report.phases.find(p => p.name === 'sync_context');
      expect(syncPhase?.status).toBe('completed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('continues pipeline when refreshContext fails', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-refresh-fail-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        {
          store: env.store,
          db: env.db,
          refreshContext: async () => { throw new Error('refresh failed'); },
        },
        { checkpointPath },
      );

      // Pipeline should continue despite refresh failure
      expect(report.status).toBe('completed');
      const refreshPhase = report.phases.find(p => p.name === 'refresh_context');
      expect(refreshPhase?.status).toBe('failed');
      expect(report.errors.some(e => e.includes('refresh failed'))).toBe(true);

      // Decay should still run
      const decayPhase = report.phases.find(p => p.name === 'decay_evaluation');
      expect(decayPhase?.status).toBe('completed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('continues pipeline when syncContext fails', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-sync-fail-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        {
          store: env.store,
          db: env.db,
          syncContext: async () => { throw new Error('sync failed'); },
        },
        { checkpointPath },
      );

      expect(report.status).toBe('completed');
      const syncPhase = report.phases.find(p => p.name === 'sync_context');
      expect(syncPhase?.status).toBe('failed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('fails pipeline when decay evaluation fails', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-decay-fail-'));
    const checkpointPath = join(tmpDir, 'report.json');

    // Close the store to force a failure
    env.store.close();

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.status).toBe('failed');
      const decayPhase = report.phases.find(p => p.name === 'decay_evaluation');
      expect(decayPhase?.status).toBe('failed');
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('uses custom lease holder and TTL', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-lease-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        {
          checkpointPath,
          leaseHolder: 'custom-holder',
          leaseTTLMs: 60_000,
        },
      );

      expect(report.status).toBe('completed');
      const leasePhase = report.phases.find(p => p.name === 'lease_acquire');
      expect(leasePhase?.result).toMatchObject({ holder: 'custom-holder', ttlMs: 60_000 });
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});

// ═════════════════════════════════════════════════════════════════════
// Sleep-Run Pipeline: Report Structure
// ═════════════════════════════════════════════════════════════════════

describe('sleep-run pipeline: report structure', () => {
  let env: Awaited<ReturnType<typeof makeTestEnv>>;

  beforeEach(async () => { env = await makeTestEnv(); });
  afterEach(async () => { await env.cleanup(); });

  it('includes runId and timestamps', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-report-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.runId).toBeTruthy();
      expect(report.startedAt).toBeTruthy();
      expect(report.completedAt).toBeTruthy();
      expect(new Date(report.completedAt!).getTime()).toBeGreaterThanOrEqual(
        new Date(report.startedAt).getTime(),
      );
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('validates against the schema', async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-validate-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      const errors = validateReport(report);
      expect(errors).toEqual([]);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });

  it('summary reflects actual processing counts', async () => {
    // Create 2 memories: one fresh, one stale
    await env.store.store({ content: 'fresh', category: 'learning', tags: [] });
    const stale = await env.store.store({ content: 'stale', category: 'learning', tags: [] });
    const old = new Date(Date.now() - 35 * MS_PER_DAY).toISOString();
    env.db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old, stale.id);

    const tmpDir = await mkdtemp(join(tmpdir(), 'sleep-summary-'));
    const checkpointPath = join(tmpDir, 'report.json');

    try {
      const report = await executeSleepRun(
        { store: env.store, db: env.db },
        { checkpointPath },
      );

      expect(report.summary.decayEvaluated).toBe(2);
      expect(report.summary.decayTransitioned).toBe(1);
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  });
});
