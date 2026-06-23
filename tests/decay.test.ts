import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import {
  evaluateActivationScore,
  transitionDecayState,
  DEFAULT_FORGETTING_CONFIG,
  DECAY_STAGE_ORDER,
} from '../src/decay.js';
import type { ForgettingConfig, DecayState } from '../src/decay.js';
import { MemoryNotFoundError } from '../src/errors.js';

const MS_PER_DAY = 86_400_000;

// ── PR 2a: Gradual forgetting schema & state machine ───────────────
//
// Strict TDD for Phase 2a:
//   - Task 2.1 RED: schema extensions (tables, columns, indexes, FK policy)
//   - Task 2.3 RED: decay state transitions (pure decision logic)
//   - Task 2.4 RED: activation scoring + store-wired transition logging

// ── Activation scoring (Task 2.4) ──────────────────────────────────

describe('evaluateActivationScore', () => {
  it('returns 1.0 for active', () => {
    expect(evaluateActivationScore('active')).toBe(1.0);
  });

  it('decreases by ~20% per stage', () => {
    expect(evaluateActivationScore('cooling')).toBeCloseTo(0.8, 5);
    expect(evaluateActivationScore('dimmed')).toBeCloseTo(0.6, 5);
    expect(evaluateActivationScore('shadow')).toBeCloseTo(0.4, 5);
    expect(evaluateActivationScore('gist_only')).toBeCloseTo(0.2, 5);
  });

  it('returns 0.0 for purged', () => {
    expect(evaluateActivationScore('purged')).toBe(0.0);
  });

  it('is monotonically decreasing across the stage order', () => {
    const scores = DECAY_STAGE_ORDER.map((s) => evaluateActivationScore(s));
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeLessThan(scores[i - 1]);
    }
  });
});

// ── Decay state transitions (Task 2.3) ─────────────────────────────

describe('transitionDecayState (pure decision logic)', () => {
  const cfg: ForgettingConfig = DEFAULT_FORGETTING_CONFIG;
  // Default boundaries (cumulative days since last reinforcement):
  //   cooling=30, dimmed=30+90=120, shadow=120+180=300,
  //   gist_only=300+365=665, purged=665+90=755
  const now = Date.now();
  const iso = (daysAgo: number) => new Date(now - daysAgo * MS_PER_DAY).toISOString();

  function run(daysAgo: number, currentState: DecayState) {
    return transitionDecayState({
      memoryId: 'm1',
      currentState,
      lastReinforcedAt: iso(daysAgo),
      config: cfg,
      now,
    });
  }

  it('advances active → cooling after coolingAfterDays', () => {
    expect(run(29, 'active').newState).toBe('active');
    expect(run(31, 'active').newState).toBe('cooling');
  });

  it('advances cooling → dimmed after additional dimmedAfterDays', () => {
    expect(run(119, 'active').newState).toBe('cooling');
    expect(run(121, 'active').newState).toBe('dimmed');
  });

  it('advances dimmed → shadow after additional shadowAfterDays', () => {
    expect(run(299, 'active').newState).toBe('dimmed');
    expect(run(301, 'active').newState).toBe('shadow');
  });

  it('advances shadow → gist_only after additional gistCandidateAfterDays', () => {
    expect(run(664, 'active').newState).toBe('shadow');
    expect(run(666, 'active').newState).toBe('gist_only');
  });

  it('advances gist_only → purged after additional gistOnlyRetentionDays', () => {
    expect(run(754, 'active').newState).toBe('gist_only');
    expect(run(756, 'active').newState).toBe('purged');
  });

  it('jumps multiple stages when a lot of time elapsed', () => {
    expect(run(400, 'active').newState).toBe('shadow');
    expect(run(800, 'active').newState).toBe('purged');
  });

  it('does not regress: a shadow memory stays shadow below the gist threshold', () => {
    const r = run(50, 'shadow');
    expect(r.newState).toBe('shadow');
    expect(r.transitioned).toBe(false);
  });

  it('reports transitioned=true only when the state actually changes', () => {
    expect(run(31, 'active').transitioned).toBe(true);
    expect(run(10, 'active').transitioned).toBe(false);
  });

  it('activation score reflects the new stage', () => {
    expect(run(31, 'active').newScore).toBeCloseTo(0.8, 5);
    expect(run(301, 'active').newScore).toBeCloseTo(0.4, 5);
  });

  it('resets the transition timer when lastReinforcedAt is recent', () => {
    // Memory was created long ago but reinforced recently → stays active.
    const r = transitionDecayState({
      memoryId: 'm1',
      currentState: 'shadow',
      lastReinforcedAt: new Date(now - 3 * MS_PER_DAY).toISOString(),
      config: cfg,
      now,
    });
    // Reinforcement-reset path: a recently-reinforced memory must NOT advance.
    // (Full revert-to-active is PR 2b reinforcement; PR 2a only guarantees no
    // forward advance while reinforced.) Here elapsed=3d < 30d → no advance.
    expect(r.transitioned).toBe(false);
    expect(r.newState).toBe('shadow');
  });

  it('invokes the optional log writer with the transition record', () => {
    const recorded: Parameters<Parameters<typeof transitionDecayState>[1]>[] = [];
    transitionDecayState(
      { memoryId: 'm1', currentState: 'active', lastReinforcedAt: iso(31), config: cfg, now },
      (rec) => recorded.push(rec),
    );
    expect(recorded).toHaveLength(1);
    expect(recorded[0].memory_id).toBe('m1');
    expect(recorded[0].from_state).toBe('active');
    expect(recorded[0].to_state).toBe('cooling');
    expect(recorded[0].score_before).toBe(1.0);
    expect(recorded[0].score_after).toBeCloseTo(0.8, 5);
  });

  it('does not call the log writer when there is no transition', () => {
    let calls = 0;
    transitionDecayState(
      { memoryId: 'm1', currentState: 'active', lastReinforcedAt: iso(10), config: cfg, now },
      () => calls++,
    );
    expect(calls).toBe(0);
  });
});

// ── Schema validation (Task 2.1) ───────────────────────────────────

describe('gradual forgetting schema (PR 2a)', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'decay-schema-'));
    dbPath = join(tmpDir, 'test.db');
    const embedder = new MockEmbedder();
    store = new SqliteMemoryStore(dbPath, embedder);
    await store.initialize();
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  function columnsOf(table: string): string[] {
    const rows = verify.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
    return rows.map((r) => r.name);
  }

  function indexesOn(table: string): string[] {
    const rows = verify.prepare(`PRAGMA index_list(${table})`).all() as { name: string }[];
    return rows.map((r) => r.name);
  }

  it('extends memories with activation_score, memory_state, last_reinforced_at', () => {
    const cols = columnsOf('memories');
    expect(cols).toContain('activation_score');
    expect(cols).toContain('memory_state');
    expect(cols).toContain('last_reinforced_at');
  });

  it('applies defaults: activation_score=1.0, memory_state=active', async () => {
    const mem = await store.store({
      content: 'schema default check',
      category: 'learning',
      tags: [],
    });
    const row = verify
      .prepare('SELECT activation_score, memory_state, last_reinforced_at FROM memories WHERE id = ?')
      .get(mem.id) as { activation_score: number; memory_state: string; last_reinforced_at: string | null };
    expect(row.activation_score).toBe(1.0);
    expect(row.memory_state).toBe('active');
    // last_reinforced_at may be NULL until reinforced (PR 2b); transition falls back to created_at.
  });

  it('creates the memory_gists table with required columns', () => {
    const cols = columnsOf('memory_gists');
    for (const c of [
      'id', 'original_id', 'original_source', 'kind', 'core_fact',
      'why_it_matters', 'revive_triggers', 'non_forgettable',
      'created_at', 'updated_at', 'original_pruned_at', 'status',
    ]) {
      expect(cols).toContain(c);
    }
  });

  it('creates the memory_decay_log table with required columns', () => {
    const cols = columnsOf('memory_decay_log');
    for (const c of [
      'id', 'memory_id', 'from_state', 'to_state', 'reason',
      'score_before', 'score_after', 'created_at',
    ]) {
      expect(cols).toContain(c);
    }
  });

  it('creates indexes on memory_gists (original_id, status, kind)', () => {
    const idx = indexesOn('memory_gists');
    expect(idx.some((n) => n.includes('original_id'))).toBe(true);
    expect(idx.some((n) => n.includes('status'))).toBe(true);
    expect(idx.some((n) => n.includes('kind'))).toBe(true);
  });

  it('creates indexes on memory_decay_log (memory_id, created_at)', () => {
    const idx = indexesOn('memory_decay_log');
    expect(idx.some((n) => n.includes('memory_id'))).toBe(true);
    expect(idx.some((n) => n.includes('created_at'))).toBe(true);
  });

  it('makes memory_gists.original_id a NON-strict FK so the gist survives original purge', () => {
    // Insert a gist referencing a (never-existing) memory id, then ensure no
    // FK constraint blocks it and that deleting a real memory leaves the gist.
    verify.prepare(
      `INSERT INTO memory_gists (id, original_id, original_source, kind, core_fact, why_it_matters, revive_triggers, non_forgettable, created_at, updated_at, status)
       VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
    ).run(
      'g1', 'mem-xyz', 'memory', 'fact', 'core', 'matters',
      JSON.stringify(['trigger-a']), 1,
      new Date().toISOString(), new Date().toISOString(), 'active',
    );

    // Insert a real memory, link a gist to it, delete the memory, gist survives.
    const mem = verify.prepare(
      `INSERT INTO memories (id, content, category, tags, created_at, updated_at, last_accessed_at) VALUES (?,?,?,?,?,?,?)`,
    ).run('real-mem-1', 'c', 'learning', '[]', new Date().toISOString(), new Date().toISOString(), new Date().toISOString());
    expect(mem.changes).toBe(1);

    verify.prepare(
      `INSERT INTO memory_gists (id, original_id, original_source, kind, core_fact, why_it_matters, revive_triggers, non_forgettable, created_at, updated_at, status)
       VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
    ).run(
      'g2', 'real-mem-1', 'memory', 'fact', 'core2', 'matters2',
      JSON.stringify([]), 1,
      new Date().toISOString(), new Date().toISOString(), 'active',
    );

    verify.prepare('DELETE FROM memories WHERE id = ?').run('real-mem-1');

    const surviving = verify
      .prepare('SELECT id FROM memory_gists WHERE original_id = ?')
      .all('real-mem-1') as { id: string }[];
    expect(surviving.map((r) => r.id)).toEqual(['g2']);
  });
});

// ── Store-wired decay transition + logging (Task 2.4 integration) ──

describe('SqliteMemoryStore.transitionDecayStateForMemory', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'decay-store-'));
    dbPath = join(tmpDir, 'test.db');
    const embedder = new MockEmbedder();
    store = new SqliteMemoryStore(dbPath, embedder);
    await store.initialize();
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('transitions a memory and persists memory_state + activation_score', async () => {
    const mem = await store.store({
      content: 'will decay',
      category: 'learning',
      tags: [],
    });
    // Force last_reinforced_at to 35 days ago.
    const old = new Date(Date.now() - 35 * MS_PER_DAY).toISOString();
    verify.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old, mem.id);

    const result = store.transitionDecayStateForMemory(mem.id, DEFAULT_FORGETTING_CONFIG);

    expect(result.newState).toBe('cooling');
    expect(result.transitioned).toBe(true);

    const row = verify
      .prepare('SELECT memory_state, activation_score FROM memories WHERE id = ?')
      .get(mem.id) as { memory_state: string; activation_score: number };
    expect(row.memory_state).toBe('cooling');
    expect(row.activation_score).toBeCloseTo(0.8, 5);
  });

  it('inserts a transition record into memory_decay_log', async () => {
    const mem = await store.store({
      content: 'will log decay',
      category: 'learning',
      tags: [],
    });
    const old = new Date(Date.now() - 301 * MS_PER_DAY).toISOString(); // → shadow
    verify.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(old, mem.id);

    store.transitionDecayStateForMemory(mem.id, DEFAULT_FORGETTING_CONFIG);

    const logs = verify
      .prepare('SELECT from_state, to_state, score_before, score_after, reason FROM memory_decay_log WHERE memory_id = ?')
      .all(mem.id) as { from_state: string; to_state: string; score_before: number; score_after: number; reason: string }[];
    expect(logs).toHaveLength(1);
    expect(logs[0].from_state).toBe('active');
    expect(logs[0].to_state).toBe('shadow');
    expect(logs[0].score_before).toBe(1.0);
    expect(logs[0].score_after).toBeCloseTo(0.4, 5);
    expect(logs[0].reason).toBeTruthy();
  });

  it('does not log when the memory is within its active window', async () => {
    const mem = await store.store({
      content: 'fresh memory',
      category: 'learning',
      tags: [],
    });
    // last_reinforced_at defaults to created_at (recent) → no transition.
    const result = store.transitionDecayStateForMemory(mem.id, DEFAULT_FORGETTING_CONFIG);
    expect(result.transitioned).toBe(false);

    const count = verify
      .prepare('SELECT COUNT(*) AS n FROM memory_decay_log WHERE memory_id = ?')
      .get(mem.id) as { n: number };
    expect(count.n).toBe(0);
  });

  it('falls back to created_at when last_reinforced_at is NULL', async () => {
    const mem = await store.store({
      content: 'null reinforced',
      category: 'learning',
      tags: [],
    });
    // Make created_at look old (and keep last_reinforced_at NULL).
    const old = new Date(Date.now() - 121 * MS_PER_DAY).toISOString();
    verify.prepare('UPDATE memories SET created_at = ?, last_reinforced_at = NULL WHERE id = ?').run(old, mem.id);

    const result = store.transitionDecayStateForMemory(mem.id, DEFAULT_FORGETTING_CONFIG);
    expect(result.newState).toBe('dimmed');
  });
});
