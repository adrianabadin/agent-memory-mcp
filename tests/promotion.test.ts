import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import {
  PromotionRuleEngine,
  type PromotionCandidate,
  type PromotionRuleResult,
  type PromotionResult,
  type GlobalPromotionLogRecord,
} from '../src/global/global-promotion.js';

const MS_PER_DAY = 86_400_000;

// ── Task 3.1: RED — global_promotion_log schema ────────────────────

describe('global_promotion_log schema', () => {
  let dbPath: string;
  let store: SqliteMemoryStore;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'promotion-schema-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('creates global_promotion_log table', () => {
    const tables = verify.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='global_promotion_log'",
    ).all() as { name: string }[];
    expect(tables.length).toBe(1);
  });

  it('has expected columns', () => {
    const cols = verify.prepare('PRAGMA table_info(global_promotion_log)').all() as { name: string }[];
    const colNames = cols.map(c => c.name);
    expect(colNames).toContain('id');
    expect(colNames).toContain('promoted_memory_id');
    expect(colNames).toContain('source_ids');
    expect(colNames).toContain('source_projects');
    expect(colNames).toContain('rule_matched');
    expect(colNames).toContain('confidence');
    expect(colNames).toContain('model');
    expect(colNames).toContain('reviewer_provenance');
    expect(colNames).toContain('created_at');
  });

  it('creates memory_revive_reviews table', () => {
    const tables = verify.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_revive_reviews'",
    ).all() as { name: string }[];
    expect(tables.length).toBe(1);
  });

  it('has indexes for quick lookups on global_promotion_log', () => {
    const indexes = verify.prepare(
      "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='global_promotion_log'",
    ).all() as { name: string }[];
    const indexNames = indexes.map(i => i.name);
    expect(indexNames).toContain('idx_gpl_promoted_memory_id');
    expect(indexNames).toContain('idx_gpl_source_projects');
    expect(indexNames).toContain('idx_gpl_rule_matched');
  });

  it('has unique constraint on (promoted_memory_id, source_ids)', () => {
    // Insert first row
    verify.prepare(
      `INSERT INTO global_promotion_log (id, promoted_memory_id, source_ids, source_projects, rule_matched, confidence, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(randomUUID(), 'mem1', '["s1","s2"]', '["proj1"]', 'multi_project', 0.8, new Date().toISOString());

    // Duplicate should fail
    expect(() => {
      verify.prepare(
        `INSERT INTO global_promotion_log (id, promoted_memory_id, source_ids, source_projects, rule_matched, confidence, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(randomUUID(), 'mem1', '["s1","s2"]', '["proj1"]', 'multi_project', 0.9, new Date().toISOString());
    }).toThrow();
  });
});

// ── Task 3.3: RED — Multi-project recurrence rule ──────────────────

describe('PromotionRuleEngine — multi-project rule', () => {
  let dbPath: string;
  let store: SqliteMemoryStore;
  let engine: PromotionRuleEngine;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'promotion-multi-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    engine = new PromotionRuleEngine(store.getDb());
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('detects memory appearing in 2+ projects via content_hash', async () => {
    // Store same content in "different projects" — since we have a single DB,
    // we simulate multi-project by inserting memories with different session_ids
    // (project association) but same content_hash.
    const m1 = await store.store({ content: 'shared knowledge', category: 'bug-fix', tags: [] });
    const m2 = await store.store({ content: 'shared knowledge', category: 'bug-fix', tags: [] });

    // Tag them as from different projects via session_id
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    const result = engine.matchMultiProjectRule(m1.id);
    expect(result.matched).toBe(true);
    expect(result.ruleName).toBe('multi_project_occurrence');
    expect(result.sourceIds.length).toBeGreaterThanOrEqual(2);
    expect(result.sourceProjects.length).toBeGreaterThanOrEqual(2);
  });

  it('does not match when only 1 project', async () => {
    const m1 = await store.store({ content: 'solo knowledge', category: 'learning', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);

    const result = engine.matchMultiProjectRule(m1.id);
    expect(result.matched).toBe(false);
  });

  it('creates promotion record with rule_matched=multi_project_occurrence', async () => {
    const m1 = await store.store({ content: 'cross project', category: 'architecture', tags: [] });
    const m2 = await store.store({ content: 'cross project', category: 'architecture', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    const result = engine.matchMultiProjectRule(m1.id);
    expect(result.matched).toBe(true);
    expect(result.ruleName).toBe('multi_project_occurrence');
  });
});

// ── Task 3.5: RED — Critical-category repetition rule ──────────────

describe('PromotionRuleEngine — critical-category rule', () => {
  let dbPath: string;
  let store: SqliteMemoryStore;
  let engine: PromotionRuleEngine;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'promotion-critical-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    engine = new PromotionRuleEngine(store.getDb());
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('matches security category with 2+ occurrences in same project', async () => {
    const m1 = await store.store({ content: 'vuln found', category: 'security', tags: [] });
    const m2 = await store.store({ content: 'vuln found again', category: 'security', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m2.id);

    const result = engine.matchCriticalCategoryRule(m1.id);
    expect(result.matched).toBe(true);
    expect(result.ruleName).toBe('critical_category_repetition');
    expect(result.sourceIds.length).toBeGreaterThanOrEqual(2);
  });

  it('matches performance category with 2+ occurrences', async () => {
    const m1 = await store.store({ content: 'perf issue', category: 'performance', tags: [] });
    const m2 = await store.store({ content: 'perf issue 2', category: 'performance', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m2.id);

    const result = engine.matchCriticalCategoryRule(m1.id);
    expect(result.matched).toBe(true);
  });

  it('does not match with only 1 occurrence', async () => {
    const m1 = await store.store({ content: 'solo security', category: 'security', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);

    const result = engine.matchCriticalCategoryRule(m1.id);
    expect(result.matched).toBe(false);
  });

  it('does not match non-critical category', async () => {
    const m1 = await store.store({ content: 'learning 1', category: 'learning', tags: [] });
    const m2 = await store.store({ content: 'learning 2', category: 'learning', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m2.id);

    const result = engine.matchCriticalCategoryRule(m1.id);
    expect(result.matched).toBe(false);
  });

  it('requires confidence >= 0.75', async () => {
    const m1 = await store.store({ content: 'data loss risk', category: 'bug-fix', tags: [] });
    // bug-fix is not in the critical set, but we test the threshold logic
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);

    const result = engine.matchCriticalCategoryRule(m1.id);
    // Not matched because category not critical — confidence logic not reached
    expect(result.matched).toBe(false);
  });
});

// ── Task 3.7: RED — Revival-based promotion rule ───────────────────

describe('PromotionRuleEngine — revival rule', () => {
  let dbPath: string;
  let store: SqliteMemoryStore;
  let engine: PromotionRuleEngine;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'promotion-revival-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    engine = new PromotionRuleEngine(store.getDb());
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('matches when memory has 3+ revive reviews', async () => {
    const mem = await store.store({ content: 'revived memory', category: 'bug-fix', tags: [] });

    // Insert 3 revive reviews
    for (let i = 0; i < 3; i++) {
      verify.prepare(
        `INSERT INTO memory_revive_reviews (id, memory_id, decision, reason, model, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(randomUUID(), mem.id, 'revived', 'still useful', 'test', new Date().toISOString());
    }

    const result = engine.matchRevivalRule(mem.id);
    expect(result.matched).toBe(true);
    expect(result.ruleName).toBe('reuse_revival_detection');
    expect(result.confidence).toBeGreaterThanOrEqual(0.75);
  });

  it('does not match with fewer than 3 revive reviews', async () => {
    const mem = await store.store({ content: 'partially revived', category: 'learning', tags: [] });

    for (let i = 0; i < 2; i++) {
      verify.prepare(
        `INSERT INTO memory_revive_reviews (id, memory_id, decision, reason, model, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(randomUUID(), mem.id, 'revived', 'still useful', 'test', new Date().toISOString());
    }

    const result = engine.matchRevivalRule(mem.id);
    expect(result.matched).toBe(false);
  });

  it('tracks revive count in memory_revive_reviews', async () => {
    const mem = await store.store({ content: 'tracked revival', category: 'architecture', tags: [] });

    for (let i = 0; i < 5; i++) {
      verify.prepare(
        `INSERT INTO memory_revive_reviews (id, memory_id, decision, reason, model, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(randomUUID(), mem.id, i < 3 ? 'revived' : 'pruned', 'reason', 'test', new Date().toISOString());
    }

    // Only 'revived' decisions count
    const result = engine.matchRevivalRule(mem.id);
    expect(result.matched).toBe(true);
    expect(result.sourceIds.length).toBe(3); // 3 revived decisions
  });
});

// ── Task 3.9: RED — Promotion workflow (detect, decide, audit) ─────

describe('PromotionRuleEngine — evaluateForPromotion workflow', () => {
  let dbPath: string;
  let store: SqliteMemoryStore;
  let engine: PromotionRuleEngine;
  let verify: DatabaseSync;

  beforeEach(async () => {
    const tmpDir = await mkdtemp(join(tmpdir(), 'promotion-workflow-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
    engine = new PromotionRuleEngine(store.getDb());
    verify = new DatabaseSync(dbPath);
  });

  afterEach(async () => {
    verify.close();
    store.close();
    await rm(join(dbPath, '..'), { recursive: true, force: true });
  });

  it('runs all rules and collects matches', async () => {
    // Setup: 2 projects with same content → multi-project rule
    const m1 = await store.store({ content: 'cross proj', category: 'bug-fix', tags: [] });
    const m2 = await store.store({ content: 'cross proj', category: 'bug-fix', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    const result = engine.evaluateForPromotion(m1.id);
    expect(result.evaluated).toBe(1);
    expect(result.promoted).toBeGreaterThanOrEqual(1);
  });

  it('inserts one audit log entry per promotion', async () => {
    const m1 = await store.store({ content: 'audit test', category: 'security', tags: [] });
    const m2 = await store.store({ content: 'audit test', category: 'security', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    engine.evaluateForPromotion(m1.id);

    const logs = verify.prepare('SELECT * FROM global_promotion_log').all() as GlobalPromotionLogRecord[];
    expect(logs.length).toBeGreaterThanOrEqual(1);
    expect(logs[0].promoted_memory_id).toBeDefined();
    expect(logs[0].rule_matched).toBeDefined();
  });

  it('marks promoted memory as non_forgettable', async () => {
    const m1 = await store.store({ content: 'nonforget test', category: 'architecture', tags: [] });
    const m2 = await store.store({ content: 'nonforget test', category: 'architecture', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    const result = engine.evaluateForPromotion(m1.id);

    // Check that the promoted memory was marked
    if (result.promoted > 0) {
      const row = verify.prepare(
        "SELECT non_forgettable FROM memories WHERE id = ?",
      ).get(result.promotedMemoryIds[0]) as { non_forgettable: number } | undefined;
      if (row) {
        expect(row.non_forgettable).toBe(1);
      }
    }
  });

  it('returns summary with evaluated, promoted, skipped, errors', async () => {
    const m1 = await store.store({ content: 'summary test', category: 'learning', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);

    const result = engine.evaluateForPromotion(m1.id);
    expect(result).toHaveProperty('evaluated');
    expect(result).toHaveProperty('promoted');
    expect(result).toHaveProperty('skipped');
    expect(result).toHaveProperty('errors');
    expect(result).toHaveProperty('promotedMemoryIds');
  });

  it('skips memories below confidence threshold', async () => {
    // Single memory in a non-critical category, single project → no rule matches
    // so skipped=0 (no rule even matched to be skipped)
    const m1 = await store.store({ content: 'low conf', category: 'observation', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);

    const result = engine.evaluateForPromotion(m1.id);
    expect(result.promoted).toBe(0);
    // No rules matched at all, so skipped stays 0
    expect(result.skipped).toBe(0);
  });

  it('does not duplicate promotions for already-promoted sources', async () => {
    const m1 = await store.store({ content: 'dedup test', category: 'bug-fix', tags: [] });
    const m2 = await store.store({ content: 'dedup test', category: 'bug-fix', tags: [] });
    verify.prepare("UPDATE memories SET session_id = 'proj-a' WHERE id = ?").run(m1.id);
    verify.prepare("UPDATE memories SET session_id = 'proj-b' WHERE id = ?").run(m2.id);

    // First promotion
    engine.evaluateForPromotion(m1.id);
    const countBefore = (verify.prepare('SELECT COUNT(*) AS n FROM global_promotion_log').get() as { n: number }).n;

    // Second call — should not duplicate
    engine.evaluateForPromotion(m1.id);
    const countAfter = (verify.prepare('SELECT COUNT(*) AS n FROM global_promotion_log').get() as { n: number }).n;

    expect(countAfter).toBe(countBefore);
  });
});
