// tests/pr9-validation.test.ts
//
// T-010: compare-backends validation logic — vitest tests
// Tests the core comparison function that validates migration completeness.

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { embeddingToBlob } from '../src/shared/vector-utils.js';
import { compareBackends, type ComparisonResult, type ComparisonQueryResult } from '../src/compare-backends.js';

// ── Helpers ────────────────────────────────────────────────────────

function createEngramDb(dir: string, rows: Array<{ id: number; title: string; content: string; type: string; embedding?: number[] }>): string {
  const dbPath = join(dir, 'engram.db');
  const db = new DatabaseSync(dbPath);
  db.exec(`
    CREATE TABLE observations (
      id INTEGER PRIMARY KEY,
      sync_id TEXT,
      session_id TEXT,
      type TEXT,
      title TEXT,
      content TEXT,
      tool_name TEXT,
      project TEXT,
      scope TEXT,
      topic_key TEXT,
      normalized_hash TEXT,
      revision_count INTEGER,
      duplicate_count INTEGER,
      last_seen_at TEXT,
      pinned INTEGER,
      created_at TEXT,
      updated_at TEXT,
      deleted_at TEXT,
      review_after TEXT,
      expires_at TEXT,
      embedding BLOB,
      embedding_model TEXT,
      embedding_created_at TEXT
    );
  `);

  const insert = db.prepare(
    `INSERT INTO observations (id, sync_id, session_id, type, title, content, scope, created_at, updated_at, embedding)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );

  for (const row of rows) {
    const emb = row.embedding ?? generateDeterministicVector(row.content);
    insert.run(
      row.id, `sync-${row.id}`, 'test-session', row.type,
      row.title, row.content, 'project',
      '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
      embeddingToBlob(emb),
    );
  }

  db.close();
  return dbPath;
}

function createPmcDb(dir: string, rows: Array<{ id: string; content: string; category: string; embedding?: number[] }>): string {
  const dbPath = join(dir, 'pmc.db');
  const db = new DatabaseSync(dbPath);
  db.exec(`
    CREATE TABLE memories (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      category TEXT NOT NULL,
      tags TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      access_count INTEGER NOT NULL DEFAULT 0,
      last_accessed_at TEXT NOT NULL,
      version INTEGER NOT NULL DEFAULT 1,
      status TEXT NOT NULL DEFAULT 'active'
    );
    CREATE TABLE memory_embeddings (
      id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
      embedding BLOB NOT NULL
    );
  `);

  const insertMem = db.prepare(
    `INSERT INTO memories (id, content, category, tags, created_at, updated_at, last_accessed_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  );
  const insertEmb = db.prepare(
    `INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)`
  );

  for (const row of rows) {
    const emb = row.embedding ?? generateDeterministicVector(row.content);
    const now = '2026-01-01T00:00:00Z';
    insertMem.run(row.id, row.content, row.category, '[]', now, now, now);
    insertEmb.run(row.id, embeddingToBlob(emb));
  }

  db.close();
  return dbPath;
}

/** Generate a deterministic pseudo-vector from content (matches MockEmbedder pattern). */
function generateDeterministicVector(content: string): number[] {
  const dim = 32;
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < content.length; i++) {
    vec[i % dim] += content.charCodeAt(i) / 1000;
  }
  // Normalize
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return mag > 0 ? vec.map(v => v / mag) : vec;
}

// ── Tests ──────────────────────────────────────────────────────────

describe('compareBackends', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'compare-backends-'));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('returns overlap_count > 0 and zero drift when both DBs have identical content', async () => {
    const content = 'Fixed authentication bug in middleware';
    const vec = generateDeterministicVector(content);

    const engramPath = createEngramDb(tmpDir, [
      { id: 1, title: 'Fixed auth bug', content, type: 'bugfix', embedding: vec },
    ]);
    const pmcPath = createPmcDb(tmpDir, [
      { id: 'mem-1', content: `**What**: Fixed auth bug\n\n${content}`, category: 'bug-fix', embedding: vec },
    ]);

    const result = await compareBackends({ engramDbPath: engramPath, pmcDbPath: pmcPath, queries: ['auth'] });

    expect(result.overlap_count).toBeGreaterThan(0);
    expect(result.engram_only_count).toBe(0);
    expect(result.pmc_only_count).toBe(0);
  });

  it('counts engram-only memories in engram_only_count (drift)', async () => {
    const content = 'Engram-only memory about config';
    const vec = generateDeterministicVector(content);

    const engramPath = createEngramDb(tmpDir, [
      { id: 1, title: 'Config memory', content, type: 'config', embedding: vec },
    ]);
    const pmcPath = createPmcDb(tmpDir, []); // Empty PMC DB

    const result = await compareBackends({ engramDbPath: engramPath, pmcDbPath: pmcPath, queries: ['config'] });

    expect(result.engram_only_count).toBeGreaterThan(0);
  });

  it('counts pmc-only memories in pmc_only_count (new since migration)', async () => {
    const content = 'PMC-only memory about decisions';
    const vec = generateDeterministicVector(content);

    const engramPath = createEngramDb(tmpDir, []); // Empty Engram DB
    const pmcPath = createPmcDb(tmpDir, [
      { id: 'mem-1', content, category: 'architecture', embedding: vec },
    ]);

    const result = await compareBackends({ engramDbPath: engramPath, pmcDbPath: pmcPath, queries: ['decision'] });

    expect(result.pmc_only_count).toBeGreaterThan(0);
  });

  it('--window-minutes N filters to recent memories only', async () => {
    // Old memory (2020)
    const engramPath = createEngramDb(tmpDir, [
      { id: 1, title: 'Old bug', content: 'Old bug fix', type: 'bugfix' },
    ]);
    // Override created_at to be old
    const db = new DatabaseSync(engramPath);
    db.exec("UPDATE observations SET created_at = '2020-01-01T00:00:00Z'");
    db.close();

    const pmcPath = createPmcDb(tmpDir, []);

    const result = await compareBackends({
      engramDbPath: engramPath,
      pmcDbPath: pmcPath,
      queries: ['bug'],
      windowMinutes: 5, // Only last 5 minutes
    });

    // Old memories should be filtered out
    expect(result.engram_only_count).toBe(0);
    expect(result.total_engram_memories).toBe(0);
  });

  it('includes side-by-side top-5 results for each query', async () => {
    const engramPath = createEngramDb(tmpDir, [
      { id: 1, title: 'Auth bug', content: 'Fixed auth', type: 'bugfix' },
    ]);
    const pmcPath = createPmcDb(tmpDir, [
      { id: 'mem-1', content: 'Fixed auth issue', category: 'bug-fix' },
    ]);

    const result = await compareBackends({ engramDbPath: engramPath, pmcDbPath: pmcPath, queries: ['auth'] });

    expect(result.results).toHaveLength(1);
    expect(result.results[0].query).toBe('auth');
    expect(result.results[0].engram_top5).toBeDefined();
    expect(result.results[0].pmc_top5).toBeDefined();
    expect(Array.isArray(result.results[0].engram_top5)).toBe(true);
    expect(Array.isArray(result.results[0].pmc_top5)).toBe(true);
  });

  it('computes avg_cosine_matched for matched pairs', async () => {
    const content = 'Same content for cosine check';
    const vec = generateDeterministicVector(content);

    const engramPath = createEngramDb(tmpDir, [
      { id: 1, title: 'Cosine test', content, type: 'bugfix', embedding: vec },
    ]);
    const pmcPath = createPmcDb(tmpDir, [
      { id: 'mem-1', content: `**What**: Cosine test\n\n${content}`, category: 'bug-fix', embedding: vec },
    ]);

    const result = await compareBackends({ engramDbPath: engramPath, pmcDbPath: pmcPath, queries: ['cosine'] });

    if (result.overlap_count > 0) {
      expect(result.avg_cosine_matched).toBeGreaterThan(0);
      expect(result.avg_cosine_matched).toBeLessThanOrEqual(1.01); // Allow small float imprecision
    }
  });
});
