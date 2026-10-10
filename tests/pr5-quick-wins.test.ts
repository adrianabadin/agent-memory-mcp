import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { HardcopyMemoryStore } from '../src/hardcopy-store.js';
import { MockEmbedder, MockMemoryStore } from './mocks.js';
import { MemoryNotFoundError } from '../src/errors.js';

// ── Helpers ──────────────────────────────────────────────────────────

function tableColumns(db: DatabaseSync, table: string): string[] {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return rows.map(r => r.name);
}

function indexExists(db: DatabaseSync, indexName: string): boolean {
  const row = db.prepare(
    `SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?`,
  ).get(indexName);
  return !!row;
}

function userVersion(db: DatabaseSync): number {
  const row = db.prepare('PRAGMA user_version').get() as { user_version: number } | undefined;
  return row?.user_version ?? 0;
}

// ── T-001: Migration v8 — is_pinned column ───────────────────────────

describe('T-001: migration v8 — is_pinned column', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr5-migration-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('bumps user_version to 8 after initialize', () => {
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });

  it('adds is_pinned column to memories with default 0', () => {
    const db = new DatabaseSync(dbPath);
    const cols = tableColumns(db, 'memories');
    expect(cols).toContain('is_pinned');
    // Verify default value by inserting a row
    db.exec(`INSERT INTO memories (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version)
             VALUES ('test-pin-default', 'content', 'learning', '[]', '2025-01-01', '2025-01-01', 0, '2025-01-01', 1)`);
    const row = db.prepare('SELECT is_pinned FROM memories WHERE id = ?').get('test-pin-default') as { is_pinned: number };
    expect(row.is_pinned).toBe(0);
    db.close();
  });

  it('creates idx_memories_is_pinned index', () => {
    const db = new DatabaseSync(dbPath);
    expect(indexExists(db, 'idx_memories_is_pinned')).toBe(true);
    db.close();
  });

  it('v8 migration is idempotent (re-initialize stays at v8, no error)', async () => {
    await expect(store.initialize()).resolves.toBeUndefined();
    const db = new DatabaseSync(dbPath);
    expect(userVersion(db)).toBe(9);
    db.close();
  });

  it('v1-v7 still work — upgrades from v1 to v8', async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });

    const v1Dir = await mkdtemp(join(tmpdir(), 'amm-pr5-v1upgrade-'));
    const v1Path = join(v1Dir, 'test.db');

    const v1db = new DatabaseSync(v1Path);
    v1db.exec(`
      CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        tags TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed_at TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1
      );
      CREATE VIRTUAL TABLE memories_fts USING fts5(
        id UNINDEXED, content, tags,
        tokenize='unicode61', content='memories', content_rowid='rowid'
      );
      CREATE TABLE memory_embeddings (
        id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
        embedding BLOB NOT NULL
      );
      CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
      END;
      CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
      END;
      CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
        INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
      END;
      PRAGMA user_version = 1;
    `);
    v1db.close();

    const s = new SqliteMemoryStore(v1Path, new MockEmbedder());
    await s.initialize();
    s.close();

    const check = new DatabaseSync(v1Path);
    expect(userVersion(check)).toBe(9);
    const cols = tableColumns(check, 'memories');
    expect(cols).toContain('is_pinned');
    expect(cols).toContain('status');
    expect(cols).toContain('symbol_key');
    expect(indexExists(check, 'idx_memories_is_pinned')).toBe(true);
    check.close();

    await rm(v1Dir, { recursive: true, force: true });
  });
});

// ── T-002: pin/unpin methods ─────────────────────────────────────────

describe('T-002: pin/unpin on SqliteMemoryStore', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr5-pin-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('pin(id) sets is_pinned=1 and returns updated memory', async () => {
    const mem = await store.store({
      id: 'pin-1',
      content: 'pinnable memory',
      category: 'learning',
      tags: [],
    });

    const pinned = await store.pin('pin-1');
    expect(pinned.id).toBe('pin-1');
    expect((pinned as any).isPinned).toBe(true);

    // Verify in DB
    const db = new DatabaseSync(dbPath);
    const row = db.prepare('SELECT is_pinned FROM memories WHERE id = ?').get('pin-1') as { is_pinned: number };
    expect(row.is_pinned).toBe(1);
    db.close();
  });

  it('pin(id) is idempotent on already-pinned memory', async () => {
    await store.store({
      id: 'pin-2',
      content: 'already pinned',
      category: 'learning',
      tags: [],
    });

    await store.pin('pin-2');
    const pinnedAgain = await store.pin('pin-2');
    expect(pinnedAgain.id).toBe('pin-2');
    expect((pinnedAgain as any).isPinned).toBe(true);
  });

  it('pin(non-existent) throws MemoryNotFoundError', async () => {
    await expect(store.pin('does-not-exist')).rejects.toThrow(MemoryNotFoundError);
  });

  it('unpin(id) sets is_pinned=0 and returns updated memory', async () => {
    await store.store({
      id: 'unpin-1',
      content: 'to be unpinned',
      category: 'learning',
      tags: [],
    });

    await store.pin('unpin-1');
    const unpinned = await store.unpin('unpin-1');
    expect(unpinned.id).toBe('unpin-1');
    expect((unpinned as any).isPinned).toBe(false);

    // Verify in DB
    const db = new DatabaseSync(dbPath);
    const row = db.prepare('SELECT is_pinned FROM memories WHERE id = ?').get('unpin-1') as { is_pinned: number };
    expect(row.is_pinned).toBe(0);
    db.close();
  });

  it('unpin(id) is idempotent on non-pinned memory', async () => {
    await store.store({
      id: 'unpin-2',
      content: 'not pinned',
      category: 'learning',
      tags: [],
    });

    const unpinned = await store.unpin('unpin-2');
    expect(unpinned.id).toBe('unpin-2');
    expect((unpinned as any).isPinned).toBe(false);
  });

  it('unpin(non-existent) throws MemoryNotFoundError', async () => {
    await expect(store.unpin('does-not-exist')).rejects.toThrow(MemoryNotFoundError);
  });
});

describe('T-002: pin/unpin on MockMemoryStore', () => {
  let store: MockMemoryStore;

  beforeEach(() => {
    store = new MockMemoryStore();
  });

  it('pin(id) sets isPinned=true', async () => {
    await store.store({ id: 'mock-pin', content: 'test', category: 'learning', tags: [] });
    const pinned = await store.pin('mock-pin');
    expect((pinned as any).isPinned).toBe(true);
  });

  it('unpin(id) sets isPinned=false', async () => {
    await store.store({ id: 'mock-unpin', content: 'test', category: 'learning', tags: [] });
    await store.pin('mock-unpin');
    const unpinned = await store.unpin('mock-unpin');
    expect((unpinned as any).isPinned).toBe(false);
  });

  it('pin(non-existent) throws', async () => {
    await expect(store.pin('ghost')).rejects.toThrow();
  });
});

describe('T-002: pin/unpin on HardcopyMemoryStore', () => {
  let inner: MockMemoryStore;
  let hardcopy: HardcopyMemoryStore;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr5-hardcopy-'));
    inner = new MockMemoryStore();
    hardcopy = new HardcopyMemoryStore(inner, tmpDir);
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('pin delegates to inner store', async () => {
    await hardcopy.store({ id: 'hc-pin', content: 'test', category: 'learning', tags: [] });
    const pinned = await hardcopy.pin('hc-pin');
    expect((pinned as any).isPinned).toBe(true);
  });

  it('unpin delegates to inner store', async () => {
    await hardcopy.store({ id: 'hc-unpin', content: 'test', category: 'learning', tags: [] });
    await hardcopy.pin('hc-unpin');
    const unpinned = await hardcopy.unpin('hc-unpin');
    expect((unpinned as any).isPinned).toBe(false);
  });
});
