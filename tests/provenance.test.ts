import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder, MockMemoryStore } from './mocks.js';
import { handleStore } from '../src/tools.js';

describe('provenance', () => {
  let tmpDir: string;
  let dbPath: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-provenance-'));
    dbPath = join(tmpDir, 'test.db');
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('initialize upgrades a v0 database in place and preserves old rows', async () => {
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
        version INTEGER NOT NULL DEFAULT 1
      );
      CREATE VIRTUAL TABLE memories_fts USING fts5(
        id UNINDEXED,
        content,
        tags,
        tokenize='unicode61',
        content='memories',
        content_rowid='rowid'
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
      PRAGMA user_version = 0;
    `);
    db.prepare(`
      INSERT INTO memories (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('legacy-1', 'Legacy row', 'learning', '[]', '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 0, '2026-01-01T00:00:00.000Z', 1);
    db.close();

    const store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    const migrated = new DatabaseSync(dbPath);
    const cols = migrated.prepare(`PRAGMA table_info(memories)`).all() as Array<{ name: string }>;
    expect(cols.map((c) => c.name)).toEqual(expect.arrayContaining(['origin', 'source_tool', 'session_id', 'content_hash']));
    const userVersionRow = migrated.prepare(`PRAGMA user_version`).get() as { user_version: number };
    // v0 → v1 (provenance) → v2 (lifecycle/session ledger) → v3 (session decay)
    // → v4 (gradual forgetting); migrations run automatically and bump the
    // target version. Updated for PR-2a.
    expect(userVersionRow.user_version).toBe(4);
    const row = migrated.prepare(`SELECT id, content, origin, source_tool, session_id, content_hash FROM memories WHERE id = ?`).get('legacy-1') as Record<string, unknown>;
    expect(row.content).toBe('Legacy row');
    expect(row.origin).toBe('unknown');
    expect(row.source_tool).toBeNull();
    expect(row.session_id).toBeNull();
    expect(row.content_hash).toBeNull();
    migrated.close();
    store.close();
  });

  it('store with provenance round-trips through search', async () => {
    const store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    await store.store({
      content: 'Traceable memory payload',
      category: 'architecture',
      tags: ['pmc'],
      provenance: {
        origin: 'inferred',
        sourceTool: 'pmc-enrichment/ollama',
        sessionId: 'run-123',
        contentHash: 'hash-abc',
      },
    });

    const results = await store.search('Traceable memory payload', 'keyword', {});
    expect(results).toHaveLength(1);
    expect(results[0].memory.provenance).toEqual({
      origin: 'inferred',
      sourceTool: 'pmc-enrichment/ollama',
      sessionId: 'run-123',
      contentHash: 'hash-abc',
    });
    store.close();
  });

  it('store without provenance reads back with origin unknown', async () => {
    const store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();

    await store.store({
      content: 'No provenance provided',
      category: 'learning',
      tags: [],
    });

    const migrated = new DatabaseSync(dbPath);
    const row = migrated.prepare(`SELECT origin FROM memories WHERE content = ?`).get('No provenance provided') as { origin: string };
    expect(row.origin).toBe('unknown');
    migrated.close();

    const results = await store.search('No provenance provided', 'keyword', {});
    expect(results).toHaveLength(1);
    expect(results[0].memory.provenance).toBeUndefined();
    store.close();
  });

  it('tool handler preserves valid provenance on store', async () => {
    const store = new MockMemoryStore();
    const handler = handleStore(store);
    const result = await handler({
      content: 'Stored through tool',
      category: 'learning',
      tags: ['tool'],
      provenance: {
        origin: 'manual',
        sourceTool: 'user',
        sessionId: 'ses-1',
        contentHash: 'hash-tool',
      },
    });
    expect(result.isError).toBeUndefined();
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.provenance).toEqual({
      origin: 'manual',
      sourceTool: 'user',
      sessionId: 'ses-1',
      contentHash: 'hash-tool',
    });
  });
});
