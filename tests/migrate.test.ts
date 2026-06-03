import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtemp, rm, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { existsSync, readFileSync } from 'node:fs';
import { MockEmbedder } from './mocks.js';

// Mock LanceMemoryStore so we don't need native LanceDB binaries in tests.
// We also preserve all non-class named exports so sqlite-store.ts (which
// imports computeDecayFactor etc. from memory-store.js) continues to work.
vi.mock('../src/memory-store.js', async (importOriginal) => {
  const original = await importOriginal<typeof import('../src/memory-store.js')>();
  return {
    ...original,
    LanceMemoryStore: class MockLanceStore {
      constructor(public path: string, public embedder: any) {}
      async initialize() {}
      async listRecent(_limit: number) {
        return [
          {
            id: 'mem1',
            content: 'Hello world',
            category: 'other' as const,
            tags: ['test'],
            createdAt: '2026-01-01T00:00:00.000Z',
            updatedAt: '2026-01-01T00:00:00.000Z',
            version: 1,
          },
          {
            id: 'mem2',
            content: 'Second memory',
            category: 'learning' as const,
            tags: [],
            createdAt: '2026-01-02T00:00:00.000Z',
            updatedAt: '2026-01-02T00:00:00.000Z',
            version: 2,
          },
        ];
      }
    },
  };
});

// Import after mocking
import { migrateLanceToSqlite } from '../src/migrate.js';
import * as migrateModule from '../src/migrate.js';
import { createStore } from '../src/store-factory.js';
import { SqliteMemoryStore } from '../src/sqlite-store.js';

// ── migrateLanceToSqlite ───────────────────────────────────────────

describe('migrateLanceToSqlite', () => {
  let tmpDir: string;
  let lanceDir: string;
  let sqliteDbPath: string;
  let embedder: MockEmbedder;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'migrate-test-'));
    lanceDir = join(tmpDir, 'lance-db');
    sqliteDbPath = join(tmpDir, 'memory.db');
    embedder = new MockEmbedder();
    // Simulate a LanceDB directory
    await mkdir(lanceDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('migrates 2 memories and they appear in SQLite', async () => {
    const result = await migrateLanceToSqlite(lanceDir, sqliteDbPath, embedder);
    expect(result.migrated).toBe(2);

    // Verify SQLite has the memories; close immediately to release WAL lock
    const store = new SqliteMemoryStore(sqliteDbPath, embedder);
    await store.initialize();
    let memories;
    try {
      memories = await store.listRecent(100);
    } finally {
      store.close();
    }
    expect(memories).toHaveLength(2);
    const ids = memories.map(m => m.id).sort();
    expect(ids).toEqual(['mem1', 'mem2']);
  });

  it('writes .migrated sentinel file in the lanceDB directory', async () => {
    await migrateLanceToSqlite(lanceDir, sqliteDbPath, embedder);

    const sentinelPath = join(lanceDir, '.migrated');
    expect(existsSync(sentinelPath)).toBe(true);

    // Sentinel should contain a valid ISO date string
    const content = readFileSync(sentinelPath, 'utf8');
    expect(new Date(content).getTime()).not.toBeNaN();
  });

  it('returns migrated count of 2', async () => {
    const { migrated } = await migrateLanceToSqlite(lanceDir, sqliteDbPath, embedder);
    expect(migrated).toBe(2);
  });
});

// ── createStore ────────────────────────────────────────────────────

describe('createStore', () => {
  let tmpDir: string;
  let dbBase: string;
  let embedder: MockEmbedder;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'store-factory-test-'));
    dbBase = join(tmpDir, 'agent-memory');
    embedder = new MockEmbedder();
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('creates a fresh SQLite store when no LanceDB directory exists', async () => {
    const store = await createStore(dbBase, embedder);
    let memories;
    try {
      expect(existsSync(`${dbBase}.db`)).toBe(true);
      memories = await store.listRecent(100);
    } finally {
      store.close();
    }
    expect(memories).toHaveLength(0);
  });

  it('auto-migrates when LanceDB dir exists, no .db, no sentinel', async () => {
    // Simulate LanceDB directory
    await mkdir(dbBase, { recursive: true });

    const store = await createStore(dbBase, embedder);
    let memories;
    try {
      memories = await store.listRecent(100);
    } finally {
      store.close();
    }

    // Should have migrated 2 memories from mock LanceStore
    expect(memories).toHaveLength(2);
    // Sentinel should be written
    expect(existsSync(join(dbBase, '.migrated'))).toBe(true);
  });

  it('skips migration when sentinel already exists', async () => {
    // Simulate LanceDB dir + do first migration to create .db and sentinel
    await mkdir(dbBase, { recursive: true });
    const firstStore = await createStore(dbBase, embedder);
    firstStore.close();

    // Reset embedder call log to count only second pass calls
    const callsBefore = embedder.callLog.length;

    // Second call — sentinel exists, should NOT re-migrate
    const store = await createStore(dbBase, embedder);
    let callsAfter: number;
    try {
      callsAfter = embedder.callLog.length;
    } finally {
      store.close();
    }

    // On second pass, no new embed calls from migration
    // (store.initialize() alone makes no embed calls)
    expect(callsAfter).toBe(callsBefore);
  });

  it('catches migration errors and returns a fresh store instead of throwing', async () => {
    // Simulate LanceDB dir so createStore attempts migration
    const lanceDir = join(tmpDir, 'fail-lance');
    await mkdir(lanceDir, { recursive: true });
    const dbPath = join(tmpDir, 'fail-lance');

    // Force migrateLanceToSqlite to throw, exercising the catch branch in createStore
    vi.spyOn(migrateModule, 'migrateLanceToSqlite').mockRejectedValueOnce(new Error('simulated failure'));

    const store = await createStore(dbPath, embedder);
    try {
      const stats = await store.stats();
      expect(stats.totalMemories).toBe(0); // fresh store, no data
    } finally {
      store.close();
    }
  });
});
