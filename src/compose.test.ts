import { describe, it, expect, expectTypeOf } from 'vitest';
import { mkdtemp } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import { createLedgerOnlyStore } from './compose.js';
import type { LedgerStore } from './compose.js';

/**
 * T-001 (PR-1): createLedgerOnlyStore factory + LedgerStore interface.
 *
 * The factory instantiates SqliteMemoryStore with a dummy (no-op) embedder so
 * the session-capture drainer can write session-ledger rows without pulling in
 * the embedding model. Only the session-ledger surface is exposed via the
 * LedgerStore interface; the memory store/search methods are intentionally
 * hidden so the drainer cannot accidentally fragment the memory tables.
 */
describe('createLedgerOnlyStore (LedgerOnlyStore factory)', () => {
  // Test 1 — Factory creates the SQLite DB at `dbPath + '.db'`.
  it('creates the database file at `${dbPath}.db`', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'pmc-ledger-'));
    const dbPath = join(dir, 'memory'); // expect file at memory.db
    const expectedFile = `${dbPath}.db`;

    const store = await createLedgerOnlyStore(dbPath);
    try {
      expect(existsSync(expectedFile)).toBe(true);
    } finally {
      store.close();
    }
  });

  // Test 2 — LedgerStore exposes the 5 session methods (names from sqlite-store.ts).
  it('exposes the 5 session methods: storeSessionPrompt, storeSessionResponse, storeSessionToolCall, setSessionContext, getSessionContext', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'pmc-ledger-'));
    const store = await createLedgerOnlyStore(join(dir, 'memory'));
    try {
      expect(typeof store.storeSessionPrompt).toBe('function');
      expect(typeof store.storeSessionResponse).toBe('function');
      expect(typeof store.storeSessionToolCall).toBe('function');
      expect(typeof store.setSessionContext).toBe('function');
      expect(typeof store.getSessionContext).toBe('function');
    } finally {
      store.close();
    }
  });

  // Test 3 — Non-session methods are NOT on the LedgerStore interface.
  // This is a type-level check: the LedgerStore interface must not surface the
  // memory store/search methods, so the drainer is statically prevented from
  // fragmenting the memory tables.
  it('does not expose non-session (memory store/search) methods on the LedgerStore interface', () => {
    const nonSessionMethods = [
      'store',
      'storeBatch',
      'search',
      'findRelated',
      'listRecent',
      'update',
      'delete',
      'stats',
      'prune',
    ] as const;

    for (const method of nonSessionMethods) {
      expectTypeOf<LedgerStore>().not.toHaveProperty(method);
    }
  });

  // Test 4 — PRAGMA busy_timeout = 5000 (capture-pipeline R2: WAL contention).
  it('sets PRAGMA busy_timeout to 5000ms on the underlying database', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'pmc-ledger-'));
    const store = await createLedgerOnlyStore(join(dir, 'memory'));
    try {
      // The LedgerStore interface deliberately hides the raw DB handle; reach
      // through to the SqliteMemoryStore implementation to read the PRAGMA.
      // node:sqlite exposes `PRAGMA busy_timeout` results under the column
      // name `timeout`.
      const db = (store as unknown as { getDb(): { prepare(sql: string): { get(): { timeout: number } } } }).getDb();
      const row = db.prepare('PRAGMA busy_timeout').get() as { timeout: number };
      expect(row.timeout).toBe(5000);
    } finally {
      store.close();
    }
  });
});
