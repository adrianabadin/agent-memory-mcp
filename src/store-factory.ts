import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { SqliteMemoryStore } from './sqlite-store.js';
import { migrateLanceToSqlite } from './migrate.js';
import type { Embedder } from './types.js';

/**
 * Create a SqliteMemoryStore for the given base path.
 *
 * Path convention:
 *   MEMORY_DB_PATH = /some/path/agent-memory
 *   LanceDB dir   = /some/path/agent-memory        (directory, no extension)
 *   SQLite file   = /some/path/agent-memory.db
 *   Sentinel      = /some/path/agent-memory/.migrated
 *
 * Auto-migration: if a LanceDB directory exists at `dbPath` (no .db extension)
 * and `dbPath + '.db'` does NOT exist and the sentinel is NOT present,
 * migrate automatically before returning the store.
 */
export async function createStore(
  dbPath: string,
  embedder: Embedder,
): Promise<SqliteMemoryStore> {
  const sqlitePath = `${dbPath}.db`;
  const sentinelPath = join(dbPath, '.migrated');
  const lanceExists =
    existsSync(dbPath) && !existsSync(sqlitePath) && !existsSync(sentinelPath);

  if (lanceExists) {
    try {
      await migrateLanceToSqlite(dbPath, sqlitePath, embedder);
    } catch (err) {
      // Migration failed — log and continue with fresh SQLite store
      console.error('[store-factory] LanceDB migration failed, starting fresh:', err);
    }
  }

  const store = new SqliteMemoryStore(sqlitePath, embedder);
  await store.initialize();
  return store;
}
