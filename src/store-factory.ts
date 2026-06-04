import { SqliteMemoryStore } from './sqlite-store.js';
import type { Embedder } from './types.js';

/**
 * Create a SqliteMemoryStore for the given base path.
 *
 * Path convention:
 *   MEMORY_DB_PATH = /some/path/agent-memory
 *   SQLite file    = /some/path/agent-memory.db
 *
 * For one-time migration from LanceDB, run:
 *   npx tsx scripts/migrate-lance.ts
 */
export async function createStore(
  dbPath: string,
  embedder: Embedder,
): Promise<SqliteMemoryStore> {
  const sqlitePath = `${dbPath}.db`;
  const store = new SqliteMemoryStore(sqlitePath, embedder);
  await store.initialize();
  return store;
}
