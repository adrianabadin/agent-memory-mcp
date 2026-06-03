import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { LanceMemoryStore } from './memory-store.js';
import { SqliteMemoryStore } from './sqlite-store.js';
import type { Embedder } from './types.js';

/**
 * Migrate all memories from a LanceDB store to a SqliteMemoryStore.
 * Writes a sentinel file at join(lanceDbPath, '.migrated') on success.
 * LanceDB data is left intact as backup.
 *
 * Note: storeBatch re-embeds each memory using the current embedder model.
 * This is intentional — it ensures SQLite has fresh embeddings.
 * LanceDB embeddings are not directly portable.
 */
export async function migrateLanceToSqlite(
  lanceDbPath: string,
  sqliteDbPath: string,
  embedder: Embedder,
): Promise<{ migrated: number }> {
  // 1. Open LanceDB store
  const lanceStore = new LanceMemoryStore(lanceDbPath, embedder);
  await lanceStore.initialize();

  // 2. Fetch all memories
  const memories = await lanceStore.listRecent(999_999);

  // 3. Open SQLite store and write
  const sqliteStore = new SqliteMemoryStore(sqliteDbPath, embedder);
  await sqliteStore.initialize();

  try {
    if (memories.length > 0) {
      await sqliteStore.storeBatch(memories as any[]);
    }
  } finally {
    sqliteStore.close();
  }

  // 4. Write sentinel
  const sentinelPath = join(lanceDbPath, '.migrated');
  await writeFile(sentinelPath, new Date().toISOString(), 'utf8');

  return { migrated: memories.length };
}
