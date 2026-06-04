import { homedir } from 'node:os';
import { join } from 'node:path';
import type { Embedder } from '../types.js';
import { SqliteGlobalStore, type GlobalStore } from './global-store.js';

/**
 * Create and initialize the global context store.
 *
 * Default path: ~/.pmc/global.db
 * Override with: PMC_GLOBAL_DB_PATH env var
 *
 * Returns null if the global store cannot be opened (e.g. permissions, disk full).
 * This ensures the main agent-memory server starts even if the global store fails.
 */
export async function createGlobalStore(embedder: Embedder): Promise<GlobalStore | null> {
  const dbPath = process.env.PMC_GLOBAL_DB_PATH ?? join(homedir(), '.pmc', 'global.db');
  try {
    const store = new SqliteGlobalStore(dbPath, embedder);
    await store.initialize();
    return store;
  } catch (err) {
    console.error('[global-factory] Failed to open global store, global context disabled:', err);
    return null;
  }
}
