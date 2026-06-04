#!/usr/bin/env npx tsx
/**
 * One-time offline migration: LanceDB → SQLite.
 *
 * Usage:
 *   npx tsx scripts/migrate-lance.ts
 *
 * Reads MEMORY_DB_PATH from env. If a LanceDB directory exists at that
 * path and SQLite does NOT exist at `${path}.db`, migrates all memories
 * and writes a .migrated sentinel. Safe to run multiple times.
 */
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import * as lancedb from '@lancedb/lancedb';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { TransformersEmbedder } from '../src/embedder.js';

async function main() {
  const dbPath = process.env.MEMORY_DB_PATH;
  if (!dbPath) {
    console.error('MEMORY_DB_PATH is required');
    process.exit(1);
  }

  const sqlitePath = `${dbPath}.db`;
  const sentinelPath = join(dbPath, '.migrated');

  if (!existsSync(dbPath)) {
    console.log('No LanceDB directory found — nothing to migrate.');
    return;
  }
  if (existsSync(sentinelPath)) {
    console.log('Already migrated (.migrated sentinel found) — skipping.');
    return;
  }
  if (existsSync(sqlitePath)) {
    console.log('SQLite file already exists — skipping migration.');
    return;
  }

  console.log(`Migrating ${dbPath} → ${sqlitePath} ...`);

  const modelName = process.env.EMBEDDING_MODEL ?? 'Xenova/bge-m3';
  const embedder = new TransformersEmbedder(modelName);
  await embedder.initialize();

  // ── Read from LanceDB ──
  const lanceConn = await lancedb.connect(dbPath);
  let lanceTable: lancedb.Table | null = null;
  let allRows: unknown[] = [];

  try {
    lanceTable = await lanceConn.openTable('memories');
    allRows = await lanceTable.query().execute();
  } catch (err) {
    console.error('Could not open LanceDB table "memories":', err);
    process.exit(1);
  }

  console.log(`Found ${allRows.length} memories in LanceDB.`);

  // ── Write to SQLite ──
  const sqlite = new SqliteMemoryStore(sqlitePath, embedder);
  await sqlite.initialize();

  if (allRows.length > 0) {
    const requests = allRows.map((row: any) => ({
      id: String(row.id),
      content: String(row.content),
      category: String(row.category),
      tags: (() => { try { return JSON.parse(String(row.tags ?? '[]')); } catch { return []; } })(),
      created_at: String(row.created_at ?? new Date().toISOString()),
      updated_at: String(row.updated_at ?? new Date().toISOString()),
      access_count: Number(row.access_count ?? 0),
      last_accessed_at: String(row.last_accessed_at ?? new Date().toISOString()),
      version: Number(row.version ?? 1),
    }));

    await sqlite.storeBatch(requests);
    console.log(`Migrated ${requests.length} memories to SQLite.`);
  }

  sqlite.close();

  // ── Write sentinel ──
  await writeFile(sentinelPath, new Date().toISOString(), 'utf8');
  console.log(`Migration complete. Sentinel written to ${sentinelPath}`);
  console.log(`LanceDB data preserved at ${dbPath} (safe to delete manually after verification).`);
}

main().catch(err => {
  console.error('Migration failed:', err);
  process.exit(1);
});
