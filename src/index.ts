#!/usr/bin/env node
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { EmbeddingCache } from './embedding-cache.js';
import { TransformersEmbedder } from './embedder.js';
import { HardcopyMemoryStore } from './hardcopy-store.js';
import { createStore } from './store-factory.js';
import { createServer } from './server.js';
import { createGlobalStore } from './global/global-factory.js';
import type { MemoryStore } from './types.js';

async function main(): Promise<void> {
  const dbPath = process.env.MEMORY_DB_PATH;
  if (!dbPath) {
    console.error('MEMORY_DB_PATH environment variable is required');
    process.exit(1);
  }

  const modelName = process.env.EMBEDDING_MODEL ?? 'Xenova/bge-m3';
  const dimensions = process.env.EMBEDDING_DIMENSIONS ? parseInt(process.env.EMBEDDING_DIMENSIONS, 10) : undefined;
  const pooling = process.env.EMBEDDING_POOLING ?? undefined;
  const cachePath = process.env.EMBEDDING_CACHE_PATH;

  // ── Compose dependencies ──
  const cache = cachePath ? new EmbeddingCache(cachePath, modelName) : undefined;
  const embedder = new TransformersEmbedder(modelName, dimensions, pooling, cache);
  let store: MemoryStore = await createStore(dbPath, embedder);

  if (process.env.ENABLE_HARDCOPY === 'true' && process.env.HARDCOPY_PATH) {
    store = new HardcopyMemoryStore(store, process.env.HARDCOPY_PATH);
    console.error(`[hardcopy] Mirroring mutations to ${process.env.HARDCOPY_PATH}`);
  }

  // ── Global context store (optional — failure does not block startup) ──
  const globalStore = await createGlobalStore(embedder);
  if (globalStore) {
    console.error(`[global] Global context store ready`);
  }

  const server = createServer(store, globalStore ?? undefined);

  // ── Initialise (download model on first run, connect to DB) ──
  await embedder.initialize();

  // ── Start MCP transport ──
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
