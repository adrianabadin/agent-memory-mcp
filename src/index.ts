#!/usr/bin/env node
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createProjectEmbedder, createGlobalEmbedder } from './embedder.js';
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

  // ── Compose dependencies ──
  // Project embedder follows the standard env vars (may be a 384-dim E5).
  const embedder = createProjectEmbedder();
  let store: MemoryStore = await createStore(dbPath, embedder);

  if (process.env.ENABLE_HARDCOPY === 'true' && process.env.HARDCOPY_PATH) {
    store = new HardcopyMemoryStore(store, process.env.HARDCOPY_PATH);
    console.error(`[hardcopy] Mirroring mutations to ${process.env.HARDCOPY_PATH}`);
  }

  // ── Global context store (optional — failure does not block startup) ──
  // ALWAYS an independent BGE-M3 1024 embedder with its own cache: the global
  // DB stores 1024-dim BGE vectors and must never inherit the project model.
  const globalEmbedder = createGlobalEmbedder();
  const globalStore = await createGlobalStore(globalEmbedder);
  if (globalStore) {
    console.error(`[global] Global context store ready`);
  }

  const server = createServer(store, globalStore ?? undefined);

  // ── Initialise (download model on first run, connect to DB) ──
  await embedder.initialize();
  if (globalStore) {
    try {
      await globalEmbedder.initialize();
    } catch (err) {
      // Preserve the "global failure does not block startup" contract.
      console.error('[global] Embedder initialisation failed; global semantic ops degraded:', err);
    }
  }

  // ── Start MCP transport ──
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
