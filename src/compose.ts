import { TransformersEmbedder } from './embedder.js';
import { EmbeddingCache } from './embedding-cache.js';
import { createStore } from './store-factory.js';
import { createGlobalStore } from './global/global-factory.js';
import type { Embedder, MemoryStore } from './types.js';
import type { GlobalStore } from './global/global-store.js';

// Re-export the composition primitives so consumers (the unified `pmc` server)
// can import everything they need from a single side-effect-free entry point,
// without triggering the standalone `index.ts` main() that runs when this
// package is invoked as a bin.
export { createServer } from './server.js';
export { registerTools } from './tools.js';
export { TransformersEmbedder } from './embedder.js';
export type { Embedder, MemoryStore } from './types.js';
export type { GlobalStore } from './global/global-store.js';

/**
 * Wraps an Embedder and defers its (potentially heavy) `initialize()` — which
 * downloads the embedding model — until the first embed call.
 *
 * This lets a unified server boot, register tools, and answer `tools/list`
 * before any model download, while keeping embed calls working on first real
 * use. The standalone agent-memory bin still eagerly initializes (its own
 * `index.ts`); this wrapper only affects composed consumers.
 */
export class LazyEmbedder implements Embedder {
  private initPromise: Promise<void> | null = null;
  private ready = false;

  constructor(private readonly inner: Embedder) {}

  async initialize(): Promise<void> {
    if (!this.initPromise) {
      this.initPromise = this.inner
        .initialize()
        .then(() => {
          this.ready = true;
        })
        .catch((err) => {
          // Allow a retry on the next embed if initialization failed.
          this.initPromise = null;
          throw err;
        });
    }
    return this.initPromise;
  }

  /** Whether the underlying model has finished initializing. */
  isInitialized(): boolean {
    return this.ready;
  }

  async embed(text: string): Promise<number[]> {
    await this.ensureReady();
    return this.inner.embed(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    await this.ensureReady();
    return this.inner.embedBatch(texts);
  }

  dimensions(): number {
    // Synchronous metadata; must not trigger model init.
    return this.inner.dimensions();
  }

  private async ensureReady(): Promise<void> {
    if (!this.ready) {
      await this.initialize();
    }
  }
}

export interface MemoryComponents {
  store: MemoryStore;
  globalStore: GlobalStore | null;
  embedder: Embedder;
}

/**
 * Build the memory-side components (store, global store, embedder) needed to
 * register memory tools on an MCP server.
 *
 * Side-effect-free apart from opening the SQLite databases (schema creation) —
 * does NOT download the embedding model. The embedder is wrapped in
 * {@link LazyEmbedder} so the model is only fetched on the first `store`/`search`.
 *
 * Intended for composition: the unified `pmc` server calls this, then
 * `registerTools(server, components.store, components.globalStore ?? undefined)`
 * on a shared McpServer that also hosts query tools.
 *
 * @param options.dbPath     Base path for the memory SQLite DB (REQUIRED).
 * @param options.embedder   Optional injected embedder (used by tests). When
 *                           omitted, a real TransformersEmbedder is constructed
 *                           from the standard embedding env vars.
 */
export async function buildMemoryComponents(options: {
  dbPath: string;
  embedder?: Embedder;
}): Promise<MemoryComponents> {
  const dbPath = options?.dbPath;
  if (!dbPath) {
    throw new Error('buildMemoryComponents requires a dbPath');
  }

  const modelName = process.env.EMBEDDING_MODEL ?? 'Xenova/bge-m3';
  const dimensions = process.env.EMBEDDING_DIMENSIONS
    ? parseInt(process.env.EMBEDDING_DIMENSIONS, 10)
    : undefined;
  const pooling = process.env.EMBEDDING_POOLING ?? undefined;
  const cachePath = process.env.EMBEDDING_CACHE_PATH;
  const cache = cachePath ? new EmbeddingCache(cachePath, modelName) : undefined;

  const baseEmbedder =
    options.embedder ?? new TransformersEmbedder(modelName, dimensions, pooling, cache);
  const embedder = new LazyEmbedder(baseEmbedder);

  const store = await createStore(dbPath, embedder);
  const globalStore = await createGlobalStore(embedder);

  return { store, globalStore, embedder };
}
