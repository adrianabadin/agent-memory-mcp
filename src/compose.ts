import { TransformersEmbedder } from './embedder.js';
import { EmbeddingCache } from './embedding-cache.js';
import { createStore } from './store-factory.js';
import { createGlobalStore } from './global/global-factory.js';
import type {
  Embedder,
  MemoryStore,
  SessionPrompt,
  SessionResponse,
  SessionToolCall,
} from './types.js';
import type { GlobalStore } from './global/global-store.js';
import { SqliteMemoryStore } from './sqlite-store.js';

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

// ── Ledger-only store (session-capture drainer) ──────────────────────

/**
 * The session-ledger surface exposed to the capture-pipeline drainer.
 *
 * The drainer only needs to append session prompts / responses / tool calls
 * and manage the in-memory session context. It must NOT touch the memory
 * store/search tables (store, storeBatch, search, findRelated, listRecent,
 * update, delete, stats, prune) so a runaway drainer cannot fragment the
 * memory data. This interface statically enforces that boundary: callers get
 * exactly the session-ledger methods plus lifecycle (`initialize` / `close`)
 * and nothing else.
 */
export interface LedgerStore {
  /** Open the database and run idempotent schema migrations. */
  initialize(): Promise<void>;
  /** Close the underlying SQLite handle (idempotent; safe to call repeatedly). */
  close(): void;

  /** Append a raw prompt to the session ledger. */
  storeSessionPrompt(sessionId: string, rawPrompt: string): Promise<SessionPrompt>;
  /** Append an assistant response linked to a prompt. */
  storeSessionResponse(sessionId: string, promptId: string, fullResponse: string): Promise<SessionResponse>;
  /** Append a tool-call record to the session ledger. */
  storeSessionToolCall(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    toolName: string;
    argsSafe?: string;
    resultSummary?: string;
    importance?: string;
  }): Promise<SessionToolCall>;
  /** Set the active session context (in-memory; not persisted). */
  setSessionContext(sessionId: string, projectId: string): void;
  /** Read the active session context, or null when none is set. */
  getSessionContext(): { sessionId: string; projectId: string } | null;
}

/**
 * No-op embedder used by {@link createLedgerOnlyStore}. The session-ledger
 * write path calls `embedder.embed(...)` for every row, but the drainer does
 * not need real vectors (the capture pipeline never runs semantic search on
 * raw session items directly). Returning an empty vector avoids the embedding
 * model download entirely, so the drainer boots instantly.
 */
const dummyEmbedder: Embedder = {
  initialize: async () => {},
  embed: async () => [],
  embedBatch: async () => [],
  dimensions: () => 1024,
};

/**
 * Build a ledger-only store over `SqliteMemoryStore` for the session-capture
 * drainer.
 *
 * The drainer writes session-ledger rows (prompts / responses / tool calls)
 * directly to the project memory database without going through the MCP server
 * and without loading the embedding model. It reuses the canonical
 * `SqliteMemoryStore` ledger implementation (no schema drift) but is typed as
 * {@link LedgerStore} so only the session surface is reachable.
 *
 * The returned handle:
 *   - is already initialized (schema + migrations applied);
 *   - has `PRAGMA busy_timeout = 5000` set, so transient WAL lock contention
 *     from the main agent process retries for up to 5s instead of failing
 *     immediately (capture-pipeline R2);
 *   - uses the path convention `${dbPath}.db` to match the main store.
 *
 * **W2 — session-ledger-only surface.** The handle is typed as
 * {@link LedgerStore}, which exposes ONLY `initialize` / `close` plus the
 * session-ledger methods (`storeSessionPrompt`, `storeSessionResponse`,
 * `storeSessionToolCall`, `setSessionContext`, `getSessionContext`). The
 * memory store/search methods (`store`, `storeBatch`, `search`, `findRelated`,
 * `listRecent`, `update`, `delete`, `stats`, `prune`) are NOT exposed — they
 * remain on the underlying object but are invisible through this type, so a
 * runaway drainer cannot fragment the memory data.
 *
 * **Windows EBUSY mitigation.** SQLite runs in WAL mode; combined with the
 * 5s `busy_timeout` above, the drainer retries transient `SQLITE_BUSY`/EBUSY
 * errors for up to 5s at the driver level. The drainer CLI
 * (`cli/capture-drain.mjs`) additionally wraps each entry write in a 3x retry
 * with exponential backoff (100ms, 200ms, 400ms) so a persistent lock is
 * retried at the application layer as well. The two layers compose: driver
 * retry first, then application retry, then the failed entry stays in the
 * queue for the next drain run (no data loss).
 *
 * Callers MUST `close()` the store when the drainer exits to release the file
 * lock.
 *
 * @param dbPath Base path for the memory SQLite DB (the `.db` suffix is appended).
 * @returns An initialized, ledger-only handle over the shared memory database.
 */
export async function createLedgerOnlyStore(dbPath: string): Promise<LedgerStore> {
  // Match the path convention used by the main store: append `.db`.
  const sqlitePath = `${dbPath}.db`;
  const store = new SqliteMemoryStore(sqlitePath, dummyEmbedder);
  await store.initialize();

  // Enforce a 5s busy_timeout so the drainer survives transient lock
  // contention with the main agent process (capture-pipeline R2). The PRAGMA
  // is set after initialize() so it applies to the already-opened handle.
  store.getDb().exec('PRAGMA busy_timeout = 5000');

  // Expose only the session-ledger surface; the memory store/search methods
  // remain on the underlying object but are invisible through this type.
  return store as unknown as LedgerStore;
}
