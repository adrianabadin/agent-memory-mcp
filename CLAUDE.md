# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP (Model Context Protocol) server for a memory system backed by SQLite (WAL mode) with hybrid FTS5 BM25 + cosine vector search. Exposes 9 tools (store, store_batch, search, recall, find_related, list_recent, update, stats, get_memory_protocol) over stdio transport. Uses Xenova/bge-m3 embeddings (1024 dimensions) via Hugging Face transformers.

## Commands

- **Build**: `npm run build` (runs `tsc`, outputs to `dist/`)
- **Dev**: `npm run dev` (runs `tsx src/index.ts` directly)
- **Start**: `npm start` (runs compiled `dist/index.js`)
- **Test all**: `npm test` (runs `vitest run`)
- **Test watch**: `npm run test:watch`
- **Run single test**: `npx vitest run tests/tools.test.ts`
- **Run single test by name**: `npx vitest run -t "store tool"`

## Environment Variables

- `MEMORY_DB_PATH` (required) — SQLite database path on disk (e.g., path/to/memory.db)
- `EMBEDDING_MODEL` (optional) — HuggingFace model ID, defaults to `Xenova/bge-m3`
- `EMBEDDING_DIMENSIONS` (optional) — override vector dimensions if needed for a custom model
- `EMBEDDING_POOLING` (optional) — override pooling mode (`cls` for BGE models, `mean` for MiniLM-style models)
- `MEMORY_DECAY_HALF_LIFE` (optional) — temporal decay half-life in days, defaults to `30`. Set to `0` to disable decay
- `ENABLE_HARDCOPY` — set to `'true'` to enable JSON file backup
- `HARDCOPY_PATH` (required if hardcopy enabled) — directory for JSON mirror files
- `EMBEDDING_CACHE_PATH` (optional) — directory for content-addressed binary embedding cache. Speeds up re-runs significantly by skipping ONNX inference for previously-seen text. Cache is invalidated automatically when `EMBEDDING_MODEL` changes.

## Architecture

**Composition chain** (wired in `src/index.ts`):
```
Embedder → SqliteMemoryStore (default) or LanceMemoryStore (migration source) → [HardcopyMemoryStore] → MCP Server → stdio transport
```

**Key interfaces** (`src/types.ts`): `Embedder`, `MemoryStore`, `SearchFilters`, `Memory`, `SearchResult`. All components code against these interfaces, enabling mock-based unit testing.

**SqliteMemoryStore** (`src/sqlite-store.ts`): The main storage layer. Uses SQLite (WAL mode) with a split-table layout (`memories` for metadata/FTS, `memory_embeddings` for vectors). Stores vectors in a BLOB table using IEEE 754 float32 arrays and performs in-process cosine distance calculations. Key features include: FTS5 virtual tables (`memories_fts`) with triggers for keyword indexing; hybrid search combining FTS5 BM25 keyword scoring and cosine vector distance using Reciprocal Rank Fusion (RRF); exponential temporal decay applied to scores (exempting `evergreen` or `never-forget` tagged memories); access tracking (`access_count` and `last_accessed_at`) with spaced-repetition access intervals; and optimistic concurrency control (`version` increment and check).

**LanceMemoryStore** (`src/memory-store.ts`): Read-only migration source for importing legacy LanceDB databases. It is not used for active writes.

**HardcopyMemoryStore** (`src/hardcopy-store.ts`): Decorator pattern — wraps any MemoryStore and mirrors mutations to `{id}.json` files on disk. Write-only (reads delegate to inner store). Errors are logged to stderr but never propagate.

**Tools** (`src/tools.ts`): Registers 8 MCP tools on the server. Each tool handler validates input with Zod schemas, calls the store, and returns JSON-stringified results. The `recall` tool is a composite that runs multiple searches + recent memories.

## Testing

Test files in `tests/`:
- `sqlite-store.test.ts` — unit and integration tests for SqliteMemoryStore
- `migrate.test.ts` — tests migration from LanceDB to SQLite
- `concurrent-update.test.ts` — tests concurrency and version conflicts on SQLite updates
- `embedding-cache.test.ts` — tests the embedding cache mechanism
- `version.test.ts` — checks version constraints and metadata
- `tools.test.ts` — unit tests for all tool handlers using mocks from `tests/mocks.ts`
- `integration.test.ts` — tests LanceMemoryStore with real LanceDB on a temp directory (includes temporal decay integration tests)
- `hardcopy.test.ts` — tests the hardcopy decorator file operations
- `temporal-decay.test.ts` — unit tests for decay math (`computeDecayFactor`, `parseDecayHalfLife`, `EVERGREEN_TAGS`)

`MockEmbedder` produces deterministic hash-based pseudo-vectors. `MockMemoryStore` is an in-memory implementation with substring search matching.

## Conventions

- ES modules (`"type": "module"` in package.json), TypeScript strict mode, target ES2022
- 2-space indentation
- camelCase in TypeScript, snake_case for database column names (e.g., `created_at`)
- 12 memory categories defined as a union type in `types.ts`
