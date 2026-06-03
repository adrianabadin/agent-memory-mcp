# @aabadin/agent-memory-mcp

MCP server for persistent agent memory, backed by **SQLite** (`node:sqlite` built-in, Node 22.5+) with hybrid BM25 + vector search. Gives AI agents the ability to store, search, and manage memories across sessions using the [Model Context Protocol](https://modelcontextprotocol.io/).

All data stays on your machine. Embeddings are generated locally using [bge-m3](https://huggingface.co/Xenova/bge-m3) via ONNX — no API keys, no network dependencies after initial setup.

## Features

- **Hybrid search** — combines BM25 full-text search with cosine vector similarity via Reciprocal Rank Fusion (RRF)
- **Local embeddings** — runs Xenova/bge-m3 locally via ONNX, no external API calls
- **12 memory categories** — structured taxonomy for organising memories
- **Batch operations** — store multiple memories in a single call
- **Hardcopy backup** — optional JSON file mirror of all mutations for human-readable backup
- **Temporal decay** — exponential time-based decay favors recent memories when relevance is similar. Configurable half-life, with `evergreen` and `never-forget` tag exemptions
- **Fully local** — all data stays on disk, no network dependencies after first model download

## Install

```bash
npm install -g @aabadin/agent-memory-mcp
```

Warning: the first run downloads `Xenova/bge-m3`. Keep at least 1 GB free for the local model cache and SQLite store.

Supported platforms: Linux x64/arm64, macOS x64/arm64, Windows x64.

Then add the server to your MCP client configuration.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "env": {
        "MEMORY_DB_PATH": "/path/to/your/memory-db"
      }
    }
  }
}
```

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "agent-memory": {
    "command": "agent-memory-mcp",
    "env": {
      "MEMORY_DB_PATH": "/path/to/your/memory-db"
    }
  }
}
```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `MEMORY_DB_PATH` | Yes | Base path for the SQLite database. The actual file is created at `${MEMORY_DB_PATH}.db`. If a LanceDB directory exists at this path, the store migrates it automatically on first start. |
| `EMBEDDING_MODEL` | No | HuggingFace model ID (default: `Xenova/bge-m3`) |
| `EMBEDDING_DIMENSIONS` | No | Expected vector size. Default: inferred from model (`1024` for `Xenova/bge-m3`) |
| `EMBEDDING_POOLING` | No | Pooling strategy. Default: inferred from model (`cls` for BGE models, `mean` otherwise) |
| `EMBEDDING_CACHE_PATH` | No | Directory for content-addressed binary embedding cache. Speeds up repeated runs by skipping ONNX inference for previously-seen text |
| `MEMORY_DECAY_HALF_LIFE` | No | Decay half-life in days (default: `30`). Set to `0` to disable temporal decay |
| `ENABLE_HARDCOPY` | No | Set to `true` to enable JSON file backup |
| `HARDCOPY_PATH` | If hardcopy enabled | Directory for JSON mirror files |

## Storage Backend

As of v2.0, the storage backend is **SQLite** via Node's built-in `node:sqlite` module (requires Node ≥ 22.5.0). No external database dependency is needed.

**Automatic LanceDB migration:** If a LanceDB directory exists at `MEMORY_DB_PATH` when the server starts, it is automatically migrated to the SQLite file at `${MEMORY_DB_PATH}.db`. The original LanceDB directory is left intact after migration.

## Upgrading From MiniLM

If you already have a database created with a smaller embedding model such as `Xenova/all-MiniLM-L6-v2` (384 dimensions), you must rebuild that database before using `bge-m3` (1024 dimensions). Mixing vector sizes in the same store is not supported.

Typical upgrade flow:

1. Stop the MCP server.
2. Backup or delete the old `MEMORY_DB_PATH` files (the `.db` file and any LanceDB directory).
3. Start the server again so all memories are re-embedded with the new model.

## Tools

| Tool | Description |
|---|---|
| `store` | Store a single memory with content, category, and tags |
| `store_batch` | Store multiple memories in one call |
| `search` | Search memories by meaning and/or keywords. Supports hybrid, keyword, and semantic modes |
| `recall` | Multi-topic contextual recall — searches multiple topics in parallel and includes recent memories |
| `find_related` | Find memories similar to a specific memory |
| `list_recent` | List most recent memories, optionally filtered by category |
| `update` | Update an existing memory — re-embeds automatically if content changes |
| `delete` | Permanently remove a memory by ID |
| `stats` | Get database statistics: total count, breakdown by category, timestamps |
| `prune` | Preview or prune low-strength and dormant memories. Dry-run by default |

## Search Modes

The `search` tool supports three modes:

- **`hybrid`** (default) — combines BM25 keyword scoring with vector similarity using RRF reranking. Falls back to semantic-only if the full-text index is unavailable.
- **`keyword`** — BM25 full-text search only.
- **`semantic`** — cosine vector similarity only.

All modes support filtering by category, tags, and date range.

## Temporal Decay

Search results are scored with exponential time-based decay so that recent memories surface above older ones when semantic relevance is similar. The decay follows a half-life model: a memory one half-life old has its score halved, two half-lives old gets quartered, and so on.

- **Default half-life**: 30 days (configurable via `MEMORY_DECAY_HALF_LIFE`)
- **Disable**: set `MEMORY_DECAY_HALF_LIFE=0`
- **Exempt tags**: memories tagged `evergreen` or `never-forget` are never decayed

## Memory Categories

`code-solution` · `bug-fix` · `architecture` · `learning` · `tool-usage` · `debugging` · `performance` · `security` · `observation` · `personal` · `relationship` · `other`

## Development

```bash
git clone https://github.com/adamrdrew/agent-memory-mcp.git
cd agent-memory-mcp
npm install
npm run dev          # Run with tsx (no build step)
npm run build        # Compile TypeScript to dist/
npm test             # Run all tests
npm run test:watch   # Run tests in watch mode
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
