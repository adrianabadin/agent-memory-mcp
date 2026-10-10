# @aabadin/agent-memory-mcp

MCP server for persistent agent memory, backed by **SQLite** (`node:sqlite` built-in, Node 22.5+) with hybrid BM25 + vector search. Gives AI agents the ability to store, search, and manage memories across sessions using the [Model Context Protocol](https://modelcontextprotocol.io/).

All data stays on your machine. Embeddings are generated locally using [bge-m3](https://huggingface.co/Xenova/bge-m3) via ONNX — no API keys, no network dependencies after initial setup.

## Features

- **Hybrid search** — combines BM25 full-text search with cosine vector similarity via Reciprocal Rank Fusion (RRF)
- **Cross-project search** — opt-in `all_projects` flag queries across multiple project memory DBs and merges results by score
- **Conflict resolution** — semantic collision detection during `store()` (configurable threshold), `compare` and `judge` MCP tools, and 6-relation `memory_relations` table for LLM-driven verdicts
- **Pin / unpin memories** — explicitly mark memories as important; pinned memories sort first and are exempt from temporal decay
- **Review queue** — `review_queue` tool surfaces memories flagged by staged decay (dimmed / shadow / gist_only); pinned and `non_forgettable` memories are skipped
- **Local embeddings** — runs Xenova/bge-m3 locally via ONNX, no external API calls
- **12 memory categories** — structured taxonomy for organising memories
- **Batch operations** — store multiple memories in a single call
- **Session ledger** — captures prompts, responses, tool calls, and summaries; auto-links stores to active session
- **Hardcopy backup** — optional JSON file mirror of all mutations for human-readable backup
- **Temporal decay** — exponential time-based decay favors recent memories when relevance is similar. Configurable half-life, with `evergreen` and `never-forget` tag exemptions
- **Runtime protocol flag** — `get_memory_protocol` exposes dual/persistent memory backend state to MCP clients
- **Fully local** — all data stays on disk, no network dependencies after first model download

## Install

```bash
npm install -g @aabadin/agent-memory-mcp
```

Or install from this repo's GitHub fork at a branch or commit SHA (the `prepare` script runs `npm run build`, so `dist/` is compiled on install):

```bash
npm install -g github:adrianabadin/agent-memory-mcp#feat/symbol-linked-memory-context
# or pin a SHA: npm install -g github:adrianabadin/agent-memory-mcp#<commit-sha>
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

### Core CRUD
| Tool | Description |
|---|---|
| `store` | Store a single memory with content, category, and tags. Opt-in `detect_conflicts` flag triggers collision detection (returns `{ judgment_required, candidates }` without persisting when matches found) |
| `store_batch` | Store multiple memories in one call |
| `search` | Search memories by meaning and/or keywords. Supports `hybrid` (default), `keyword`, `semantic`. Opt-in `all_projects` flag for cross-project search. Pinned memories sort first |
| `recall` | Multi-topic contextual recall — searches multiple topics in parallel and includes recent memories |
| `find_related` | Find memories similar to a specific memory |
| `list_recent` | List most recent memories, optionally filtered by category. Excludes archived. Pinned sort first |
| `update` | Update an existing memory — re-embeds automatically if content changes. Preserves lifecycle status |
| `delete` | Permanently remove a memory by ID |
| `stats` | Get database statistics: total count, breakdown by category, timestamps, prune candidates |
| `prune` | Preview or prune low-strength and dormant memories. Dry-run by default |

### Pin & Lifecycle
| Tool | Description |
|---|---|
| `pin` | Mark a memory as explicitly important. Pinned memories sort first and are exempt from temporal decay |
| `unpin` | Remove pin marker (memory re-enters normal decay schedule) |
| `update_memory_status` | Transition a memory through lifecycle states: `active` → `needs_review` → `archived`. Pinned memories and `archived` (terminal) are rejected for further transitions |

### Conflict Resolution
| Tool | Description |
|---|---|
| `compare` | Compare two memories: cosine similarity, shared entities, content diff, and a relation recommendation |
| `judge` | Persist a verdict to the `memory_relations` table. Relations: `related`, `compatible`, `scoped`, `conflicts_with`, `supersedes`, `not_conflict`. The `supersedes` relation cascades to archive the older memory |

### Review Queue
| Tool | Description |
|---|---|
| `review_queue` | List memories with `status='needs_review'`, ordered by oldest first. Filterable by `project_id`. Default limit 50 |

### Topic Aliases
| Tool | Description |
|---|---|
| `upsert_topic_alias` | Bind a `topic_key` to a memory id. Topic keys enable upsert-by-topic semantics: re-storing with the same topic updates the existing memory |
| `resolve_topic` | Resolve a `topic_key` to its memory |
| `suggest_topic_key` | Slugify content into a deterministic `topic_key`, deduplicating against existing keys |

### Session Ledger
| Tool | Description |
|---|---|
| `store_session_prompt` | Record a user prompt in the session ledger |
| `store_session_response` | Record an assistant response linked to its prompt |
| `store_session_tool_call` | Record an important tool call during the session |
| `store_session_summary` | Generate and store a structured session summary (decisions, key titles, memory ids, artifact refs) |
| `get_session` | Retrieve all ledger entries for a session, ordered by `created_at ASC` |
| `list_sessions` | List distinct sessions, ordered by last activity desc, with prompt counts |
| `link_prompt_to_memory` | Create a many-to-many link between a prompt and a memory (with optional response id) |
| `set_session_context` | Set the active session context. Subsequent stores auto-link to this session |
| `get_session_context` | Retrieve the current session context |

### Cross-Project & Errors
| Tool | Description |
|---|---|
| `register_project` | Register a project in the global project registry |
| `sync_project_metadata` | Update the global context snapshot for a project (objective, stack, architecture, minimap) |
| `merge_projects` | Merge memories from one project into another in the global store |
| `search_global_errors` | Search the cross-project error database |
| `record_error` | Record a debugged error and its solution in the global error store |

### Symbols & Protocol
| Tool | Description |
|---|---|
| `attach_symbol` | Bind a memory to a code symbol (lang/path/kind/scope/name/arity) for graph-aware lookup |
| `get_by_symbol` | Retrieve all memories and session items linked to a symbol key |
| `get_memory_protocol` | Read the dual-save memory protocol flag (`dual` \| `pmc-only`) and runtime availability of PMC + Engram backends |

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
git clone https://github.com/adrianabadin/agent-memory-mcp.git
cd agent-memory-mcp
npm install
npm run dev          # Run with tsx (no build step)
npm run build        # Compile TypeScript to dist/
npm test             # Run all tests
npm run test:watch   # Run tests in watch mode
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
