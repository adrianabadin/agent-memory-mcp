# Changelog

All notable changes to `@aabadin/agent-memory-mcp` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.3.0] — 2026-06-24

### Added (since 2.1.0)

This release rolls up two changes from the prior SDD cycle plus the
post-verify fixes from `engram-parity-final`:

#### From `engram-features-to-pmc` (SDD, archived 2026-06-22)

- **Memory lifecycle** — `update_memory_status` tool + `MemoryStatus` enum
  (`active` | `needs_review` | `archived`). Transitions are validated;
  archived is terminal. `needs_review` requires a reason.
- **Topic aliases** — `upsert_topic_alias`, `resolve_topic`, `suggest_topic_key`.
  Re-storing with the same topic updates the existing memory instead of
  inserting a duplicate.
- **Session ledger** — `store_session_prompt`, `store_session_response`,
  `store_session_tool_call`, `store_session_summary`, `get_session`,
  `list_sessions`, `link_prompt_to_memory`. Sessions auto-link to active
  `set_session_context`. Embeddings persisted as float32 BLOBs (1024-d).
- **Project merging** — `merge_projects` for consolidating memories across
  projects in the global store.
- **Cross-project error store** — `search_global_errors` and `record_error`.
- **Symbol attachment** — `attach_symbol` + `get_by_symbol` bridge memories
  to the code graph (lang/path/kind/scope/name/arity).
- **Upsert semantics fix** — `store()` no longer overwrites `created_at`,
  `version`, or `status` on re-store; preserves provenance intelligently.

#### From `engram-parity-final` (SDD, archived 2026-06-24)

- **Pin / unpin** — `pin` and `unpin` MCP tools + `is_pinned` column
  (migration v8). Pinned memories sort first in search/list_recent/recall
  and are exempt from temporal decay.
- **Cross-project search** — `search` accepts `all_projects: true` to
  query across multiple project memory DBs via the
  `siblingStores` / `searchAllProjects` plumbing. Pinned-first sort
  applies across the merged result set.
- **Conflict resolution** — `store()` accepts opt-in `detect_conflicts`
  flag. When set, cosine-similarity + FTS title-overlap threshold check
  fires BEFORE the upsert; on a hit the call returns
  `{ judgment_required, candidates }` without persisting. The new
  `compare` and `judge` MCP tools let the LLM drive relation
  classification across 6 types: `related`, `compatible`, `scoped`,
  `conflicts_with`, `supersedes`, `not_conflict`. The `memory_relations`
  table (migration v9) persists verdicts. `supersedes` cascades to
  archive the older memory.
- **Review queue** — `review_queue` MCP tool returns memories with
  `status='needs_review'`, ordered by `reviewed_at ASC`. Decay wiring
  flags memories when they transition to `dimmed`, `shadow`, or
  `gist_only` (skipping pinned and `non_forgettable=1`).
- **Memory protocol flag** — `get_memory_protocol` MCP tool exposes the
  runtime dual-save protocol flag (`dual` | `pmc-only`) plus the
  availability of the PMC and Engram backends. Used by tooling that
  decides whether to mirror saves to Engram.

### Changed

- `MemoryStore.store()` return type widened from `Promise<Memory>` to
  `Promise<Memory | StoreResult>` (cross-cutting when conflict detection
  is enabled). Backward compatible: callers that don't pass
  `detect_conflicts` still receive `Memory` directly.
- `applyFiltersAndDecay` adds pinned-first sort and the `all_projects`
  branch. Default search behavior unchanged.
- `decay.ts` skips pinned and `non_forgettable=1` memories before
  advancing decay stage; on a stage crossing it auto-flags the memory
  for review.

### Fixed

- `context-composition.test.mjs` in the host `pmc-graph-explorer` package
  had 6 pre-existing failures (missing Semantic Memory section in
  `context-renderer.mjs`). Fixed as a courtesy for downstream consumers;
  the section renders memories embedded in the active project context.
- `store(detect_conflicts=true)` originally performed the upsert before
  flagging collisions (semantic deviation from the spec). Now checks
  collisions first; the `not_conflict` verdict + retry pattern is the
  intended flow.

### Test counts

- 761 vitest tests passing across 38 test files (Strict TDD).
- 697 PMC CLI tests passing (host package, includes CLI smoke tests for
  `migrate-from-engram` and `compare-backends`).
- 32 / 32 spec scenarios compliant.

### Migration notes

- v2.x → v2.3.0: additive. The schema gains `is_pinned INTEGER NOT NULL
  DEFAULT 0` and the `memory_relations` table. Both are added by
  idempotent migration blocks (v8 + v9). Existing rows default to
  `is_pinned=0`. The `user_version` PRAGMA is set to 9 after the first
  successful run on a v7-or-newer database.
- The single-DB SQLite file path is unchanged; no re-embed is required.

## [2.2.0] — 2026-06-23

- Added `get_memory_protocol` MCP tool exposing the runtime
  dual-save protocol flag and backend availability.

## [2.1.0] — 2026-05 (approx)

- Prior stable baseline before the SDD cycles. Hybrid search, local
  embeddings, hardcopy backup, temporal decay.