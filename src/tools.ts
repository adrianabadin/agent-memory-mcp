import { z } from 'zod';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import {
  MEMORY_CATEGORIES,
  type MemoryStore,
  type SearchMode,
  type SearchResult,
  type StoreRequest,
  type EnrichSymbolResult,
} from './types.js';
import type { SqliteMemoryStore } from './sqlite-store.js';
import type { GlobalStore } from './global/global-store.js';

// ── Zod schemas ────────────────────────────────────────────────────

const categorySchema = z.enum(
  MEMORY_CATEGORIES as unknown as [string, ...string[]],
);

const tagsSchema = z.array(z.string()).describe('Free-form tags for organisation');

const provenanceSchema = z.object({
  origin: z.enum(['extracted', 'inferred', 'ambiguous', 'manual', 'unknown'])
    .describe('How this memory was produced: deterministic extraction, LLM inference, manual, or unknown'),
  sourceTool: z.string().optional().describe('Optional producing tool identifier (e.g. pmc-enrichment/ollama)'),
  sessionId: z.string().optional().describe('Optional producer session or run identifier'),
  contentHash: z.string().optional().describe('Optional content hash or source fragment hash for traceability'),
}).optional().describe('Optional provenance metadata for trust and traceability');

const filtersSchema = {
  category: categorySchema.optional().describe('Filter by category'),
  tags: z.array(z.string()).optional().describe('Filter: memory must have at least one of these tags'),
  after: z.string().optional().describe('Filter: created after this ISO 8601 date'),
  before: z.string().optional().describe('Filter: created before this ISO 8601 date'),
  limit: z.number().optional().describe('Max results to return (default 10)'),
};

const modeSchema = z
  .enum(['hybrid', 'keyword', 'semantic'])
  .optional()
  .describe('Search mode: hybrid (default), keyword-only, or semantic-only');

// ── Tool result helpers ────────────────────────────────────────────
// Returns plain objects compatible with the MCP SDK's CallToolResult
// (which requires an index signature [key: string]: unknown).

function success(data: unknown) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(data, null, 2) }] };
}

function error(message: string) {
  return { content: [{ type: 'text' as const, text: message }], isError: true };
}

// ── Handler factories ──────────────────────────────────────────────
// Each factory closes over a MemoryStore, returning a handler function
// that the MCP server can invoke. This keeps tools testable without
// an MCP server instance.

export function handleStore(store: MemoryStore) {
  return async (args: { content: string; category: string; tags: string[]; provenance?: StoreRequest['provenance'] }): Promise<ReturnType<typeof success>> => {
    try {
      const memory = await store.store(args as StoreRequest);
      return success(memory);
    } catch (err) {
      return error(`Failed to store memory: ${String(err)}`);
    }
  };
}

export function handleStoreBatch(store: MemoryStore) {
  return async (args: { memories: Array<{ content: string; category: string; tags: string[]; provenance?: StoreRequest['provenance'] }> }): Promise<ReturnType<typeof success>> => {
    try {
      const memories = await store.storeBatch(args.memories as StoreRequest[]);
      return success({ stored: memories.length, memories });
    } catch (err) {
      return error(`Failed to store batch: ${String(err)}`);
    }
  };
}

export function handleSearch(store: MemoryStore) {
  return async (args: {
    query: string;
    mode?: string;
    category?: string;
    tags?: string[];
    after?: string;
    before?: string;
    limit?: number;
    explain?: boolean;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const mode = (args.mode ?? 'hybrid') as SearchMode;
      const filters = {
        category: args.category as StoreRequest['category'],
        tags: args.tags,
        after: args.after,
        before: args.before,
        limit: args.limit,
        explain: args.explain,
      };

      if (args.explain && 'searchWithSummary' in store) {
        const { results, summary } = await (store as SqliteMemoryStore).searchWithSummary(
          args.query, mode, filters,
        );
        return success({ count: results.length, results, ...(summary ? { summary } : {}) });
      }

      const results = await store.search(args.query, mode, filters);
      return success({ count: results.length, results });
    } catch (err) {
      return error(`Search failed: ${String(err)}`);
    }
  };
}

export function handleRecall(store: MemoryStore) {
  return async (args: {
    topics: string[];
    include_recent?: number;
    limit_per_topic?: number;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const limitPerTopic = args.limit_per_topic ?? 5;
      const includeRecent = args.include_recent ?? 5;

      const byTopic: Record<string, SearchResult[]> = {};
      await Promise.all(
        args.topics.map(async (topic) => {
          byTopic[topic] = await store.search(topic, 'hybrid', { limit: limitPerTopic });
        }),
      );

      const recent = await store.listRecent(includeRecent);
      return success({ byTopic, recent });
    } catch (err) {
      return error(`Recall failed: ${String(err)}`);
    }
  };
}

export function handleFindRelated(store: MemoryStore) {
  return async (args: { memory_id: string; limit?: number }): Promise<ReturnType<typeof success>> => {
    try {
      const results = await store.findRelated(args.memory_id, args.limit ?? 5);
      return success({ count: results.length, results });
    } catch (err) {
      return error(`Find related failed: ${String(err)}`);
    }
  };
}

export function handleListRecent(store: MemoryStore) {
  return async (args: { limit?: number; category?: string }): Promise<ReturnType<typeof success>> => {
    try {
      const memories = await store.listRecent(
        args.limit ?? 10,
        args.category as StoreRequest['category'],
      );
      return success({ count: memories.length, memories });
    } catch (err) {
      return error(`List recent failed: ${String(err)}`);
    }
  };
}

export function handleUpdate(store: MemoryStore) {
  return async (args: {
    id: string;
    content?: string;
    category?: string;
    tags?: string[];
  }): Promise<ReturnType<typeof success>> => {
    try {
      const memory = await store.update(args.id, {
        content: args.content,
        category: args.category as StoreRequest['category'],
        tags: args.tags,
      });
      return success(memory);
    } catch (err) {
      return error(`Update failed: ${String(err)}`);
    }
  };
}

export function handleDelete(store: MemoryStore) {
  return async (args: { id: string }): Promise<ReturnType<typeof success>> => {
    try {
      await store.delete(args.id);
      return success({ deleted: true, id: args.id });
    } catch (err) {
      return error(`Delete failed: ${String(err)}`);
    }
  };
}

export function handleStats(store: MemoryStore) {
  return async (): Promise<ReturnType<typeof success>> => {
    try {
      const stats = await store.stats();
      return success(stats);
    } catch (err) {
      return error(`Stats failed: ${String(err)}`);
    }
  };
}

export function handlePrune(store: MemoryStore) {
  return async (args: {
    dryRun?: boolean;
    minStrength?: number;
    maxDormantDays?: number;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const result = await store.prune({
        dryRun: args.dryRun,
        minStrength: args.minStrength,
        maxDormantDays: args.maxDormantDays,
      });
      return success(result);
    } catch (err) {
      return error(`Prune failed: ${String(err)}`);
    }
  };
}

// ── Global context tool handlers ──────────────────────────────────

export function handleRegisterProject(globalStore: GlobalStore) {
  return async (args: {
    name: string;
    rootPath: string;
    objective?: string;
    stack?: unknown;
    architecture?: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const rec = globalStore.registerProject({
        name: args.name,
        rootPath: args.rootPath,
        objective: args.objective,
        stack: args.stack,
        architecture: args.architecture,
      });
      return success(rec);
    } catch (err) {
      return error(`register_project failed: ${String(err)}`);
    }
  };
}

export function handleSyncProjectMetadata(globalStore: GlobalStore) {
  return async (args: {
    projectId: string;
    objective?: string;
    stack?: unknown;
    architecture?: string;
    dependencies?: unknown;
    minimap?: unknown;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const rec = globalStore.syncMetadata(args.projectId, {
        objective: args.objective,
        stack: args.stack,
        architecture: args.architecture,
        dependencies: args.dependencies,
        minimap: args.minimap,
      });
      return success(rec);
    } catch (err) {
      return error(`sync_project_metadata failed: ${String(err)}`);
    }
  };
}

export function handleRecordError(globalStore: GlobalStore) {
  return async (args: {
    projectId: string;
    message: string;
    stack?: string;
    rootCause?: string;
    fix: string;
    files?: string[];
    tags?: string[];
    source: 'auto' | 'manual';
  }): Promise<ReturnType<typeof success>> => {
    try {
      const rec = await globalStore.recordError({
        projectId: args.projectId,
        message: args.message,
        stack: args.stack,
        rootCause: args.rootCause,
        fix: args.fix,
        files: args.files,
        tags: args.tags,
        source: args.source,
      });
      return success(rec);
    } catch (err) {
      return error(`record_error failed: ${String(err)}`);
    }
  };
}

export function handleSearchGlobalErrors(globalStore: GlobalStore) {
  return async (args: {
    query: string;
    projectId?: string;
    limit?: number;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const results = await globalStore.searchErrors(args.query, args.projectId, args.limit);
      return success({ count: results.length, results });
    } catch (err) {
      return error(`search_global_errors failed: ${String(err)}`);
    }
  };
}

// ── Lifecycle, topic, and session tool handlers ────────────────────

export function handleUpdateMemoryStatus(store: MemoryStore) {
  return async (args: {
    memory_id: string;
    status: 'active' | 'needs_review' | 'archived';
    review_reason?: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const memory = await store.updateMemoryStatus(args.memory_id, args.status, args.review_reason);
      return success(memory);
    } catch (err) {
      return error(`update_memory_status failed: ${String(err)}`);
    }
  };
}

export function handleUpsertTopicAlias(store: MemoryStore) {
  return async (args: {
    topic_key: string;
    memory_id: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      await store.upsertTopicAlias(args.topic_key, args.memory_id);
      return success({ topic_key: args.topic_key, memory_id: args.memory_id });
    } catch (err) {
      return error(`upsert_topic_alias failed: ${String(err)}`);
    }
  };
}

export function handleResolveTopic(store: MemoryStore) {
  return async (args: {
    topic_key: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const memory = await store.resolveTopic(args.topic_key);
      return success(memory);
    } catch (err) {
      return error(`resolve_topic failed: ${String(err)}`);
    }
  };
}

export function handleSetSessionContext(store: MemoryStore) {
  return async (args: {
    session_id: string;
    project_id: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      store.setSessionContext(args.session_id, args.project_id);
      return success({ sessionId: args.session_id, projectId: args.project_id });
    } catch (err) {
      return error(`set_session_context failed: ${String(err)}`);
    }
  };
}

export function handleGetSessionContext(store: MemoryStore) {
  return async (): Promise<ReturnType<typeof success>> => {
    try {
      const ctx = store.getSessionContext();
      return success(ctx);
    } catch (err) {
      return error(`get_session_context failed: ${String(err)}`);
    }
  };
}

export function handleMergeProjects(globalStore: GlobalStore) {
  return async (args: {
    source_project_id: string;
    target_project_id: string;
    dry_run?: boolean;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const dryRun = args.dry_run ?? true;
      const result = await globalStore.mergeProjects(args.source_project_id, args.target_project_id, dryRun);
      return success(result);
    } catch (err) {
      return error(`merge_projects failed: ${String(err)}`);
    }
  };
}

// ── Session ledger tool handlers ───────────────────────────────────

export function handleStoreSessionPrompt(store: MemoryStore) {
  return async (args: {
    session_id: string;
    raw_prompt: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const prompt = await store.storeSessionPrompt(args.session_id, args.raw_prompt);
      return success(prompt);
    } catch (err) {
      return error(`store_session_prompt failed: ${String(err)}`);
    }
  };
}

export function handleStoreSessionResponse(store: MemoryStore) {
  return async (args: {
    session_id: string;
    prompt_id: string;
    full_response: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const response = await store.storeSessionResponse(args.session_id, args.prompt_id, args.full_response);
      return success(response);
    } catch (err) {
      return error(`store_session_response failed: ${String(err)}`);
    }
  };
}

export function handleStoreSessionToolCall(store: MemoryStore) {
  return async (args: {
    session_id: string;
    prompt_id?: string;
    response_id?: string;
    tool_name: string;
    args_safe?: string;
    result_summary?: string;
    importance?: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const toolCall = await store.storeSessionToolCall({
        sessionId: args.session_id,
        promptId: args.prompt_id,
        responseId: args.response_id,
        toolName: args.tool_name,
        argsSafe: args.args_safe,
        resultSummary: args.result_summary,
        importance: args.importance,
      });
      return success(toolCall);
    } catch (err) {
      return error(`store_session_tool_call failed: ${String(err)}`);
    }
  };
}

export function handleStoreSessionSummary(store: MemoryStore) {
  return async (args: {
    session_id: string;
    prompt_id?: string;
    response_id?: string;
    summary: string;
    decisions?: string[];
    key_titles?: string[];
    memory_ids?: string[];
    artifact_refs?: string[];
  }): Promise<ReturnType<typeof success>> => {
    try {
      const summary = await store.storeSessionSummary({
        sessionId: args.session_id,
        promptId: args.prompt_id,
        responseId: args.response_id,
        summary: args.summary,
        decisions: args.decisions,
        keyTitles: args.key_titles,
        memoryIds: args.memory_ids,
        artifactRefs: args.artifact_refs,
      });
      return success(summary);
    } catch (err) {
      return error(`store_session_summary failed: ${String(err)}`);
    }
  };
}

export function handleGetSession(store: MemoryStore) {
  return async (args: {
    session_id: string;
    limit?: number;
    include_summary?: boolean;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const ledger = await store.getSession(args.session_id, {
        limit: args.limit,
        includeSummary: args.include_summary,
      });
      return success(ledger);
    } catch (err) {
      return error(`get_session failed: ${String(err)}`);
    }
  };
}

export function handleListSessions(store: MemoryStore) {
  return async (args: {
    limit?: number;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const sessions = await store.listSessions(args.limit);
      return success({ count: sessions.length, sessions });
    } catch (err) {
      return error(`list_sessions failed: ${String(err)}`);
    }
  };
}

export function handleLinkPromptToMemory(store: MemoryStore) {
  return async (args: {
    prompt_id: string;
    memory_id: string;
    relation: 'created_from' | 'updated_by' | 'referenced';
    response_id?: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      await store.linkPromptToMemory(args.prompt_id, args.memory_id, args.relation, args.response_id);
      return success({ linked: true, prompt_id: args.prompt_id, memory_id: args.memory_id });
    } catch (err) {
      return error(`link_prompt_to_memory failed: ${String(err)}`);
    }
  };
}

// ── Session forget / purge (explicit soft vs hard delete) ──────────
//
// `forget_session_item` is a SOFT delete: it sets status='forgotten' so the
// row is hidden from default retrieval/search but remains recoverable until
// it is explicitly purged. `purge_session_item` is a HARD, DESTRUCTIVE
// delete: it removes the row AND its embedding permanently. Hard purge is
// never invoked automatically by decay/retention — only by this explicit
// tool. Both surface store-level NOT_FOUND / AMBIGUOUS errors as safe
// `isError` results rather than throwing out of the MCP call.

export function handleForgetSessionItem(store: MemoryStore) {
  return async (args: { id: string }): Promise<ReturnType<typeof success>> => {
    try {
      const result = await store.forgetSessionItem(args.id);
      return success(result);
    } catch (err) {
      return error(`forget_session_item failed: ${String(err)}`);
    }
  };
}

export function handlePurgeSessionItem(store: MemoryStore) {
  return async (args: { id: string }): Promise<ReturnType<typeof success>> => {
    try {
      const result = await store.purgeSessionItem(args.id);
      return success(result);
    } catch (err) {
      return error(`purge_session_item failed: ${String(err)}`);
    }
  };
}

// ── Symbol-linked memory context tools (v7) ────────────────────────

export function handleAttachSymbol(store: MemoryStore) {
  return async (args: {
    symbol_key: string;
    memory_id?: string;
    session_item_id?: string;
  }): Promise<ReturnType<typeof success>> => {
    try {
      if (!args.memory_id && !args.session_item_id) {
        return error('Either memory_id or session_item_id must be provided');
      }
      const result = await store.attachSymbol({
        symbolKey: args.symbol_key,
        memoryId: args.memory_id,
        sessionItemId: args.session_item_id,
      });
      return success(result);
    } catch (err) {
      return error(`attach_symbol failed: ${String(err)}`);
    }
  };
}

export function handleGetBySymbol(store: MemoryStore) {
  return async (args: {
    symbol_key: string;
    sources?: string[];
    limit?: number;
  }): Promise<ReturnType<typeof success>> => {
    try {
      const results = await store.getBySymbol(args.symbol_key, {
        sources: args.sources,
        limit: args.limit,
      });
      return success({ count: results.length, results });
    } catch (err) {
      return error(`get_by_symbol failed: ${String(err)}`);
    }
  };
}

const MAX_ENRICH_SYMBOL_KEYS = 50;

export function handleEnrichSymbols(store: MemoryStore, enrichmentResolver?: (symbolKey: string) => Promise<EnrichSymbolResult | null>) {
  return async (args: {
    symbol_keys: string[];
  }): Promise<ReturnType<typeof success>> => {
    try {
      if (!Array.isArray(args.symbol_keys) || args.symbol_keys.length === 0) {
        return error('symbol_keys must be a non-empty array');
      }
      if (args.symbol_keys.length > MAX_ENRICH_SYMBOL_KEYS) {
        return error(`symbol_keys exceeds maximum of ${MAX_ENRICH_SYMBOL_KEYS}`);
      }

      const results: EnrichSymbolResult[] = [];
      for (const key of args.symbol_keys) {
        // Try the enrichment resolver first (PMC disk-based lookup).
        if (enrichmentResolver) {
          try {
            const resolved = await enrichmentResolver(key);
            if (resolved) {
              results.push(resolved);
              continue;
            }
          } catch {
            // Resolver failed — fall through to not-found.
          }
        }

        // Fallback: look up memories with matching symbolKey via the store.
        const linked = await store.getBySymbol(key, { sources: ['memory'], limit: 1 });
        if (linked.length > 0) {
          results.push({
            symbolKey: key,
            found: true,
            raw: linked[0].content,
          });
        } else {
          results.push({ symbolKey: key, found: false });
        }
      }

      return success({ results });
    } catch (err) {
      return error(`enrich_symbols failed: ${String(err)}`);
    }
  };
}

// ── Topic key suggestion (deterministic; queued version in PR-5) ───

/**
 * Pure slugifier for topic keys: lowercase, non-alphanumeric runs → single
 * hyphen, trimmed, capped at 60 chars. Falls back to "memory" for empty
 * or symbol-only input.
 */
export function slugifyTopicKey(content: string): string {
  const slug = content
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
  return slug || 'memory';
}

export function handleSuggestTopicKey(store: MemoryStore) {
  return async (args: {
    content: string;
    category?: string;
    tags?: string[];
  }): Promise<ReturnType<typeof success>> => {
    try {
      const base = slugifyTopicKey(args.content);
      let candidate = base;
      let suffix = 1;
      // Deduplicate against existing topic keys
      while (await store.resolveTopic(candidate)) {
        suffix++;
        candidate = `${base}-${suffix}`;
      }
      return success({ topic_key: candidate });
    } catch (err) {
      return error(`suggest_topic_key failed: ${String(err)}`);
    }
  };
}

// ── Registration ───────────────────────────────────────────────────

export function registerTools(server: McpServer, store: MemoryStore, globalStore?: GlobalStore): void {
  // ── Storage tools ──

  server.tool(
    'store',
    'Store a single memory with content, category, and tags. Returns the stored memory with its ID.',
    {
      content: z.string().describe('The memory content — what you learnt, observed, or want to remember'),
      category: categorySchema.describe('Memory category'),
      tags: tagsSchema,
      provenance: provenanceSchema,
    },
    handleStore(store),
  );

  server.tool(
    'store_batch',
    'Store multiple memories in one call. Use at end-of-session to capture everything learnt. Returns all stored memories.',
    {
      memories: z.array(z.object({
        content: z.string().describe('The memory content'),
        category: categorySchema,
        tags: tagsSchema,
        provenance: provenanceSchema,
      })).describe('Array of memories to store'),
    },
    handleStoreBatch(store),
  );

  // ── Search tools ──

  server.tool(
    'search',
    'Search memories by meaning and/or keywords. Default mode is hybrid (BM25 + vector). Returns ranked results with relevance scores.',
    {
      query: z.string().describe('What to search for — a concept, phrase, or question'),
      mode: modeSchema,
      ...filtersSchema,
      explain: z.boolean().optional().describe('Debugging aid; adds per-stage scores (keyword rank/bm25, semantic rank/cosine, RRF score, decay factor) to each result. Hybrid mode only; ignored otherwise.'),
    },
    handleSearch(store),
  );

  server.tool(
    'recall',
    'Multi-topic contextual recall — the morning coffee tool. Searches multiple topics in parallel and includes recent memories. Use at session start to restore context efficiently.',
    {
      topics: z.array(z.string()).describe('List of topics to search for'),
      include_recent: z.number().optional().describe('Number of recent memories to include (default 5)'),
      limit_per_topic: z.number().optional().describe('Max results per topic (default 5)'),
    },
    handleRecall(store),
  );

  server.tool(
    'find_related',
    'Find memories similar to a specific memory. Use for associative exploration — "what else do I know that connects to this?"',
    {
      memory_id: z.string().describe('ID of the memory to find relatives of'),
      limit: z.number().optional().describe('Max related memories to return (default 5)'),
    },
    handleFindRelated(store),
  );

  server.tool(
    'list_recent',
    'List most recent memories, optionally filtered by category. Use to see what you have been learning lately.',
    {
      limit: z.number().optional().describe('Max memories to return (default 10)'),
      category: categorySchema.optional().describe('Filter by category'),
    },
    handleListRecent(store),
  );

  // ── Management tools ──

  server.tool(
    'update',
    'Update an existing memory — change its content, category, or tags. If content changes, the embedding is regenerated automatically.',
    {
      id: z.string().describe('ID of the memory to update'),
      content: z.string().optional().describe('New content (triggers re-embedding)'),
      category: categorySchema.optional().describe('New category'),
      tags: tagsSchema.optional().describe('New tags (replaces existing)'),
    },
    handleUpdate(store),
  );

  server.tool(
    'delete',
    'Permanently remove a memory by ID.',
    {
      id: z.string().describe('ID of the memory to delete'),
    },
    handleDelete(store),
  );

  server.tool(
    'stats',
    'Get memory database statistics: total count, breakdown by category, oldest and newest timestamps, access patterns, and prune-eligible counts.',
    {},
    handleStats(store),
  );

  server.tool(
    'prune',
    'Prune low-strength and dormant memories. Dry-run by default — shows candidates without deleting. Set dryRun: false to actually prune. Evergreen memories are always preserved.',
    {
      dryRun: z.boolean().optional().default(true).describe('Preview mode — show what would be pruned without deleting'),
      minStrength: z.number().optional().default(0.05).describe('Prune memories with strength below this value'),
      maxDormantDays: z.number().optional().default(90).describe('Prune never-accessed memories older than this many days'),
    },
    handlePrune(store),
  );

  // ── Lifecycle & topic tools ──

  server.tool(
    'update_memory_status',
    'Transition a memory between lifecycle states: active -> needs_review -> archived.',
    {
      memory_id: z.string().describe('ID of the memory to transition'),
      status: z.enum(['active', 'needs_review', 'archived']).describe('Target lifecycle status'),
      review_reason: z.string().optional().describe('Reason for the transition (required for needs_review and archived)'),
    },
    handleUpdateMemoryStatus(store),
  );

  server.tool(
    'upsert_topic_alias',
    'Create or update a stable topic key that points to a memory.',
    {
      topic_key: z.string().describe('Stable key, e.g. "architecture/auth-model"'),
      memory_id: z.string().describe('ID of the memory the key resolves to'),
    },
    handleUpsertTopicAlias(store),
  );

  server.tool(
    'resolve_topic',
    'Find a memory by its topic key. Returns null if not found.',
    {
      topic_key: z.string().describe('The topic key to resolve'),
    },
    handleResolveTopic(store),
  );

  server.tool(
    'suggest_topic_key',
    'Suggest a deterministic topic key derived from content, deduplicated against existing keys.',
    {
      content: z.string().describe('Content to derive the topic key from'),
      category: z.string().optional(),
      tags: z.array(z.string()).optional(),
    },
    handleSuggestTopicKey(store),
  );

  // ── Session context tools ──

  server.tool(
    'set_session_context',
    'Set the active session context. All subsequent memory stores are auto-linked to this session.',
    {
      session_id: z.string().describe('Active session identifier'),
      project_id: z.string().describe('Project the session belongs to'),
    },
    handleSetSessionContext(store),
  );

  server.tool(
    'get_session_context',
    'Get the currently active session context. Returns null if no session is active.',
    {},
    handleGetSessionContext(store),
  );

  // ── Session ledger tools ──

  server.tool(
    'store_session_prompt',
    'Record a user prompt in the active session ledger. Returns the prompt ID.',
    {
      session_id: z.string().describe('Session the prompt belongs to'),
      raw_prompt: z.string().describe('The raw user prompt text'),
    },
    handleStoreSessionPrompt(store),
  );

  server.tool(
    'store_session_response',
    'Record an assistant response linked to a prompt. Returns the response ID.',
    {
      session_id: z.string().describe('Session the response belongs to'),
      prompt_id: z.string().describe('ID of the prompt this response answers'),
      full_response: z.string().describe('The full assistant response text'),
    },
    handleStoreSessionResponse(store),
  );

  server.tool(
    'store_session_tool_call',
    'Record an important tool call during a session.',
    {
      session_id: z.string().describe('Session the tool call belongs to'),
      prompt_id: z.string().optional().describe('Prompt that triggered the call'),
      response_id: z.string().optional().describe('Response that triggered the call'),
      tool_name: z.string().describe('Name of the tool invoked'),
      args_safe: z.string().optional().describe('Sanitised arguments (no secrets)'),
      result_summary: z.string().optional().describe('Short summary of the result'),
      importance: z.enum(['high', 'normal', 'low']).optional().describe('Importance level'),
    },
    handleStoreSessionToolCall(store),
  );

  server.tool(
    'store_session_summary',
    'Store a generated session summary. The summary content should already be generated by the PMC local model pipeline.',
    {
      session_id: z.string().describe('Session the summary covers'),
      prompt_id: z.string().optional(),
      response_id: z.string().optional(),
      summary: z.string().describe('Generated summary text'),
      decisions: z.array(z.string()).optional().describe('Key decisions made during the session'),
      key_titles: z.array(z.string()).optional().describe('Titles of key artifacts'),
      memory_ids: z.array(z.string()).optional().describe('Memory IDs created/updated during the session'),
      artifact_refs: z.array(z.string()).optional().describe('References to artifacts (files, specs, etc.)'),
    },
    handleStoreSessionSummary(store),
  );

  server.tool(
    'get_session',
    'Retrieve the full session ledger: prompts, responses, tool calls, and optionally summaries.',
    {
      session_id: z.string().describe('Session to retrieve'),
      limit: z.number().optional().describe('Max items per type'),
      include_summary: z.boolean().optional().describe('Include summaries (default true)'),
    },
    handleGetSession(store),
  );

  server.tool(
    'list_sessions',
    'List recent sessions with prompt count and last activity time.',
    {
      limit: z.number().optional().describe('Max sessions to return'),
    },
    handleListSessions(store),
  );

  server.tool(
    'link_prompt_to_memory',
    'Explicitly link a session prompt/response to a memory.',
    {
      prompt_id: z.string().describe('Prompt to link'),
      memory_id: z.string().describe('Memory to link to'),
      relation: z.enum(['created_from', 'updated_by', 'referenced']).describe('Relationship type'),
      response_id: z.string().optional().describe('Optional response ID for finer linking'),
    },
    handleLinkPromptToMemory(store),
  );

  // ── Session forget / purge (explicit lifecycle controls) ──
  // forget = SOFT delete (status='forgotten', row hidden but recoverable).
  // purge  = HARD, DESTRUCTIVE delete (row + embedding removed permanently;
  //          never triggered automatically by decay/retention).

  server.tool(
    'forget_session_item',
    'Soft-delete a session-ledger record (prompt, response, tool call, or summary) by id: sets status="forgotten" so it is hidden from default retrieval/search but remains recoverable until explicitly purged. Returns { type, success }. Rejects unknown or ambiguous ids.',
    {
      id: z.string().describe('ID of the session prompt / response / tool call / summary to soft-forget'),
    },
    handleForgetSessionItem(store),
  );

  server.tool(
    'purge_session_item',
    'PERMANENTLY and destructively delete a session-ledger record (prompt, response, tool call, or summary) by id, including its embedding. This is irreversible and never runs automatically — unlike forget, purged rows cannot be recovered. Returns { type, success }. Rejects unknown or ambiguous ids.',
    {
      id: z.string().describe('ID of the session prompt / response / tool call / summary to hard-purge (irreversible)'),
    },
    handlePurgeSessionItem(store),
  );

  // ── Symbol-linked memory context tools (v7) ──────────────────────

  server.tool(
    'attach_symbol',
    'Attach a symbol key to an existing memory or session-ledger record. Upserts — overwrites previous symbol_key on the same record. Returns { id, symbolKey }.',
    {
      symbol_key: z.string().describe('Soft symbol key (lang|path|kind|scope|name|arity)'),
      memory_id: z.string().optional().describe('ID of the memory to attach the symbol key to'),
      session_item_id: z.string().optional().describe('ID of the session prompt/response/tool_call/summary to attach the symbol key to'),
    },
    handleAttachSymbol(store),
  );

  server.tool(
    'get_by_symbol',
    'Retrieve all memories and session items linked to a symbol key. Returns { count, results } with per-record source, content, and metadata. Empty results for unknown keys (not an error).',
    {
      symbol_key: z.string().describe('Soft symbol key to look up'),
      sources: z.array(z.enum(['memory', 'session_summary', 'session_prompt', 'session_response', 'session_tool_call'])).optional().describe('Sources to search (default: memory + session_summary)'),
      limit: z.number().optional().describe('Max results to return (default 50)'),
    },
    handleGetBySymbol(store),
  );

  server.tool(
    'enrich_symbols',
    'Fetch semantic enrichment for one or more symbol keys. Returns per-symbol results with found flag and optional raw/structured enrichment. Batch never fails — unknown symbols return { found: false }.',
    {
      symbol_keys: z.array(z.string()).min(1).max(50).describe('Array of symbol keys to enrich (max 50)'),
    },
    handleEnrichSymbols(store),
  );

  // ── Global context tools (registered only when globalStore is provided) ──
  if (globalStore) {
    server.tool(
      'register_project',
      'Register a project in the global context store. Idempotent — safe to call on every pmc init.',
      {
        name: z.string().describe('Project name (e.g. repo name)'),
        rootPath: z.string().describe('Absolute path to the project root'),
        objective: z.string().optional().describe('One-sentence project objective'),
        stack: z.unknown().optional().describe('Languages/frameworks as JSON object'),
        architecture: z.string().optional().describe('Architecture summary'),
      },
      handleRegisterProject(globalStore),
    );

    server.tool(
      'sync_project_metadata',
      'Update the global context snapshot for a project: objective, architecture, dependencies, minimap.',
      {
        projectId: z.string().describe('Project ID (from register_project)'),
        objective: z.string().optional(),
        stack: z.unknown().optional(),
        architecture: z.string().optional(),
        dependencies: z.unknown().optional(),
        minimap: z.unknown().optional(),
      },
      handleSyncProjectMetadata(globalStore),
    );

    server.tool(
      'record_error',
      'Record a debugged error and its solution in the global error database. Deduplicated by (message + root_cause) per project.',
      {
        projectId: z.string().describe('Project ID where this error occurred'),
        message: z.string().describe('Error message or exception text'),
        stack: z.string().optional().describe('Stack trace'),
        rootCause: z.string().optional().describe('Root cause analysis'),
        fix: z.string().describe('Solution that resolved the error'),
        files: z.array(z.string()).optional().describe('Files affected'),
        tags: z.array(z.string()).optional(),
        source: z.enum(['auto', 'manual']).describe('auto = from KNOWN-ISSUES-AND-FIXES.md, manual = live'),
      },
      handleRecordError(globalStore),
    );

    server.tool(
      'search_global_errors',
      'Search the global cross-project error database for similar errors and their solutions. Hybrid FTS5 + semantic search.',
      {
        query: z.string().describe('Error description or symptom to search for'),
        projectId: z.string().optional().describe('Filter to a specific project ID'),
        limit: z.number().optional().describe('Max results (default 10)'),
      },
      handleSearchGlobalErrors(globalStore),
    );

    server.tool(
      'merge_projects',
      'Consolidate duplicate project names. Moves memories/errors from source project to target project.',
      {
        source_project_id: z.string().describe('Project to move data FROM'),
        target_project_id: z.string().describe('Project to move data TO'),
        dry_run: z.boolean().optional().default(true).describe('Preview without modifying (default true)'),
      },
      handleMergeProjects(globalStore),
    );
  }
}
