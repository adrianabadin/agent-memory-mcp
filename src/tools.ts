import { z } from 'zod';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import {
  MEMORY_CATEGORIES,
  type MemoryStore,
  type SearchMode,
  type SearchResult,
  type StoreRequest,
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
  }
}
