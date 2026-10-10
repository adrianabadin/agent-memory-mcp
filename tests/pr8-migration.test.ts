// tests/pr8-migration.test.ts
//
// T-009: migrate-from-engram CLI — migration logic tests
// Strict TDD: these tests are written BEFORE implementation.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MockEmbedder } from './mocks.js';
import type { Embedder, MemoryStore } from '../src/types.js';

// ── Mock Engram DB (simulates node:sqlite DatabaseSync) ──────────

interface EngramObservation {
  id: number;
  sync_id: string;
  session_id: string;
  type: string;
  title: string;
  content: string;
  tool_name: string | null;
  project: string | null;
  scope: string;
  topic_key: string | null;
  normalized_hash: string | null;
  revision_count: number;
  duplicate_count: number;
  last_seen_at: string | null;
  pinned: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
  review_after: string | null;
  expires_at: string | null;
  embedding: Buffer | null;
  embedding_model: string | null;
  embedding_created_at: string | null;
}

interface EngramUserPrompt {
  id: number;
  sync_id: string;
  session_id: string;
  content: string;
  project: string | null;
  created_at: string;
}

interface EngramMemoryRelation {
  id: number;
  sync_id: string;
  source_id: string;
  target_id: string;
  relation: string;
  reason: string | null;
  evidence: string | null;
  confidence: number | null;
  judgment_status: string;
  marked_by_actor: string | null;
  marked_by_kind: string | null;
  marked_by_model: string | null;
  session_id: string | null;
  superseded_at: string | null;
  superseded_by_relation_id: number | null;
  created_at: string;
  updated_at: string;
}

function makeObservation(overrides: Partial<EngramObservation> = {}): EngramObservation {
  return {
    id: 1,
    sync_id: 'obs-abc123',
    session_id: 'ses_test123',
    type: 'decision',
    title: 'Test decision',
    content: 'Decided to use vitest for testing',
    tool_name: null,
    project: 'test-project',
    scope: 'project',
    topic_key: 'testing/vitest',
    normalized_hash: null,
    revision_count: 1,
    duplicate_count: 1,
    last_seen_at: null,
    pinned: 0,
    created_at: '2026-06-22 02:17:50',
    updated_at: '2026-06-22 02:17:50',
    deleted_at: null,
    review_after: null,
    expires_at: null,
    embedding: null,
    embedding_model: null,
    embedding_created_at: null,
    ...overrides,
  };
}

function makeUserPrompt(overrides: Partial<EngramUserPrompt> = {}): EngramUserPrompt {
  return {
    id: 1,
    sync_id: 'prompt-abc123',
    session_id: 'ses_test123',
    content: 'Help me fix this bug',
    project: 'test-project',
    created_at: '2026-06-22 02:11:41',
    ...overrides,
  };
}

function makeRelation(overrides: Partial<EngramMemoryRelation> = {}): EngramMemoryRelation {
  return {
    id: 1,
    sync_id: 'rel-abc123',
    source_id: 'obs-001',
    target_id: 'obs-002',
    relation: 'not_conflict',
    reason: null,
    evidence: null,
    confidence: null,
    judgment_status: 'judged',
    marked_by_actor: 'engram',
    marked_by_kind: null,
    marked_by_model: null,
    session_id: null,
    superseded_at: null,
    superseded_by_relation_id: null,
    created_at: '2026-06-22 02:17:50',
    updated_at: '2026-06-22 02:17:50',
    ...overrides,
  };
}

// ── Mock PMC store (captures writes) ──────────────────────────────

interface PmcStoreWrite {
  content: string;
  category: string;
  tags: string[];
  provenance?: { origin: string; sessionId?: string; sourceTool?: string };
}

interface PmcSessionPromptWrite {
  id: string;
  sessionId: string;
  rawPrompt: string;
  createdAt: string;
}

interface PmcSessionSummaryWrite {
  id: string;
  sessionId: string;
  summary: string;
  createdAt: string;
}

interface PmcRelationWrite {
  sourceId: string;
  targetId: string;
  relation: string;
  reason: string | null;
  confidence: number | null;
  judgmentStatus: string;
}

class MockPmcStore {
  readonly memoryWrites: PmcStoreWrite[] = [];
  readonly promptWrites: PmcSessionPromptWrite[] = [];
  readonly summaryWrites: PmcSessionSummaryWrite[] = [];
  readonly relationWrites: PmcRelationWrite[] = [];
  readonly symbolLinks: { memoryId: string; symbolKey: string }[] = [];
  embedCallCount = 0;

  async storeBatch(requests: Array<{ content: string; category: string; tags: string[]; provenance?: unknown }>) {
    for (const r of requests) {
      this.memoryWrites.push({
        content: r.content,
        category: r.category,
        tags: r.tags,
        provenance: r.provenance as PmcStoreWrite['provenance'],
      });
    }
    // Return fake memory IDs
    return requests.map((_, i) => ({
      id: `pmc-mem-${this.memoryWrites.length - requests.length + i}`,
      content: '',
      category: 'learning' as const,
      tags: [],
      createdAt: '',
      updatedAt: '',
      version: 1,
    }));
  }

  async storeSessionPrompt(id: string, sessionId: string, rawPrompt: string, createdAt: string) {
    this.promptWrites.push({ id, sessionId, rawPrompt, createdAt });
  }

  async storeSessionSummary(data: PmcSessionSummaryWrite) {
    this.summaryWrites.push(data);
  }

  async storeRelation(data: PmcRelationWrite) {
    this.relationWrites.push(data);
  }

  async attachSymbol(memoryId: string, symbolKey: string) {
    this.symbolLinks.push({ memoryId, symbolKey });
  }

  async embed(text: string): Promise<number[]> {
    this.embedCallCount++;
    return new Array(1024).fill(0).map((_, i) => (text.charCodeAt(i % text.length) / 1000));
  }
}

// ── Import the module under test ──────────────────────────────────
// We need to import the migration logic. Since it may not exist yet,
// we import from the expected path.

import {
  migrateObservations,
  migrateSessionLedger,
  migrateRelations,
  linkSymbols,
  mapEngramTypeToPmcCategory,
  runMigration,
} from '../src/migration.js';

describe('T-009: migrate-from-engram — migration logic', () => {
  let pmcStore: MockPmcStore;
  let embedder: MockEmbedder;

  beforeEach(() => {
    pmcStore = new MockPmcStore();
    embedder = new MockEmbedder();
  });

  // ── Type mapping (uses existing mapEngramTypeToPmcCategory) ──

  describe('mapEngramTypeToPmcCategory (re-export)', () => {
    it('maps bugfix → bug-fix', () => {
      expect(mapEngramTypeToPmcCategory('bugfix')).toBe('bug-fix');
    });

    it('maps decision → architecture', () => {
      expect(mapEngramTypeToPmcCategory('decision')).toBe('architecture');
    });

    it('maps architecture → architecture', () => {
      expect(mapEngramTypeToPmcCategory('architecture')).toBe('architecture');
    });

    it('maps discovery → learning', () => {
      expect(mapEngramTypeToPmcCategory('discovery')).toBe('learning');
    });

    it('maps pattern → learning', () => {
      expect(mapEngramTypeToPmcCategory('pattern')).toBe('learning');
    });

    it('maps config → tool-usage', () => {
      expect(mapEngramTypeToPmcCategory('config')).toBe('tool-usage');
    });

    it('maps preference → personal', () => {
      expect(mapEngramTypeToPmcCategory('preference')).toBe('personal');
    });

    it('maps unknown types → learning', () => {
      expect(mapEngramTypeToPmcCategory('documentation')).toBe('learning');
      expect(mapEngramTypeToPmcCategory(null)).toBe('learning');
      expect(mapEngramTypeToPmcCategory(undefined)).toBe('learning');
    });
  });

  // ── Observation migration ──────────────────────────────────────

  describe('migrateObservations', () => {
    it('migrates observations to PMC memories with correct type mapping', async () => {
      const observations = [
        makeObservation({ id: 1, type: 'bugfix', title: 'Fixed auth bug', content: 'JWT was expired' }),
        makeObservation({ id: 2, type: 'decision', title: 'Chose vitest', content: 'Vitest is faster' }),
        makeObservation({ id: 3, type: 'discovery', title: 'Found issue', content: 'SQLite WAL mode' }),
      ];

      const result = await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(result.migrated).toBe(3);
      expect(result.skipped).toBe(0);
      expect(result.errors).toBe(0);

      expect(pmcStore.memoryWrites).toHaveLength(3);
      expect(pmcStore.memoryWrites[0].category).toBe('bug-fix');
      expect(pmcStore.memoryWrites[1].category).toBe('architecture');
      expect(pmcStore.memoryWrites[2].category).toBe('learning');
    });

    it('composes title + content for PMC memory content', async () => {
      const observations = [
        makeObservation({ title: 'Fixed auth', content: 'JWT expired issue' }),
      ];

      await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(pmcStore.memoryWrites[0].content).toContain('Fixed auth');
      expect(pmcStore.memoryWrites[0].content).toContain('JWT expired issue');
    });

    it('preserves topic_key as tag segments', async () => {
      const observations = [
        makeObservation({ topic_key: 'architecture/auth-model', project: 'my-project' }),
      ];

      await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      const tags = pmcStore.memoryWrites[0].tags;
      expect(tags).toContain('my-project');
      expect(tags).toContain('architecture');
      expect(tags).toContain('auth-model');
    });

    it('re-embeds content using PMC embedder (not Engram embeddings)', async () => {
      const observations = [
        makeObservation({ content: 'Test content for re-embedding' }),
      ];

      await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      // Embedder should have been called (re-embedding, not copying Engram's blob)
      expect(embedder.callLog.length).toBeGreaterThan(0);
      // The composed content includes title + body
      const embeddedText = embedder.callLog[0];
      expect(embeddedText).toContain('Test content for re-embedding');
    });

    it('skips deleted observations (deleted_at IS NOT NULL)', async () => {
      const observations = [
        makeObservation({ id: 1, deleted_at: null }),
        makeObservation({ id: 2, deleted_at: '2026-06-23 10:00:00' }),
      ];

      const result = await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(result.migrated).toBe(1);
      expect(result.skipped).toBe(1);
      expect(pmcStore.memoryWrites).toHaveLength(1);
    });

    it('dry-run reports counts without writing to PMC', async () => {
      const observations = [
        makeObservation({ id: 1 }),
        makeObservation({ id: 2 }),
        makeObservation({ id: 3, deleted_at: '2026-06-23' }),
      ];

      const result = await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: true });

      expect(result.migrated).toBe(2);
      expect(result.skipped).toBe(1);
      expect(pmcStore.memoryWrites).toHaveLength(0); // no writes in dry-run
    });

    it('respects --limit N', async () => {
      const observations = [
        makeObservation({ id: 1 }),
        makeObservation({ id: 2 }),
        makeObservation({ id: 3 }),
      ];

      const result = await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false, limit: 2 });

      expect(result.migrated).toBe(2);
      expect(pmcStore.memoryWrites).toHaveLength(2);
    });

    it('continues on individual observation error', async () => {
      const observations = [
        makeObservation({ id: 1, content: 'Good observation' }),
        makeObservation({ id: 2, content: '' }), // might cause issues
        makeObservation({ id: 3, content: 'Another good one' }),
      ];

      // Even if one fails, the others should still migrate
      const result = await migrateObservations(observations, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(result.migrated).toBeGreaterThanOrEqual(2);
    });
  });

  // ── Session ledger migration ───────────────────────────────────

  describe('migrateSessionLedger', () => {
    it('migrates user_prompts to PMC session_prompts', async () => {
      const prompts = [
        makeUserPrompt({ id: 1, sync_id: 'prompt-001', session_id: 'ses_abc', content: 'Fix the bug', project: 'my-proj' }),
        makeUserPrompt({ id: 2, sync_id: 'prompt-002', session_id: 'ses_abc', content: 'Now test it', project: 'my-proj' }),
      ];

      const result = await migrateSessionLedger(prompts, [], pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(result.promptsMigrated).toBe(2);
      expect(pmcStore.promptWrites).toHaveLength(2);
      expect(pmcStore.promptWrites[0].sessionId).toBe('ses_abc');
      expect(pmcStore.promptWrites[0].rawPrompt).toBe('Fix the bug');
    });

    it('migrates session_summary observations to PMC session_summaries', async () => {
      const sessionSummaryObs = [
        makeObservation({
          id: 100,
          type: 'session_summary',
          title: 'Session summary: test-project',
          content: '## Goal\nFixed auth bug\n## Accomplished\n- JWT fix',
          session_id: 'ses_xyz',
          project: 'test-project',
        }),
      ];

      const result = await migrateSessionLedger([], sessionSummaryObs, pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: false });

      expect(result.summariesMigrated).toBe(1);
      expect(pmcStore.summaryWrites).toHaveLength(1);
      expect(pmcStore.summaryWrites[0].sessionId).toBe('ses_xyz');
      expect(pmcStore.summaryWrites[0].summary).toContain('Fixed auth bug');
    });

    it('dry-run reports counts without writing', async () => {
      const prompts = [makeUserPrompt()];

      const result = await migrateSessionLedger(prompts, [], pmcStore as unknown as MemoryStore, embedder as Embedder, { dryRun: true });

      expect(result.promptsMigrated).toBe(1);
      expect(pmcStore.promptWrites).toHaveLength(0);
    });
  });

  // ── Relation migration ─────────────────────────────────────────

  describe('migrateRelations', () => {
    it('migrates Engram memory_relations to PMC memory_relations', async () => {
      const relations = [
        makeRelation({ sync_id: 'rel-001', source_id: 'obs-001', target_id: 'obs-002', relation: 'not_conflict' }),
        makeRelation({ sync_id: 'rel-002', source_id: 'obs-003', target_id: 'obs-004', relation: 'related', reason: 'Same topic', confidence: 0.9 }),
      ];

      const result = await migrateRelations(relations, pmcStore as unknown as MemoryStore, { dryRun: false });

      expect(result.migrated).toBe(2);
      expect(pmcStore.relationWrites).toHaveLength(2);
      expect(pmcStore.relationWrites[0].relation).toBe('not_conflict');
      expect(pmcStore.relationWrites[1].confidence).toBe(0.9);
    });

    it('dry-run reports counts without writing', async () => {
      const relations = [makeRelation()];

      const result = await migrateRelations(relations, pmcStore as unknown as MemoryStore, { dryRun: true });

      expect(result.migrated).toBe(1);
      expect(pmcStore.relationWrites).toHaveLength(0);
    });
  });

  // ── Symbol linking ─────────────────────────────────────────────

  describe('linkSymbols', () => {
    it('matches content paths against graph.json nodes and calls attach_symbol', async () => {
      const graph = {
        nodes: [
          { id: 'n1', path: 'src/auth/middleware.ts', name: 'middleware' },
          { id: 'n2', path: 'src/config/config.ts', name: 'config' },
        ],
      };

      const memories = [
        { id: 'pmc-mem-0', content: 'Fixed JWT validation in src/auth/middleware.ts' },
        { id: 'pmc-mem-1', content: 'Updated config in src/config/config.ts' },
        { id: 'pmc-mem-2', content: 'No path reference here' },
      ];

      const result = await linkSymbols(memories, graph, pmcStore as unknown as MemoryStore);

      expect(result.linked).toBe(2);
      expect(pmcStore.symbolLinks).toHaveLength(2);
      expect(pmcStore.symbolLinks[0].memoryId).toBe('pmc-mem-0');
      expect(pmcStore.symbolLinks[0].symbolKey).toContain('middleware');
    });

    it('symbol link failure is logged but does NOT abort', async () => {
      const graph = { nodes: [{ id: 'n1', path: 'src/foo.ts', name: 'foo' }] };

      const memories = [
        { id: 'pmc-mem-0', content: 'Reference to src/foo.ts' },
        { id: 'pmc-mem-1', content: 'Reference to src/foo.ts' },
      ];

      // Make attachSymbol fail for the first call
      let callCount = 0;
      const failingStore = {
        ...pmcStore,
        async attachSymbol(memoryId: string, symbolKey: string) {
          callCount++;
          if (callCount === 1) throw new Error('DB locked');
          pmcStore.symbolLinks.push({ memoryId, symbolKey });
        },
      };

      const result = await linkSymbols(memories, graph, failingStore as unknown as MemoryStore);

      // Second one should still succeed
      expect(result.linked).toBe(1);
      expect(result.errors).toBe(1);
    });
  });

  // ── Full runMigration ──────────────────────────────────────────

  describe('runMigration', () => {
    it('returns progress table with table, migrated, skipped, errors columns', async () => {
      const engramDb = {
        observations: [makeObservation({ id: 1 }), makeObservation({ id: 2 })],
        userPrompts: [makeUserPrompt()],
        relations: [makeRelation()],
      };

      const result = await runMigration(engramDb, pmcStore as unknown as MemoryStore, embedder as Embedder, {
        dryRun: true,
        graphNodes: [],
      });

      expect(result.table).toBeDefined();
      expect(result.table.length).toBeGreaterThan(0);
      expect(result.table[0]).toHaveProperty('table');
      expect(result.table[0]).toHaveProperty('migrated');
      expect(result.table[0]).toHaveProperty('skipped');
      expect(result.table[0]).toHaveProperty('errors');
    });

    it('exits with success when all tables migrate cleanly', async () => {
      const engramDb = {
        observations: [makeObservation()],
        userPrompts: [makeUserPrompt()],
        relations: [],
      };

      const result = await runMigration(engramDb, pmcStore as unknown as MemoryStore, embedder as Embedder, {
        dryRun: false,
        graphNodes: [],
      });

      expect(result.success).toBe(true);
    });
  });
});
