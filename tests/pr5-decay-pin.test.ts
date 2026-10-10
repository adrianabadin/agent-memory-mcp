import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { DatabaseSync } from 'node:sqlite';

import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { MockEmbedder } from './mocks.js';
import { DEFAULT_FORGETTING_CONFIG } from '../src/decay.js';

// ── T-004: decay.ts skip-if-pinned wiring ───────────────────────────

describe('T-004: decay skip-if-pinned', () => {
  let store: SqliteMemoryStore;
  let dbPath: string;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), 'amm-pr5-decaypin-'));
    dbPath = join(tmpDir, 'test.db');
    store = new SqliteMemoryStore(dbPath, new MockEmbedder());
    await store.initialize();
  });

  afterEach(async () => {
    store.close();
    await rm(tmpDir, { recursive: true, force: true });
  });

  it('pinned memory does NOT advance decay state even when old enough to transition', async () => {
    // Create a memory with a very old last_reinforced_at that SHOULD transition to cooling
    const oldDate = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(); // 60 days ago
    await store.store({
      id: 'pinned-old',
      content: 'pinned old memory',
      category: 'learning',
      tags: [],
    });
    // Manually set last_reinforced_at to 60 days ago
    const db = new DatabaseSync(dbPath);
    db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(oldDate, 'pinned-old');
    db.close();

    // Pin the memory
    await store.pin('pinned-old');

    // Run batch decay evaluation
    const result = store.evaluateAllMemoriesForDecay(DEFAULT_FORGETTING_CONFIG);

    // The pinned memory should NOT have transitioned (skipped by decay)
    const pinnedResult = result.results.find(r => r.memoryId === 'pinned-old');
    if (pinnedResult) {
      // If it's in the batch result, it should NOT have transitioned
      expect(pinnedResult.result.transitioned).toBe(false);
      expect(pinnedResult.result.newState).toBe('active');
    }
    // If it was skipped entirely (not in results), that's also correct behavior
  });

  it('unpinned memory DOES advance decay state when old enough to transition', async () => {
    // Create an unpinned memory with old last_reinforced_at
    const oldDate = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(); // 60 days ago
    await store.store({
      id: 'unpinned-old',
      content: 'unpinned old memory',
      category: 'learning',
      tags: [],
    });
    const db = new DatabaseSync(dbPath);
    db.prepare('UPDATE memories SET last_reinforced_at = ? WHERE id = ?').run(oldDate, 'unpinned-old');
    db.close();

    // Run batch decay evaluation
    const result = store.evaluateAllMemoriesForDecay(DEFAULT_FORGETTING_CONFIG);

    // The unpinned memory SHOULD have transitioned to 'cooling' (60 days > 30 day threshold)
    const unpinnedResult = result.results.find(r => r.memoryId === 'unpinned-old');
    expect(unpinnedResult).toBeDefined();
    expect(unpinnedResult!.result.transitioned).toBe(true);
    expect(unpinnedResult!.result.newState).toBe('cooling');
  });

  it('is_pinned=0 memory (default) still goes through decay evaluation', async () => {
    await store.store({
      id: 'default-unpinned',
      content: 'default memory decay test',
      category: 'learning',
      tags: [],
    });

    // Verify it's not pinned
    const db = new DatabaseSync(dbPath);
    const row = db.prepare('SELECT is_pinned FROM memories WHERE id = ?').get('default-unpinned') as { is_pinned: number };
    expect(row.is_pinned).toBe(0);
    db.close();

    // Run batch decay — should be included in results
    const result = store.evaluateAllMemoriesForDecay(DEFAULT_FORGETTING_CONFIG);
    const memResult = result.results.find(r => r.memoryId === 'default-unpinned');
    expect(memResult).toBeDefined();
  });
});
