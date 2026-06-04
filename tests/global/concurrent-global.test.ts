import { describe, it, beforeEach, afterEach } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SqliteGlobalStore } from '../../src/global/global-store.js';
import { MockEmbedder } from '../mocks.js';

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pmc-concurrent-global-'));
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('concurrent global store writes', () => {
  it('two concurrent stores writing errors produce correct count', async () => {
    const dbPath = join(tmpDir, 'global.db');
    const embedder = new MockEmbedder();

    // Open two independent stores on the same file (simulates two processes)
    const storeA = new SqliteGlobalStore(dbPath, embedder);
    const storeB = new SqliteGlobalStore(dbPath, embedder);
    await storeA.initialize();
    await storeB.initialize();

    // Register the same project from both (idempotent)
    storeA.registerProject({ name: 'proj', rootPath: '/proj' });
    storeB.registerProject({ name: 'proj', rootPath: '/proj' });
    const { id: projectId } = storeA.registerProject({ name: 'proj', rootPath: '/proj' });

    // Fire 10 concurrent writes from each store
    const writes = [
      ...Array.from({ length: 10 }, (_, i) =>
        storeA.recordError({ projectId, message: `Error A${i}`, fix: `Fix A${i}`, source: 'auto' })
      ),
      ...Array.from({ length: 10 }, (_, i) =>
        storeB.recordError({ projectId, message: `Error B${i}`, fix: `Fix B${i}`, source: 'auto' })
      ),
    ];

    await Promise.all(writes);

    // All 20 distinct errors should be in the DB
    const results = await storeA.searchErrors('Error', undefined, 50);
    assert.ok(results.length >= 20, `Expected >= 20 results, got ${results.length}`);

    storeA.close();
    storeB.close();
  });
});
