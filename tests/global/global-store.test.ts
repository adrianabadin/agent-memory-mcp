import { describe, it, beforeEach, afterEach } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SqliteGlobalStore } from '../../src/global/global-store.js';
import { MockEmbedder } from '../mocks.js';

let tmpDir: string;
let store: SqliteGlobalStore;

beforeEach(async () => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pmc-global-store-test-'));
  store = new SqliteGlobalStore(join(tmpDir, 'global.db'), new MockEmbedder());
  await store.initialize();
});

afterEach(() => {
  store.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('SqliteGlobalStore', () => {
  it('registerProject creates a project record', () => {
    const rec = store.registerProject({ name: 'proj', rootPath: '/tmp/proj' });
    assert.equal(rec.name, 'proj');
    assert.equal(rec.rootPath, '/tmp/proj');
  });

  it('registerProject is idempotent', () => {
    store.registerProject({ name: 'proj', rootPath: '/tmp/proj' });
    store.registerProject({ name: 'proj', rootPath: '/tmp/proj', objective: 'Updated' });
    const rec = store.registerProject({ name: 'proj', rootPath: '/tmp/proj' });
    assert.equal(rec.objective, 'Updated');
  });

  it('syncMetadata updates project fields', async () => {
    const rec = store.registerProject({ name: 'proj', rootPath: '/tmp/proj' });
    store.syncMetadata(rec.id, { architecture: 'monolith', objective: 'Build X' });
    const err = await store.recordError({
      projectId: rec.id,
      message: 'test',
      fix: 'fix',
      source: 'auto',
    });
    assert.equal(err.projectId, rec.id);
  });

  it('recordError stores an error', async () => {
    const proj = store.registerProject({ name: 'p', rootPath: '/p' });
    const err = await store.recordError({
      projectId: proj.id,
      message: 'ReferenceError: x is not defined',
      rootCause: 'missing import',
      fix: 'add import statement at top of file',
      source: 'manual',
    });
    assert.equal(err.message, 'ReferenceError: x is not defined');
    assert.equal(err.source, 'manual');
  });

  it('searchErrors returns relevant results', async () => {
    const proj = store.registerProject({ name: 'p', rootPath: '/p' });
    await store.recordError({
      projectId: proj.id,
      message: 'ENOENT: file not found',
      rootCause: 'path is wrong',
      fix: 'use absolute path instead of relative',
      source: 'auto',
    });
    const results = await store.searchErrors('file not found path');
    assert.ok(results.length >= 1);
  });

  it('searchErrors returns empty array when store is empty', async () => {
    const results = await store.searchErrors('anything');
    assert.deepEqual(results, []);
  });

  it('close() does not throw', () => {
    assert.doesNotThrow(() => store.close());
  });
});
