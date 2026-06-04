import { describe, it, beforeEach, afterEach } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  handleRegisterProject,
  handleSyncProjectMetadata,
  handleRecordError,
  handleSearchGlobalErrors,
} from '../../src/tools.js';
import { SqliteGlobalStore } from '../../src/global/global-store.js';
import { MockEmbedder } from '../mocks.js';

let tmpDir: string;
let globalStore: SqliteGlobalStore;

beforeEach(async () => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pmc-tools-global-test-'));
  globalStore = new SqliteGlobalStore(join(tmpDir, 'global.db'), new MockEmbedder());
  await globalStore.initialize();
});

afterEach(() => {
  globalStore.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('handleRegisterProject', () => {
  it('returns success with project record', async () => {
    const handler = handleRegisterProject(globalStore);
    const result = await handler({ name: 'my-repo', rootPath: '/home/user/my-repo' });
    assert.ok(!result.isError);
    const data = JSON.parse(result.content[0].text);
    assert.equal(data.name, 'my-repo');
    assert.ok(data.id);
  });

  it('is idempotent — second call returns same project', async () => {
    const handler = handleRegisterProject(globalStore);
    await handler({ name: 'repo', rootPath: '/repo' });
    const result = await handler({ name: 'repo', rootPath: '/repo', objective: 'Do stuff' });
    const data = JSON.parse(result.content[0].text);
    assert.equal(data.objective, 'Do stuff');
  });
});

describe('handleSyncProjectMetadata', () => {
  it('updates project metadata', async () => {
    const proj = globalStore.registerProject({ name: 'p', rootPath: '/p' });
    const handler = handleSyncProjectMetadata(globalStore);
    const result = await handler({ projectId: proj.id, architecture: 'monolith' });
    assert.ok(!result.isError);
    const data = JSON.parse(result.content[0].text);
    assert.equal(data.architecture, 'monolith');
  });
});

describe('handleRecordError', () => {
  it('stores an error and returns it', async () => {
    const proj = globalStore.registerProject({ name: 'p', rootPath: '/p' });
    const handler = handleRecordError(globalStore);
    const result = await handler({
      projectId: proj.id,
      message: 'TypeError: x is null',
      rootCause: 'null check missing',
      fix: 'add if (x) before access',
      source: 'manual',
    });
    assert.ok(!result.isError);
    const data = JSON.parse(result.content[0].text);
    assert.equal(data.message, 'TypeError: x is null');
  });
});

describe('handleSearchGlobalErrors', () => {
  it('returns empty results for empty store', async () => {
    const handler = handleSearchGlobalErrors(globalStore);
    const result = await handler({ query: 'TypeError' });
    assert.ok(!result.isError);
    const data = JSON.parse(result.content[0].text);
    assert.equal(data.count, 0);
  });

  it('returns results matching the query', async () => {
    const proj = globalStore.registerProject({ name: 'p', rootPath: '/p' });
    await globalStore.recordError({
      projectId: proj.id,
      message: 'ENOENT: file.ts not found',
      fix: 'check path is correct',
      source: 'auto',
    });
    const handler = handleSearchGlobalErrors(globalStore);
    const result = await handler({ query: 'ENOENT file not found' });
    const data = JSON.parse(result.content[0].text);
    assert.ok(data.count >= 1);
  });
});
