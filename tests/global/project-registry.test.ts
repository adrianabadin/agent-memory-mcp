import { describe, it, beforeEach, afterEach } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { ProjectRegistry } from '../../src/global/project-registry.js';

let tmpDir: string;
let db: DatabaseSync;
let registry: ProjectRegistry;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pmc-project-registry-test-'));
  db = new DatabaseSync(join(tmpDir, 'global.db'));
  db.exec('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;');
  registry = new ProjectRegistry(db);
  registry.ensureSchema();
});

afterEach(() => {
  db.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('ProjectRegistry', () => {
  it('computeId returns stable slug for same path', () => {
    const id1 = ProjectRegistry.computeId('/Users/me/projects/foo');
    const id2 = ProjectRegistry.computeId('/Users/me/projects/foo');
    assert.equal(id1, id2);
    assert.match(id1, /^[a-f0-9]{64}$/);
  });

  it('computeId differs for different paths', () => {
    assert.notEqual(
      ProjectRegistry.computeId('/path/a'),
      ProjectRegistry.computeId('/path/b'),
    );
  });

  it('register creates a project record', () => {
    const rec = registry.register({
      name: 'my-project',
      rootPath: '/Users/me/my-project',
      objective: 'Build something great',
    });
    assert.equal(rec.name, 'my-project');
    assert.equal(rec.rootPath, '/Users/me/my-project');
    assert.equal(rec.objective, 'Build something great');
    assert.ok(rec.id.length === 64);
    assert.ok(rec.createdAt);
  });

  it('register is idempotent — second call updates lastSyncedAt, preserves createdAt', async () => {
    registry.register({ name: 'proj', rootPath: '/p' });
    await new Promise(r => setTimeout(r, 10));
    const second = registry.register({ name: 'proj', rootPath: '/p', objective: 'Updated' });
    const all = registry.list();
    assert.equal(all.length, 1);
    assert.equal(second.objective, 'Updated');
  });

  it('updateMetadata upserts fields without touching others', () => {
    const rec = registry.register({ name: 'p', rootPath: '/p', objective: 'Original' });
    registry.updateMetadata(rec.id, { architecture: 'microservices' });
    const got = registry.get(rec.id);
    assert.equal(got?.objective, 'Original');
    assert.equal(got?.architecture, 'microservices');
  });

  it('get returns null for unknown id', () => {
    assert.equal(registry.get('nonexistent'), null);
  });

  it('list returns all registered projects', () => {
    registry.register({ name: 'a', rootPath: '/a' });
    registry.register({ name: 'b', rootPath: '/b' });
    assert.equal(registry.list().length, 2);
  });
});
