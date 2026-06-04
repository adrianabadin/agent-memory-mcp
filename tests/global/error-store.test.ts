import { describe, it, beforeEach, afterEach } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { ErrorStore } from '../../src/global/error-store.js';
import { MockEmbedder } from '../mocks.js';

let tmpDir: string;
let db: DatabaseSync;
let store: ErrorStore;
const PROJECT_ID = 'test-project-id';

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'pmc-error-store-test-'));
  db = new DatabaseSync(join(tmpDir, 'global.db'));
  db.exec('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;');
  // errors table has a FK to projects — insert a stub project row first
  db.exec(`CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, root_path TEXT NOT NULL,
    objective TEXT, stack TEXT, architecture TEXT, dependencies TEXT, minimap TEXT,
    created_at TEXT NOT NULL, last_synced_at TEXT NOT NULL
  )`);
  db.prepare(`INSERT INTO projects VALUES (?,?,?,NULL,NULL,NULL,NULL,NULL,?,?)`)
    .run(PROJECT_ID, 'test', '/test', new Date().toISOString(), new Date().toISOString());
  store = new ErrorStore(db, new MockEmbedder());
  store.ensureSchema();
});

afterEach(() => {
  db.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe('ErrorStore', () => {
  it('computeContentHash is deterministic', () => {
    const h1 = ErrorStore.computeContentHash('TypeError', 'null dereference');
    const h2 = ErrorStore.computeContentHash('TypeError', 'null dereference');
    assert.equal(h1, h2);
    assert.match(h1, /^[a-f0-9]{64}$/);
  });

  it('computeContentHash differs for different inputs', () => {
    assert.notEqual(
      ErrorStore.computeContentHash('TypeError', 'cause A'),
      ErrorStore.computeContentHash('TypeError', 'cause B'),
    );
  });

  it('record inserts an error and returns it', async () => {
    const rec = await store.record({
      projectId: PROJECT_ID,
      message: 'TypeError: cannot read x',
      rootCause: 'null dereference',
      fix: 'add null check before accessing x',
      source: 'manual',
    });
    assert.equal(rec.message, 'TypeError: cannot read x');
    assert.equal(rec.fix, 'add null check before accessing x');
    assert.equal(rec.source, 'manual');
    assert.ok(rec.id);
    assert.ok(rec.contentHash);
  });

  it('record is idempotent by content_hash — upserts on conflict', async () => {
    await store.record({ projectId: PROJECT_ID, message: 'E1', rootCause: 'C1', fix: 'F1', source: 'auto' });
    await store.record({ projectId: PROJECT_ID, message: 'E1', rootCause: 'C1', fix: 'F1 updated', source: 'auto' });
    const all = store.list();
    assert.equal(all.length, 1);
    assert.equal(all[0].fix, 'F1 updated');
  });

  it('different errors in different projects are separate rows', async () => {
    db.prepare(`INSERT INTO projects VALUES (?,?,?,NULL,NULL,NULL,NULL,NULL,?,?)`)
      .run('project-2', 'test2', '/test2', new Date().toISOString(), new Date().toISOString());
    await store.record({ projectId: PROJECT_ID, message: 'Same error', rootCause: 'same', fix: 'fix A', source: 'auto' });
    await store.record({ projectId: 'project-2', message: 'Same error', rootCause: 'same', fix: 'fix B', source: 'auto' });
    assert.equal(store.list().length, 2);
  });

  it('search returns relevant errors by FTS keyword', async () => {
    await store.record({ projectId: PROJECT_ID, message: 'ECONNREFUSED database', rootCause: 'DB not running', fix: 'start the DB service', source: 'auto' });
    await store.record({ projectId: PROJECT_ID, message: 'SyntaxError in parser', rootCause: 'missing semicolon', fix: 'add semicolon', source: 'auto' });
    const results = await store.search('database connection refused');
    assert.ok(results.length >= 1);
    assert.ok(results.some(r => r.error.message.includes('ECONNREFUSED')));
  });

  it('search returns empty array on empty store', async () => {
    const results = await store.search('anything');
    assert.deepEqual(results, []);
  });

  it('search with projectId filter only returns errors from that project', async () => {
    db.prepare(`INSERT INTO projects VALUES (?,?,?,NULL,NULL,NULL,NULL,NULL,?,?)`)
      .run('other', 'other', '/other', new Date().toISOString(), new Date().toISOString());
    await store.record({ projectId: PROJECT_ID, message: 'error in project A', fix: 'fix A', source: 'auto' });
    await store.record({ projectId: 'other', message: 'error in project B', fix: 'fix B', source: 'auto' });
    const results = await store.search('error', PROJECT_ID);
    assert.ok(results.every(r => r.error.projectId === PROJECT_ID));
  });

  it('list() with no filter returns all errors', async () => {
    await store.record({ projectId: PROJECT_ID, message: 'e1', fix: 'f1', source: 'auto' });
    await store.record({ projectId: PROJECT_ID, message: 'e2', fix: 'f2', source: 'auto' });
    assert.equal(store.list().length, 2);
  });
});
