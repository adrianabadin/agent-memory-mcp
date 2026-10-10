// tests/embedding-model-guard.test.ts
//
// M1 (E5 small migration): embedding identity guard + query/passage callsites.
//
// Open-time contract (SqliteMemoryStore.initialize):
//   - stored identity present  → dimensions ALWAYS validated; model/pooling/
//     chunk-policy-version validated when the embedder exposes an identity.
//   - no stored identity + fresh DB (no schema) → stamp embedder identity.
//   - no stored identity + legacy schema → known-compat for 1024-dim legacy
//     embedders; a 384-dim role-aware embedder (E5) must NOT silently pretend
//     to be the legacy model.
//   - any mismatch throws BEFORE schema mutations/migrations.
//   - identityless custom embedders (mocks) keep compat, dims still checked.
//
// Also asserts role usage at callsites: stored content → passage, semantic
// query → query (search), session ledger writes → passage.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  SqliteMemoryStore,
  readEmbeddingIdentity,
  writeEmbeddingIdentity,
  EmbeddingIdentityMismatchError,
} from '../src/sqlite-store.js';
import type { Embedder, EmbeddingIdentity } from '../src/types.js';
import { createLedgerOnlyStore } from '../src/compose.js';

const E5_IDENTITY: EmbeddingIdentity = {
  model: 'Xenova/multilingual-e5-small',
  dimensions: 384,
  pooling: 'mean',
  normalize: true,
  roleAware: true,
  chunkPolicy: { version: 'e5-fulltext-v1', maxTokens: 512, overlapTokens: 64 },
};

const BGE_IDENTITY: EmbeddingIdentity = {
  model: 'Xenova/bge-m3',
  dimensions: 1024,
  pooling: 'cls',
  normalize: true,
  roleAware: false,
  chunkPolicy: null,
};

interface EmbedCall {
  text: string;
  role?: string;
}

/** Lightweight fake embedder — no model, no network. */
function fakeEmbedder(opts: {
  dims: number;
  identity?: EmbeddingIdentity;
  log?: EmbedCall[];
}): Embedder {
  return {
    initialize: async () => {},
    embed: async (text: string, role?: 'query' | 'passage') => {
      opts.log?.push({ text, role });
      return new Array<number>(opts.dims).fill(0.01);
    },
    embedBatch: async (texts: string[]) => texts.map(() => new Array<number>(opts.dims).fill(0.01)),
    dimensions: () => opts.dims,
    ...(opts.identity ? { identity: () => opts.identity } : {}),
  };
}

describe('embedding identity guard (SqliteMemoryStore open-time)', () => {
  let dir: string;
  const dbs: string[] = [];
  const freshDb = () => {
    const p = join(dir, `mem-${dbs.length}.db`);
    dbs.push(p);
    return p;
  };
  /** Pre-existing schema, NO embedding metadata → legacy database. */
  const legacyDb = () => {
    const p = freshDb();
    const db = new DatabaseSync(p);
    db.exec('CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT);');
    db.close();
    return p;
  };
  /** Legacy-style DB carrying an explicit identity stamp. */
  const stampedDb = (identity: EmbeddingIdentity) => {
    const p = legacyDb();
    const db = new DatabaseSync(p);
    writeEmbeddingIdentity(db, identity);
    db.close();
    return p;
  };

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'pmc-guard-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('opens a fresh DB with an E5 embedder and stamps its identity', async () => {
    const store = new SqliteMemoryStore(freshDb(), fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await store.initialize();
    const db = store.getDb();
    const stamped = readEmbeddingIdentity(db);
    expect(stamped).not.toBeNull();
    expect(stamped!.model).toBe(E5_IDENTITY.model);
    expect(stamped!.dimensions).toBe(384);
    expect(stamped!.chunkPolicyVersion).toBe('e5-fulltext-v1');
    expect(stamped!.chunkPolicy).toEqual(E5_IDENTITY.chunkPolicy);
    // Schema + migrations ran.
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBeGreaterThanOrEqual(1);
    store.close();
  });

  it('opens an unlabelled legacy DB with the 1024-dim BGE embedder (known compat), without stamping', async () => {
    const store = new SqliteMemoryStore(legacyDb(), fakeEmbedder({ dims: 1024, identity: BGE_IDENTITY }));
    await store.initialize();
    // Conservative: opening a legacy DB must not silently claim a model.
    expect(readEmbeddingIdentity(store.getDb())).toBeNull();
    store.close();
  });

  it('REJECTS an unlabelled legacy DB with a 384-dim E5 embedder BEFORE any schema mutation', async () => {
    const path = legacyDb();
    const store = new SqliteMemoryStore(path, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();

    // Fail BEFORE source DB mutations: no migrations, no new tables.
    const db = new DatabaseSync(path);
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBe(0);
    const tables = (db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table'`,
    ).all() as unknown as Array<{ name: string }>).map(r => r.name);
    expect(tables).toEqual(['memories']); // only the pre-existing fixture table
    expect(readEmbeddingIdentity(db)).toBeNull();
    db.close();
  });

  it('rejects opening an E5-stamped DB with the BGE embedder (model + dims mismatch)', async () => {
    const store = new SqliteMemoryStore(stampedDb(E5_IDENTITY), fakeEmbedder({ dims: 1024, identity: BGE_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('opens an E5-stamped DB with the matching E5 embedder', async () => {
    const store = new SqliteMemoryStore(stampedDb(E5_IDENTITY), fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await store.initialize();
    store.close();
  });

  it('rejects a chunk policy version drift (policy is part of identity)', async () => {
    const drifted: EmbeddingIdentity = {
      ...E5_IDENTITY,
      chunkPolicy: { version: 'e5-fulltext-v2', maxTokens: 512, overlapTokens: 64 },
    };
    const store = new SqliteMemoryStore(stampedDb(E5_IDENTITY), fakeEmbedder({ dims: 384, identity: drifted }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('keeps identityless custom embedders compatible when dimensions match the stamp', async () => {
    const mock = fakeEmbedder({ dims: 384 }); // no identity() — custom/test embedder
    const store = new SqliteMemoryStore(stampedDb(E5_IDENTITY), mock);
    await store.initialize();
    store.close();
  });

  it('still validates dimensions for identityless custom embedders (mismatch rejected)', async () => {
    const mock = fakeEmbedder({ dims: 384 }); // no identity()
    const store = new SqliteMemoryStore(stampedDb(BGE_IDENTITY), mock); // stamp says 1024
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('allows identityless custom embedders on unlabelled legacy DBs (test-compat)', async () => {
    const store = new SqliteMemoryStore(legacyDb(), fakeEmbedder({ dims: 384 }));
    await store.initialize();
    store.close();
  });

  it('readEmbeddingIdentity/writeEmbeddingIdentity round-trip the full identity', () => {
    const db = new DatabaseSync(freshDb());
    expect(readEmbeddingIdentity(db)).toBeNull(); // no table yet
    writeEmbeddingIdentity(db, E5_IDENTITY);
    const stored = readEmbeddingIdentity(db);
    expect(stored!.model).toBe(E5_IDENTITY.model);
    expect(stored!.dimensions).toBe(384);
    expect(stored!.pooling).toBe('mean');
    expect(stored!.normalize).toBe(true);
    expect(stored!.roleAware).toBe(true);
    expect(stored!.chunkPolicyVersion).toBe('e5-fulltext-v1');
    expect(stored!.chunkPolicy).toEqual(E5_IDENTITY.chunkPolicy);
    expect(typeof stored!.stampedAt).toBe('string');
    // Re-stamp overwrites (id=1 upsert).
    writeEmbeddingIdentity(db, BGE_IDENTITY);
    expect(readEmbeddingIdentity(db)!.model).toBe(BGE_IDENTITY.model);
    db.close();
  });
});

describe('query vs passage callsites (SqliteMemoryStore)', () => {
  let dir: string;
  let dbPath: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'pmc-roles-'));
    dbPath = join(dir, 'mem.db');
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('uses passage for stored content and query for semantic search', async () => {
    const log: EmbedCall[] = [];
    const store = new SqliteMemoryStore(dbPath, fakeEmbedder({ dims: 384, log }));
    await store.initialize();

    const stored = await store.store({ content: 'hello stored world', category: 'fact', tags: [] });
    expect('id' in stored).toBe(true);
    expect(log.at(-1)).toEqual({ text: 'hello stored world', role: 'passage' });

    await store.search('needle query', 'semantic', {});
    expect(log.at(-1)).toEqual({ text: 'needle query', role: 'query' });

    const id = (stored as { id: string }).id;
    await store.update(id, { content: 'hello updated world' });
    expect(log.at(-1)).toEqual({ text: 'hello updated world', role: 'passage' });

    // Session ledger writes are stored content → passage.
    await store.storeSessionPrompt('session-1', 'raw prompt text');
    expect(log.at(-1)).toEqual({ text: 'raw prompt text', role: 'passage' });
    await store.storeSessionResponse('session-1', (await store.getSession('session-1')).prompts[0].id, 'assistant reply');
    expect(log.at(-1)?.role).toBe('passage');
    await store.storeSessionToolCall({ sessionId: 'session-1', toolName: 'bash', resultSummary: 'command ran' });
    expect(log.at(-1)).toEqual({ text: 'command ran', role: 'passage' });
    await store.storeSessionSummary({ sessionId: 'session-1', promptId: (await store.getSession('session-1')).prompts[0].id, summary: 'session summary' });
    expect(log.at(-1)?.role).toBe('passage');

    store.close();
  });
});

// ── Finding 2: open-time vector histogram audit ──────────────────
// A (possibly fake) stamp must NEVER override the actual stored vector
// lengths: ALL existing non-null embedding columns are audited by cheap
// SQL typeof/length counts BEFORE schema/migrations/writes.
describe('embedding histogram audit (all vector-bearing columns)', () => {
  let dir: string;
  const dbs: string[] = [];
  const fixtureDb = (opts: {
    stamp?: EmbeddingIdentity;
    /** byte length of the memory_embeddings blob (omit = no row) */
    memoryBlobBytes?: number;
    /** typeof scenario for the session_prompts.embedding cell */
    sessionCell?:
      | { kind: 'null' }
      | { kind: 'blob'; bytes: number }
      | { kind: 'text'; value: string }
      | { kind: 'integer'; value: number };
  }) => {
    const p = join(dir, `aud-${dbs.length}.db`);
    dbs.push(p);
    const db = new DatabaseSync(p);
    db.exec(`
      CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT);
      CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY REFERENCES memories(id), embedding BLOB NOT NULL);
      CREATE TABLE session_prompts (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        prompt_id TEXT,
        raw_prompt TEXT NOT NULL,
        created_at TEXT NOT NULL,
        embedding BLOB
      );
    `);
    if (opts.memoryBlobBytes !== undefined) {
      db.prepare(`INSERT INTO memories (id, content) VALUES ('m1', 'c')`).run();
      db.prepare(`INSERT INTO memory_embeddings (id, embedding) VALUES ('m1', ?)`)
        .run(Buffer.alloc(opts.memoryBlobBytes));
    }
    if (opts.sessionCell) {
      const cell = opts.sessionCell;
      const cols = `INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, embedding) VALUES ('s1', 'sess-x', 'raw', '2024-01-01T00:00:00Z', `;
      if (cell.kind === 'null') {
        db.prepare(`${cols} NULL)`).run();
      } else if (cell.kind === 'blob') {
        db.prepare(`${cols} ?)`).run(Buffer.alloc(cell.bytes));
      } else if (cell.kind === 'text') {
        db.prepare(`${cols} ?)`).run(cell.value);
      } else {
        db.prepare(`${cols} ?)`).run(cell.value);
      }
    }
    if (opts.stamp) writeEmbeddingIdentity(db, opts.stamp);
    db.close();
    return p;
  };

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'pmc-hist-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('REJECTS a fake E5-384 stamp over a 1024-dim corpus (stamp does not override reality)', async () => {
    const p = fixtureDb({ stamp: E5_IDENTITY, memoryBlobBytes: 1024 * 4 });
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('accepts a stamped DB whose stored vector lengths match', async () => {
    const p = fixtureDb({ stamp: E5_IDENTITY, memoryBlobBytes: 384 * 4 });
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await store.initialize();
    store.close();
  });

  it('REJECTS non-blob values in embedding columns (e.g. text/integer corruption)', async () => {
    const p = fixtureDb({ stamp: E5_IDENTITY, sessionCell: { kind: 'text', value: 'not-a-vector' } });
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('REJECTS a wrong-length blob in session columns too', async () => {
    const p = fixtureDb({ stamp: E5_IDENTITY, sessionCell: { kind: 'blob', bytes: 1024 * 4 } });
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    store.close();
  });

  it('NULL (pending) session embeddings are allowed in BOTH stamped and unlabelled DBs', async () => {
    const stamped = fixtureDb({ stamp: E5_IDENTITY, sessionCell: { kind: 'null' } });
    const s1 = new SqliteMemoryStore(stamped, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await s1.initialize();
    s1.close();
    const unlabelled = fixtureDb({ sessionCell: { kind: 'null' } });
    const s2 = new SqliteMemoryStore(unlabelled, fakeEmbedder({ dims: 1024, identity: BGE_IDENTITY }));
    await s2.initialize();
    s2.close();
  });

  it('UNLABELLED legacy DBs tolerate historical session placeholder junk (0-byte / 3-byte) as pending, but zero-byte memory vectors stay corrupt', async () => {
    const openLegacy = (p: string) =>
      new SqliteMemoryStore(p, fakeEmbedder({ dims: 1024, identity: BGE_IDENTITY }));

    const zeroP = fixtureDb({ sessionCell: { kind: 'blob', bytes: 0 } });
    const s1 = openLegacy(zeroP);
    await s1.initialize(); // 0-byte legacy placeholder = pending
    s1.close();

    const junkP = fixtureDb({ sessionCell: { kind: 'blob', bytes: 3 } });
    const s2 = openLegacy(junkP);
    await s2.initialize(); // non-float32 legacy junk = pending
    s2.close();

    const corruptP = fixtureDb({ memoryBlobBytes: 0 });
    const s3 = openLegacy(corruptP);
    await expect(s3.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError); // memory vectors are strict
    s3.close();
  });

  it('E5-stamped DBs REJECT any non-NULL invalid blob (0-byte / 3-byte) BEFORE writes; NULL stays allowed', async () => {
    for (const bytes of [0, 3]) {
      const p = fixtureDb({ stamp: E5_IDENTITY, sessionCell: { kind: 'blob', bytes } });
      const s = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
      await expect(s.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
      s.close();
      // Rejection happens BEFORE schema/migrations/writes.
      const db = new DatabaseSync(p);
      const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
      expect(version).toBe(0);
      db.close();
    }
    // NULL pending remains allowed on stamped DBs.
    const nullP = fixtureDb({ stamp: E5_IDENTITY, sessionCell: { kind: 'null' } });
    const s = new SqliteMemoryStore(nullP, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await s.initialize();
    s.close();
  });

  it('audits with the TARGET embedder dims for identityless custom embedders (no legacy 1024 assumption)', async () => {
    // Custom 3-dim embedder over a 3-dim corpus passes…
    const okP = fixtureDb({ memoryBlobBytes: 3 * 4, sessionCell: { kind: 'blob', bytes: 3 * 4 } });
    const s1 = new SqliteMemoryStore(okP, fakeEmbedder({ dims: 3 }));
    await s1.initialize();
    s1.close();
    // …and a mismatching corpus is rejected even without any stamp/identity.
    const badP = fixtureDb({ memoryBlobBytes: 1024 * 4 });
    const s2 = new SqliteMemoryStore(badP, fakeEmbedder({ dims: 3 }));
    await expect(s2.initialize()).rejects.toThrow(EmbeddingIdentityMismatchError);
    s2.close();
  });
});

// ── Finding 1: ledger-only store vs E5-stamped DBs ────────────────
// createLedgerOnlyStore must derive dims/identity from the DB stamp (or
// project config), open E5-stamped DBs, and write NULL pending vectors —
// never empty BLOBs. A later real embed writes proper 384-dim vectors.
describe('createLedgerOnlyStore (capture-drain path)', () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'pmc-drain-'));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  const stampedDbPath = () => {
    const p = join(dir, 'cap.db');
    const db = new DatabaseSync(p);
    db.exec('CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT);');
    writeEmbeddingIdentity(db, E5_IDENTITY);
    db.close();
    return join(dir, 'cap'); // createLedgerOnlyStore appends .db
  };

  it('opens an E5-stamped DB and writes raw ledger rows with NULL (pending) embeddings', async () => {
    const store = await createLedgerOnlyStore(stampedDbPath());
    const prompt = await store.storeSessionPrompt('sess-1', 'raw captured prompt');
    expect(prompt.rawPrompt).toBe('raw captured prompt');
    const db = new DatabaseSync(join(dir, 'cap.db'));
    const row = db.prepare(`SELECT raw_prompt, embedding FROM session_prompts WHERE id = ?`).get(prompt.id) as
      { raw_prompt: string; embedding: Buffer | null };
    expect(row.raw_prompt).toBe('raw captured prompt');
    expect(row.embedding).toBeNull(); // pending, NOT a 0-byte blob
    db.close();
    store.close();
  });

  it('a later REAL passage embed of the same row writes a proper 384-dim vector', async () => {
    // Real (mock 384) store over the same E5-stamped DB: new rows get vectors.
    const real = new SqliteMemoryStore(join(dir, 'cap.db'), fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await real.initialize();
    await real.storeSessionPrompt('sess-2', 'properly embedded prompt');
    const db = new DatabaseSync(join(dir, 'cap.db'));
    const row = db.prepare(
      `SELECT length(embedding) AS len FROM session_prompts WHERE session_id = 'sess-2'`,
    ).get() as { len: number };
    expect(row.len).toBe(384 * 4);
    db.close();
    real.close();
  });

  it('never writes empty BLOBs from the main store either: empty vectors become NULL', async () => {
    // Deferred/no-model mode: the embedder returns an empty vector.
    const emptyEmbedder: Embedder = {
      initialize: async () => {},
      embed: async () => [],
      embedBatch: async (texts) => texts.map(() => []),
      dimensions: () => 384,
      identity: () => E5_IDENTITY,
    };
    const dbPath = join(dir, 'empty.db');
    const store = new SqliteMemoryStore(dbPath, emptyEmbedder);
    await store.initialize();
    await store.storeSessionPrompt('sess-3', 'deferred prompt');
    const db = new DatabaseSync(dbPath);
    const row = db.prepare(`SELECT embedding FROM session_prompts WHERE session_id = 'sess-3'`).get() as {
      embedding: Buffer | null;
    };
    expect(row.embedding).toBeNull();
    db.close();
    store.close();
  });

  it('opens a legacy unlabelled DB (1024 blobs, no stamp, no env) via derived legacy dims', async () => {
    const p = join(dir, 'legacy.db');
    const db = new DatabaseSync(p);
    db.exec(`
      CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT);
      CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);
    `);
    db.prepare(`INSERT INTO memories (id, content) VALUES ('m1', 'c')`).run();
    db.prepare(`INSERT INTO memory_embeddings (id, embedding) VALUES ('m1', ?)`).run(Buffer.alloc(1024 * 4));
    db.close();
    const saved = process.env.EMBEDDING_DIMENSIONS;
    delete process.env.EMBEDDING_DIMENSIONS;
    try {
      const store = await createLedgerOnlyStore(join(dir, 'legacy'));
      store.close();
    } finally {
      if (saved !== undefined) process.env.EMBEDDING_DIMENSIONS = saved;
    }
  });
});

// ── M2 runtime migration-state guard ──────────────────────────────────
//
// A re-vectorization TARGET must be rejected by the runtime until its
// migration state row says COMPLETE — even when every vector already has the
// right shape/identity. The check runs in guardEmbeddingIdentity BEFORE any
// schema/migration write, so user_version and schema stay untouched on reject.
describe('migration state guard (M2 incomplete target)', () => {
  let dir: string;
  beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'pmc-migstate-')); });
  afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

  /** Target-shaped DB: full schema + valid 384-dim E5 vectors + stamp + state row. */
  function migrationTargetDb(status: 'INCOMPLETE' | 'ERROR' | 'UNKNOWN-STATUS' | 'MISSING-ROW'): string {
    const p = join(dir, `tgt-${status}.db`);
    const db = new DatabaseSync(p);
    db.exec(`
      CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL);
      CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);
    `);
    const v = Buffer.alloc(384 * 4);
    new Float32Array(v.buffer, v.byteOffset, 384).fill(1 / Math.sqrt(384));
    db.prepare(`INSERT INTO memories (id, content) VALUES ('m1', 'alpha')`).run();
    db.prepare(`INSERT INTO memory_embeddings (id, embedding) VALUES ('m1', ?)`).run(v);
    writeEmbeddingIdentity(db, E5_IDENTITY);
    db.exec(`CREATE TABLE embedding_migration_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      status TEXT NOT NULL,
      source_path TEXT NOT NULL,
      source_fingerprint TEXT NOT NULL,
      identity_json TEXT NOT NULL,
      started_at TEXT NOT NULL,
      completed_at TEXT
    )`);
    if (status !== 'MISSING-ROW') {
      db.prepare(`INSERT INTO embedding_migration_state
        (id, status, source_path, source_fingerprint, identity_json, started_at)
        VALUES (1, ?, 'src', 'fp', '{}', 'now')`).run(status);
    }
    db.close();
    return p;
  }

  it('rejects an INCOMPLETE target with a CORRECT E5 384 embedder, before user_version/schema mutate', async () => {
    const p = migrationTargetDb('INCOMPLETE');
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(/migration incomplete/i);
    store.close();
    // No schema/migration side effects on the rejected open.
    const db = new DatabaseSync(p, { readOnly: true });
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBe(0);
    const sessionTable = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='session_prompts'`,
    ).get();
    expect(sessionTable).toBeUndefined();
    db.close();
  });

  it('rejects ERROR status, unknown status, and a missing state row (table present)', async () => {
    for (const status of ['ERROR', 'UNKNOWN-STATUS', 'MISSING-ROW'] as const) {
      const p = migrationTargetDb(status);
      const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
      await expect(store.initialize()).rejects.toThrow(/migration incomplete/i);
      store.close();
    }
  });

  it('opens a COMPLETE-stamped target with the matching E5 embedder (positive)', async () => {
    const p = migrationTargetDb('ERROR'); // row exists; flip to COMPLETE below
    const db0 = new DatabaseSync(p);
    db0.prepare(`UPDATE embedding_migration_state SET status='COMPLETE', completed_at='now' WHERE id=1`).run();
    db0.close();
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await store.initialize();
    const db = store.getDb();
    const version = (db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version;
    expect(version).toBeGreaterThanOrEqual(1); // migrations ran normally
    store.close();
  });

  it('DBs without the migration state table are unaffected (legacy/fresh paths)', async () => {
    const p = join(dir, 'no-state.db');
    const db = new DatabaseSync(p);
    db.exec(`CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL);
             CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);`);
    db.close();
    const store = new SqliteMemoryStore(p, fakeEmbedder({ dims: 384, identity: E5_IDENTITY }));
    await expect(store.initialize()).rejects.toThrow(/unlabelled legacy/i); // M1 guard, NOT the state guard
    store.close();
  });
});
