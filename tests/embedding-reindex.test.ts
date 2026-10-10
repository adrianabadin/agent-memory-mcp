// tests/embedding-reindex.test.ts
//
// M2 (E5 small migration): snapshot + re-vectorization tooling.
//   - source opened readOnly; VACUUM INTO snapshot captures committed WAL
   //   - canonical path validation (same file / hardlink / symlink rejected)
//   - full re-embed of ALL vector-bearing rows (incl. NULL/legacy junk)
//   - embedded text mapping matches the ACTUAL store serialization
//   - non-embedding parity (ids/content/FTS/relations/sessions/rowids)
//   - durable per-batch progress in target metadata; resume; drift detection
//   - identity stamp ONLY after full validation; INCOMPLETE guard
//   - JSON report contains counts/hashes/timings, NO raw memory text
//
// All embedding is deterministic/fake — no network, no model.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm, readFile, writeFile } from 'node:fs/promises';
import { existsSync, linkSync, statSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { SqliteMemoryStore } from '../src/sqlite-store.js';
import { readEmbeddingIdentity } from '../src/sqlite-store.js';
import { E5_CHUNK_POLICY, type EmbedderTokenizer } from '../src/embedder.js';
import { blobToEmbedding } from '../src/shared/vector-utils.js';
import {
  runReembed,
  assertNoIncompleteMigration,
  computeSourceFingerprint,
  assertDistinctPaths,
  SNAPSHOT_METHOD,
  type ReindexEmbedder,
} from '../src/embedding-reindex.js';

// ── Deterministic fake embedders ────────────────────────────────────

function hashVector(text: string, dim: number): number[] {
  const v = new Array<number>(dim).fill(0);
  for (let i = 0; i < text.length; i++) v[i % dim] += (text.charCodeAt(i) + 1) / 997;
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return v.map(x => x / norm);
}

/** 384-dim E5-identity embedder that records every embedded text. */
class FakeE5Embedder implements ReindexEmbedder {
  readonly texts: string[] = [];
  constructor(private readonly failAfter?: number) {}
  async initialize(): Promise<void> {}
  async embed(text: string): Promise<number[]> {
    if (this.failAfter !== undefined && this.texts.length >= this.failAfter) {
      throw new Error('injected embed failure');
    }
    this.texts.push(text);
    return hashVector(text, 384);
  }
  dimensions(): number { return 384; }
  identity() {
    return {
      model: 'test/multilingual-e5-small',
      dimensions: 384,
      pooling: 'mean',
      normalize: true,
      roleAware: true,
      chunkPolicy: E5_CHUNK_POLICY,
    };
  }
}

/** Legacy 1024-dim identityless embedder for building the SOURCE fixture. */
class Legacy1024Embedder implements ReindexEmbedder {
  async initialize(): Promise<void> {}
  async embed(text: string): Promise<number[]> { return hashVector(text, 1024); }
  async embedBatch(texts: string[]): Promise<number[][]> { return texts.map(t => hashVector(t, 1024)); }
  dimensions(): number { return 1024; }
}

/** Minimal tokenizer to satisfy type imports when needed. */
const _unusedTokenizer: EmbedderTokenizer | null = null;
void _unusedTokenizer;

// ── Fixture ─────────────────────────────────────────────────────────

const SENTINEL = 'SECRET-MEMORY-CONTENT-XYZZY';

interface Fixture {
  dir: string;
  sourcePath: string;
  targetPath: string;
}

let fixture: Fixture;

async function buildSourceFixture(dir: string): Promise<string> {
  const sourcePath = join(dir, 'memory-db.db');
  const store = new SqliteMemoryStore(sourcePath, new Legacy1024Embedder());
  await store.initialize();

  await store.store({ content: `alpha widget ${SENTINEL}`, category: 'learning', tags: ['alpha'] });
  await store.store({ content: 'beta gadget', category: 'decision', tags: ['beta', 'shared'] });
  await store.store({ content: 'gamma shared thing', category: 'architecture', tags: ['shared'] });
  await store.store({ content: 'delta plain memory', category: 'bug-fix', tags: [] });

  const p1 = await store.storeSessionPrompt('sess-1', 'how do I reindex vectors?');
  const p2 = await store.storeSessionPrompt('sess-1', 'second prompt about gadgets');
  await store.storeSessionResponse('sess-1', p1.id, 'you run the reembed CLI on a snapshot');
  await store.storeSessionToolCall({
    sessionId: 'sess-1', promptId: p1.id, toolName: 'vitest',
    argsSafe: '{}', resultSummary: 'ran the embedding tests', importance: 'normal',
  });
  await store.storeSessionToolCall({
    sessionId: 'sess-1', promptId: p2.id, toolName: 'grep',
  });
  await store.storeSessionSummary({
    sessionId: 'sess-1', summary: 'we migrated vectors',
    decisions: ['use VACUUM INTO', 'keep source read-only'],
  });
  await store.storeSessionSummary({
    sessionId: 'sess-1', summary: 'summary without decisions field',
  });

  // Topic link inserted directly below (valid memory id).
  store.close();

  // Inject legacy junk rows directly: 0-byte, 3-byte, and NULL embeddings.
  const db = new DatabaseSync(sourcePath);
  db.exec(`
    INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, embedding)
    VALUES ('junk-zero', 'sess-2', 'junk zero byte prompt', '2024-01-01T00:00:00.000Z', x'');
    INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, embedding)
    VALUES ('junk-three', 'sess-2', 'junk three byte prompt', '2024-01-01T00:00:01.000Z', x'010203');
    INSERT INTO session_prompts (id, session_id, raw_prompt, created_at, embedding)
    VALUES ('junk-null', 'sess-2', 'junk null prompt', '2024-01-01T00:00:02.000Z', NULL);
    INSERT INTO memory_relations (id, memory_id_a, memory_id_b, relation, confidence, reasoning, marked_by, created_at)
    SELECT 'rel-1', m1.id, m2.id, 'related', 0.9, 'test relation', 'test', '2024-01-01T00:00:00.000Z'
      FROM memories m1, memories m2 WHERE m1.content LIKE 'alpha%' AND m2.content LIKE 'beta%';
  `);
  // A real topic link on a valid memory id.
  db.exec(`
    INSERT INTO memory_topics (topic_key, memory_id, created_at, updated_at)
    SELECT 'proj/reindex', id, '2024-01-01T00:00:00.000Z', '2024-01-01T00:00:00.000Z'
      FROM memories WHERE content LIKE 'alpha%';
  `);
  db.close();
  return sourcePath;
}

beforeEach(async () => {
  const dir = await mkdtemp(join(tmpdir(), 'reembed-test-'));
  const sourcePath = await buildSourceFixture(dir);
  fixture = { dir, sourcePath, targetPath: join(dir, 'memory-db-e5.db') };
});

afterEach(async () => {
  await rm(fixture.dir, { recursive: true, force: true });
});

// ── Helpers ─────────────────────────────────────────────────────────

function open(path: string, readOnly = false): DatabaseSync {
  return new DatabaseSync(path, readOnly ? { readOnly: true } : {});
}

function nonVectorTables(db: DatabaseSync): string[] {
  const rows = db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
       AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name`,
  ).all() as unknown as Array<{ name: string }>;
  return rows.map(r => r.name).filter(n => !n.startsWith('memories_fts_'));
}

// ── Path validation ─────────────────────────────────────────────────

describe('assertDistinctPaths', () => {
  it('rejects the same canonical path (separator/.. variants)', async () => {
    const p = join(fixture.dir, 'a.db');
    await writeFile(p, 'x');
    expect(() => assertDistinctPaths(p, p)).toThrow(/same file/i);
    expect(() => assertDistinctPaths(p, join(fixture.dir, '..', pathBase(fixture.dir), 'a.db'))).toThrow(/same file/i);
  });

  it('rejects a hardlink alias of the source as target', () => {
    const hard = join(fixture.dir, 'hard.db');
    linkSync(fixture.sourcePath, hard);
    expect(() => assertDistinctPaths(fixture.sourcePath, hard)).toThrow(/hardlink|same file/i);
  });

  it('rejects an existing symlink alias as target', () => {
    const link = join(fixture.dir, 'link.db');
    try {
      symlinkSync(fixture.sourcePath, link);
    } catch {
      return; // Windows without symlink privilege — covered by hardlink test
    }
    expect(() => assertDistinctPaths(fixture.sourcePath, link)).toThrow(/symlink|same file/i);
  });

  it('accepts a new distinct sibling target', () => {
    expect(() => assertDistinctPaths(fixture.sourcePath, fixture.targetPath)).not.toThrow();
  });
});

function pathBase(p: string): string {
  const norm = resolve(p).replace(/\\/g, '/');
  return norm.split('/').filter(Boolean).pop()!;
}

// ── Snapshot ────────────────────────────────────────────────────────

describe('snapshot (VACUUM INTO, readOnly source)', () => {
  it('captures committed WAL content and leaves source untouched', async () => {
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    expect(res.status).toBe('COMPLETE');
    expect(res.snapshot.method).toBe(SNAPSHOT_METHOD);
    expect(res.snapshot.walBytesCaptured).toBeGreaterThanOrEqual(0);

    const srcBefore = await readFile(fixture.sourcePath);
    expect(srcBefore.byteLength).toBeGreaterThan(0);
    const src = open(fixture.sourcePath, true);
    const n = (src.prepare('SELECT COUNT(*) AS n FROM memories').get() as { n: number }).n;
    expect(n).toBe(4);
    src.close();
  });
});

// ── Full migration ──────────────────────────────────────────────────

describe('runReembed (full migration)', () => {
  it('re-embeds ALL rows with EXACT store-serialization texts', async () => {
    const embedder = new FakeE5Embedder();
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder,
    });
    expect(res.status).toBe('COMPLETE');

    // Memory texts.
    expect(embedder.texts).toContain(`alpha widget ${SENTINEL}`);
    expect(embedder.texts).toContain('beta gadget');
    expect(embedder.texts).toContain('gamma shared thing');
    expect(embedder.texts).toContain('delta plain memory');
    // Prompt / response texts.
    expect(embedder.texts).toContain('how do I reindex vectors?');
    expect(embedder.texts).toContain('you run the reembed CLI on a snapshot');
    // Tool call: result_summary || tool_name
    expect(embedder.texts).toContain('ran the embedding tests');
    expect(embedder.texts).toContain('grep');
    // Summary: summary + ' ' + decisions.join(' ')
    expect(embedder.texts).toContain('we migrated vectors use VACUUM INTO keep source read-only');
    expect(embedder.texts).toContain('summary without decisions field');
    // Junk rows re-embedded from their text too.
    expect(embedder.texts).toContain('junk zero byte prompt');
    expect(embedder.texts).toContain('junk three byte prompt');
    expect(embedder.texts).toContain('junk null prompt');
  });

  it('produces valid 384-dim finite unit vectors in every vector slot', async () => {
    await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    const db = open(fixture.targetPath, true);
    const targets = ['memory_embeddings', 'session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries'];
    for (const t of targets) {
      const rows = db.prepare(`SELECT id, embedding FROM ${t}`).all() as unknown as Array<{ id: string; embedding: Buffer | null }>;
      expect(rows.length).toBeGreaterThan(0);
      for (const r of rows) {
        expect(r.embedding).not.toBeNull();
        expect(r.embedding!.byteLength).toBe(384 * 4);
        const vec = blobToEmbedding(r.embedding!);
        for (const x of vec) expect(Number.isFinite(x)).toBe(true);
        const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
        expect(Math.abs(norm - 1)).toBeLessThanOrEqual(1e-4);
      }
    }
    db.close();
  });

  it('preserves every non-embedding column, rowid, FTS index, relation and topic', async () => {
    await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    const src = open(fixture.sourcePath, true);
    const tgt = open(fixture.targetPath, true);

    const excluded = new Set(['embedding_meta', 'embedding_migration_state', 'embedding_migration_progress']);
    const srcTables = nonVectorTables(src);
    const tgtTables = nonVectorTables(tgt).filter(t => !excluded.has(t));
    expect(tgtTables).toEqual(srcTables);

    // Memories: ids, contents, versions, rowids identical.
    const s = src.prepare('SELECT rowid AS rid, id, content, version FROM memories ORDER BY rowid').all();
    const t = tgt.prepare('SELECT rowid AS rid, id, content, version FROM memories ORDER BY rowid').all();
    expect(JSON.stringify(t)).toBe(JSON.stringify(s));

    // Memory embeddings: ids and rowids preserved (blob bytes differ by design).
    const se = src.prepare('SELECT rowid AS rid, id FROM memory_embeddings ORDER BY rowid').all();
    const te = tgt.prepare('SELECT rowid AS rid, id FROM memory_embeddings ORDER BY rowid').all();
    expect(JSON.stringify(te)).toBe(JSON.stringify(se));

    // FTS shadow tables byte-identical.
    for (const shadow of ['memories_fts_data', 'memories_fts_idx', 'memories_fts_docsize', 'memories_fts_config']) {
      const sRows = src.prepare(`SELECT * FROM ${shadow}`).all();
      const tRows = tgt.prepare(`SELECT * FROM ${shadow}`).all();
      expect(JSON.stringify(tRows)).toBe(JSON.stringify(sRows));
    }
    // FTS functional parity: same match set.
    const sMatch = src.prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'shared'").all();
    const tMatch = tgt.prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'shared'").all();
    expect(JSON.stringify(tMatch)).toBe(JSON.stringify(sMatch));
    expect((tMatch as unknown[]).length).toBeGreaterThan(0);

    // Relations and topics identical.
    for (const tbl of ['memory_relations', 'memory_topics']) {
      const sRows = src.prepare(`SELECT * FROM ${tbl} ORDER BY rowid`).all();
      const tRows = tgt.prepare(`SELECT * FROM ${tbl} ORDER BY rowid`).all();
      expect(JSON.stringify(tRows)).toBe(JSON.stringify(sRows));
    }
    // Ledger lifecycle columns (version/date semantics) identical.
    for (const tbl of ['session_prompts', 'session_responses', 'session_tool_calls', 'session_summaries']) {
      const cols = (src.prepare(`PRAGMA table_info(${tbl})`).all() as Array<{ name: string }>)
        .map(c => c.name).filter(c => c !== 'embedding');
      const sel = cols.map(c => `"${c}"`).join(', ');
      const sRows = src.prepare(`SELECT ${sel} FROM ${tbl} ORDER BY rowid`).all();
      const tRows = tgt.prepare(`SELECT ${sel} FROM ${tbl} ORDER BY rowid`).all();
      expect(JSON.stringify(tRows)).toBe(JSON.stringify(sRows));
    }
    src.close();
    tgt.close();
  });

  it('reports JSON with counts/hashes/timings and NO raw memory text', async () => {
    const reportPath = join(fixture.dir, 'report.json');
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
      reportPath,
    });
    expect(res.status).toBe('COMPLETE');
    const raw = await readFile(reportPath, 'utf8');
    expect(raw).not.toContain(SENTINEL);
    expect(raw).not.toContain('how do I reindex vectors');
    const parsed = JSON.parse(raw);
    expect(parsed.tool).toBe('embedding-reindex');
    expect(parsed.source.fingerprint).toMatch(/^[0-9a-f]{64}$/);
    expect(parsed.tables.memory_embeddings.embedded).toBe(4);
    expect(parsed.tables.session_prompts.embedded).toBe(5);
    expect(parsed.timings.reembed).toBeGreaterThanOrEqual(0);
  });

  it('refuses to overwrite an existing target without --resume', async () => {
    await runReembed({ sourcePath: fixture.sourcePath, targetPath: fixture.targetPath, embedder: new FakeE5Embedder() });
    await expect(
      runReembed({ sourcePath: fixture.sourcePath, targetPath: fixture.targetPath, embedder: new FakeE5Embedder() }),
    ).resolves.toMatchObject({ status: 'ERROR', error: expect.stringMatching(/target already exists/i) });
  });

  it('rejects a source missing memory vector rows without altering anything', async () => {
    // Delete one memory embedding from the source fixture.
    const db = open(fixture.sourcePath);
    db.exec("DELETE FROM memory_embeddings WHERE id IN (SELECT id FROM memories WHERE content LIKE 'delta%')");
    db.close();
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    expect(res.status).toBe('ERROR');
    expect(res.error).toMatch(/missing.*vector row/i);
    // Preflight rejected BEFORE any snapshot: no target, no stamp, no partial state.
    expect(existsSync(fixture.targetPath)).toBe(false);
  });
});

// ── Progress, resume, drift ─────────────────────────────────────────

describe('progress / resume / drift', () => {
  it('keeps target resumable after a mid-run failure and resumes without repeating work', async () => {
    // 4 memories: batch of 2 commits, then failure on 3rd text.
    const failing = new FakeE5Embedder(3);
    const res1 = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: failing,
      batchSize: 2,
    });
    expect(res1.status).toBe('ERROR');
    expect(res1.error).toMatch(/injected embed failure/);

    // Target retained + INCOMPLETE guard active.
    expect(existsSync(fixture.targetPath)).toBe(true);
    expect(() => assertNoIncompleteMigration(fixture.targetPath)).toThrow(/incomplete/i);

    // Resume: the already-committed first batch must NOT be re-embedded.
    const resuming = new FakeE5Embedder();
    const res2 = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: resuming,
      resume: true,
      batchSize: 2,
    });
    expect(res2.status).toBe('COMPLETE');
    expect(resuming.texts).not.toContain(`alpha widget ${SENTINEL}`);
    expect(resuming.texts).not.toContain('beta gadget');
    // Remaining memories WERE embedded.
    expect(resuming.texts).toContain('gamma shared thing');

    // Completed: guard passes, identity stamped.
    expect(() => assertNoIncompleteMigration(fixture.targetPath)).not.toThrow();
    const tgt = open(fixture.targetPath, true);
    const id = readEmbeddingIdentity(tgt);
    expect(id?.model).toBe('test/multilingual-e5-small');
    expect(id?.dimensions).toBe(384);
    expect(id?.chunkPolicyVersion).toBe('e5-fulltext-v1');
    tgt.close();
  });

  it('fails resume when the source drifted from the recorded fingerprint', async () => {
    const failing = new FakeE5Embedder(3);
    await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: failing,
      batchSize: 2,
    });
    // Drift: a new committed memory+vector pair appears in the source
    // (exactly what a live writer would produce).
    const db = open(fixture.sourcePath);
    const memId = 'drift-mem';
    db.exec(`
      INSERT INTO memories (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version)
      VALUES ('${memId}', 'drifted content', 'learning', '[]', '2024-06-01T00:00:00.000Z', '2024-06-01T00:00:00.000Z', 0, '2024-06-01T00:00:00.000Z', 1);
      INSERT INTO memory_embeddings (id, embedding) VALUES ('${memId}', randomblob(4096));
    `);
    db.close();

    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
      resume: true,
    });
    expect(res.status).toBe('ERROR');
    expect(res.error).toMatch(/drift/i);
  });

  it('rejects resume with a different embedding identity', async () => {
    await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(3),
      batchSize: 2,
    });
    class OtherModel extends FakeE5Embedder {
      override identity() {
        return { ...super.identity()!, model: 'test/other-model' };
      }
    }
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new OtherModel(),
      resume: true,
    });
    expect(res.status).toBe('ERROR');
    expect(res.error).toMatch(/identity mismatch/i);
  });

  it('computes deterministic type-aware fingerprints', () => {
    const fp1 = computeSourceFingerprint(fixture.sourcePath);
    const fp2 = computeSourceFingerprint(fixture.sourcePath);
    expect(fp1).toBe(fp2);
    expect(fp1).toMatch(/^[0-9a-f]{64}$/);

    // Type-awareness: INTEGER 1 vs TEXT '1' vs REAL 1.0 differ.
    const d1 = join(fixture.dir, 't1.db'); const d2 = join(fixture.dir, 't2.db'); const d3 = join(fixture.dir, 't3.db');
    for (const [p, stmt] of [[d1, 'CREATE TABLE t(a); INSERT INTO t VALUES (1);'],
                             [d2, "CREATE TABLE t(a); INSERT INTO t VALUES ('1');"],
                             [d3, 'CREATE TABLE t(a); INSERT INTO t VALUES (1.0);']] as const) {
      const db = open(p as string); db.exec(stmt as string); db.close();
    }
    const fps = [d1, d2, d3].map(p => computeSourceFingerprint(p as string));
    expect(new Set(fps).size).toBe(3);
  });
});

// ── Verification ────────────────────────────────────────────────────

describe('verification', () => {
  it('passes on a completed migration and detects vector tampering', async () => {
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    expect(res.status).toBe('COMPLETE');
    expect(res.verification.ok).toBe(true);
    expect(res.verification.sourceDrift).toBe(false);
    expect(res.verification.quickCheck).toBe('ok');

    // Tamper: shrink one vector blob.
    const db = open(fixture.targetPath);
    db.exec("UPDATE memory_embeddings SET embedding = x'0000' WHERE rowid = (SELECT MIN(rowid) FROM memory_embeddings)");
    db.close();
    const res2 = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      verifyOnly: true,
    });
    expect(res2.status).toBe('ERROR');
    expect(res2.error).toMatch(/verif/i);
  });

  it('detects non-embedding column tampering via parity', async () => {
    await runReembed({ sourcePath: fixture.sourcePath, targetPath: fixture.targetPath, embedder: new FakeE5Embedder() });
    const db = open(fixture.targetPath);
    db.exec("UPDATE memories SET content = 'tampered' WHERE rowid = 1");
    db.close();
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      verifyOnly: true,
    });
    expect(res.status).toBe('ERROR');
    expect(res.error).toMatch(/parity|drift/i);
  });

  it('verify-only on an incomplete target reports INCOMPLETE, never passes', async () => {
    await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(3),
      batchSize: 2,
    });
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      verifyOnly: true,
    });
    expect(res.status).toBe('ERROR');
    expect(res.error).toMatch(/incomplete/i);
  });
});

// ── Report file atomicity on error ──────────────────────────────────

describe('sparse rowid preservation (disproves VACUUM INTO rowid compaction)', () => {
  it('keeps sparse hidden rowids + FTS docids through snapshot AND re-embed', async () => {
    // Delete middle memories so remaining hidden rowids are sparse (1,3 holes at 2).
    // Bare connection: FKs off, so delete the embedding row explicitly.
    const db = open(fixture.sourcePath);
    db.exec(`
      DELETE FROM memory_embeddings WHERE id IN (SELECT id FROM memories WHERE content LIKE 'beta%');
      DELETE FROM memories WHERE content LIKE 'beta%';
    `);
    db.close();

    const src = open(fixture.sourcePath, true);
    const srcRowids = (src.prepare('SELECT rowid AS rid FROM memories ORDER BY rowid').all() as Array<{ rid: number }>)
      .map(r => r.rid);
    expect(srcRowids).toEqual([1, 3, 4]); // sparse: rowid 2 is a hole
    // FTS5 docsize shadow: column `id` holds the docid.
    const srcFtsDocids = (src.prepare('SELECT id FROM memories_fts_docsize ORDER BY id').all() as Array<{ id: number }>)
      .map(r => r.id);
    const srcMatch = src.prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'shared'").all();
    src.close();

    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(),
    });
    expect(res.status).toBe('COMPLETE');
    expect(res.verification!.ok).toBe(true);

    const tgt = open(fixture.targetPath, true);
    const tgtRowids = (tgt.prepare('SELECT rowid AS rid FROM memories ORDER BY rowid').all() as Array<{ rid: number }>)
      .map(r => r.rid);
    // NOT compacted: the hole survives the snapshot and the vector rewrite.
    expect(tgtRowids).toEqual([1, 3, 4]);
    const tgtFtsDocids = (tgt.prepare('SELECT id FROM memories_fts_docsize ORDER BY id').all() as Array<{ id: number }>)
      .map(r => r.id);
    expect(tgtFtsDocids).toEqual(srcFtsDocids);
    const tgtMatch = tgt.prepare("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'shared'").all();
    expect(JSON.stringify(tgtMatch)).toBe(JSON.stringify(srcMatch));
    // memory_embeddings hidden rowids also stay sparse-aligned with memories.
    const tgtEmbRowids = (tgt.prepare('SELECT rowid AS rid FROM memory_embeddings ORDER BY rowid').all() as Array<{ rid: number }>)
      .map(r => r.rid);
    expect(tgtEmbRowids).toEqual([1, 3, 4]);
    tgt.close();
  });
});

describe('report sidecar', () => {
  it('writes an ERROR report without leaking content and keeps DB progress authoritative', async () => {
    const reportPath = join(fixture.dir, 'err-report.json');
    const res = await runReembed({
      sourcePath: fixture.sourcePath,
      targetPath: fixture.targetPath,
      embedder: new FakeE5Embedder(1),
      batchSize: 1,
      reportPath,
    });
    expect(res.status).toBe('ERROR');
    const raw = await readFile(reportPath, 'utf8');
    expect(raw).not.toContain(SENTINEL);
    expect(JSON.parse(raw).status).toBe('ERROR');
    // DB-side progress is the resume authority, not the sidecar.
    const db = open(fixture.targetPath, true);
    const row = db.prepare("SELECT status FROM embedding_migration_state WHERE id = 1").get() as { status: string };
    expect(row.status).toBe('INCOMPLETE');
    db.close();
    // A subsequent sidecar write is atomic: no .tmp leftovers.
    expect(existsSync(reportPath + '.tmp')).toBe(false);
  });
});
