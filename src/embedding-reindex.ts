// src/embedding-reindex.ts
//
// M2 (E5 small migration): snapshot + re-vectorization tooling.
//
// Scope discipline:
//   - the SOURCE database is opened readOnly:true and is NEVER mutated,
//     checkpointed, PRAGMA-tuned, or re-stamped; the global backend is never
//     loaded;
//   - the target is a NEW sibling DB produced by `VACUUM INTO` from the
//     read-only source (captures committed WAL content — verified);
//   - ONLY vector-bearing columns are rewritten; every other column, rowid,
//     table, FTS shadow table, relation, topic and session row is preserved
//     byte-for-byte (verified by per-table parity hashes);
//   - no store/update/re-store semantics: no new timestamps, no history rows,
//     no collision detection;
//   - the embedding identity is stamped (via the M1 public API
//     `writeEmbeddingIdentity`) ONLY after every vector and every parity check
//     passes; until then the target is INCOMPLETE and the exported guard
//     `assertNoIncompleteMigration` denies runtime opening.
//
// Durability model: each batch's vector UPDATEs and its progress checkpoint
// commit in ONE transaction on the target. The JSON sidecar report is a
// best-effort observability artifact written atomically (tmp+rename) AFTER the
// DB commit — a sidecar failure can never repeat or lose committed data;
// resume reads progress from the target's metadata tables, not the sidecar.
import { DatabaseSync } from 'node:sqlite';
import { createHash } from 'node:crypto';
import { statSync, realpathSync } from 'node:fs';
import { writeFile, rename } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import type { EmbeddingIdentity } from './types.js';
import {
  writeEmbeddingIdentity,
  assertMigrationStateComplete,
  MIGRATION_STATE_TABLE as SHARED_MIGRATION_STATE_TABLE,
  MigrationStateIncompleteError,
} from './sqlite-store.js';

// Re-export the shared guard surface (single source of truth in sqlite-store).
export { assertMigrationStateComplete, MigrationStateIncompleteError };

// ── Constants ──────────────────────────────────────────────────────

/** Snapshot method (exposed for reports/tests). */
export const SNAPSHOT_METHOD = 'VACUUM INTO';

/** Target-only migration metadata tables (excluded from parity). */
export const MIGRATION_STATE_TABLE = SHARED_MIGRATION_STATE_TABLE;
export const MIGRATION_PROGRESS_TABLE = 'embedding_migration_progress';

/** Metadata tables explicitly excluded from source↔target parity. */
const METADATA_TABLES = new Set([
  'embedding_meta', // M1 identity stamp (target-only, added last)
  MIGRATION_STATE_TABLE,
  MIGRATION_PROGRESS_TABLE,
]);

/** Default streaming batch size (parent spec: 25/50). */
export const DEFAULT_BATCH_SIZE = 25;

/** L2-norm tolerance for normalized identities (parent spec: 1e-4). */
export const NORM_TOLERANCE = 1e-4;

// ── Embedder surface ───────────────────────────────────────────────

/**
 * Minimal embedder contract for re-vectorization. Structurally satisfied by
 * `TransformersEmbedder` (real offline E5) and by test fakes. `cleanup` is
 * optional and duck-typed so the CLI can dispose native resources when the
 * concrete embedder exposes it.
 */
export interface ReindexEmbedder {
  initialize(): Promise<void>;
  embed(text: string, role?: 'query' | 'passage'): Promise<number[]>;
  embedBatch(texts: string[], role?: 'query' | 'passage'): Promise<number[][]>;
  dimensions(): number;
  identity?(): EmbeddingIdentity | undefined;
  cleanup?(): void | Promise<void>;
}

// ── Result types ───────────────────────────────────────────────────

export interface TableProgress {
  table: string;
  total: number;
  embedded: number;
  lastRowid: number;
}

export interface VerificationResult {
  ok: boolean;
  quickCheck: string;
  sourceDrift: boolean;
  parityTablesChecked: number;
  vectorsChecked: number;
  maxNormDeviation: number | null;
}

export interface ReembedResult {
  status: 'COMPLETE' | 'ERROR';
  mode: 'migrate' | 'resume' | 'verify-only';
  startedAt: string;
  finishedAt: string;
  source: { path: string; fingerprint: string | null };
  target: { path: string };
  snapshot?: { method: string; walBytesCaptured: number };
  identity: EmbeddingIdentity | null;
  batchSize: number;
  tables: Record<string, { total: number; embedded: number }>;
  counts: { totalRows: number; totalEmbedded: number };
  timings: {
    preflight: number;
    snapshot: number;
    fingerprint: number;
    reembed: number;
    validate: number;
    stamp: number;
    total: number;
  };
  verification: VerificationResult | null;
  reportPath: string | null;
  error: string | null;
}

export interface ReembedOptions {
  sourcePath: string;
  targetPath: string;
  embedder?: ReindexEmbedder;
  batchSize?: number;
  /** Sidecar JSON report path (best-effort, atomic). */
  reportPath?: string;
  /** Continue an interrupted migration on the SAME snapshot. */
  resume?: boolean;
  /** Verify an existing target only (requires status COMPLETE). */
  verifyOnly?: boolean;
}

// ── Errors ─────────────────────────────────────────────────────────

// MigrationStateIncompleteError is defined (and re-exported above) in
// src/sqlite-store.ts so the runtime guard and this tooling share one class.

// ── Path validation ────────────────────────────────────────────────

function canonicalExisting(p: string): string {
  return realpathSync.native(resolve(p));
}

function canonicalNonExisting(p: string): string {
  const abs = resolve(p);
  return join(realpathSync.native(dirname(abs)), abs.split(/[\\/]/).pop()!);
}

function normPath(p: string): string {
  return process.platform === 'win32' ? p.toLowerCase() : p;
}

/**
 * Reject any target that IS the source under another name: same canonical
 * path (symlink aliases resolve here), a hardlink (same dev+ino), or a
 * case/separator variant of a not-yet-existing target path. A brand-new
 * distinct target (e.g. sibling memory-db-e5.db) is accepted.
 */
export function assertDistinctPaths(sourcePath: string, targetPath: string): void {
  const src = canonicalExisting(sourcePath);
  let targetExists = true;
  let tgt: string;
  try {
    tgt = canonicalExisting(targetPath);
  } catch {
    targetExists = false;
    tgt = canonicalNonExisting(targetPath);
  }
  if (normPath(src) === normPath(tgt)) {
    const kind = targetExists ? ' (symlink or alias)' : '';
    throw new Error(`Source and target resolve to the same file${kind}: ${src}`);
  }
  if (targetExists) {
    const s = statSync(src);
    const t = statSync(tgt);
    if (s.ino && t.ino && s.ino === t.ino && s.dev === t.dev) {
      throw new Error(`Target is a hardlink of the source: ${tgt}`);
    }
  }
}

// ── Deterministic type-aware fingerprint ───────────────────────────

interface TableInfoRow {
  cid: number;
  name: string;
  type: string;
  notnull: number;
  dflt_value: unknown;
  pk: number;
}

/** User-level logical tables (real tables incl. FTS shadow, excl. virtual + sqlite internals). */
function listLogicalTables(db: DatabaseSync): string[] {
  const rows = db.prepare(
    `SELECT name, sql FROM sqlite_master
     WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
       AND (sql IS NULL OR sql NOT LIKE 'CREATE VIRTUAL TABLE%')
     ORDER BY name`,
  ).all() as unknown as Array<{ name: string; sql: string | null }>;
  return rows.map(r => r.name);
}

interface ColumnInfo {
  name: string;
  pk: number;
}

function tableColumns(db: DatabaseSync, table: string): ColumnInfo[] {
  const rows = db.prepare(`PRAGMA table_info("${table}")`).all() as unknown as TableInfoRow[];
  return rows.map(r => ({ name: r.name, pk: r.pk }));
}

/**
 * Deterministic ORDER BY for any table: rowid tables by rowid; WITHOUT ROWID
 * tables (e.g. FTS5 shadow `_config`/`_idx`) by their PRIMARY KEY columns.
 */
function orderByClause(table: string, sql: string | null, cols: ColumnInfo[]): { orderExpr: string; ridExpr: string } {
  if (sql && sql.includes('WITHOUT ROWID')) {
    const pkCols = cols.filter(c => c.pk > 0).sort((a, b) => a.pk - b.pk).map(c => `"${c.name}"`);
    if (pkCols.length === 0) throw new Error(`WITHOUT ROWID table '${table}' has no PRIMARY KEY columns.`);
    // Hash the PK tuple as the row identity (no rowid exists).
    return { orderExpr: pkCols.join(', '), ridExpr: `(${pkCols.join(' || \'|\' || ')})` };
  }
  return { orderExpr: 'rowid', ridExpr: 'rowid' };
}

function hashValue(h: ReturnType<typeof createHash>, tag: string, payload: Buffer | null): void {
  h.update(tag);
  if (payload) {
    const len = Buffer.allocUnsafe(8);
    len.writeBigUInt64LE(BigInt(payload.length));
    h.update(len);
    h.update(payload);
  }
}

function hashSqlValue(h: ReturnType<typeof createHash>, typeTag: string, value: unknown): void {
  switch (typeTag) {
    case 'null':
      hashValue(h, '0', null);
      break;
    case 'integer':
      hashValue(h, 'I', Buffer.from(String(value), 'utf8'));
      break;
    case 'real': {
      const buf = Buffer.allocUnsafe(8);
      buf.writeDoubleLE(Number(value));
      hashValue(h, 'R', buf);
      break;
    }
    case 'text':
      hashValue(h, 'T', Buffer.from(String(value), 'utf8'));
      break;
    case 'blob': {
      const b = Buffer.isBuffer(value) ? value : Buffer.from(value as Uint8Array);
      hashValue(h, 'B', b);
      break;
    }
    default:
      hashValue(h, '?', Buffer.from(String(value), 'utf8'));
  }
}

/**
 * Hash one table's rows deterministically: rowid-ordered, type-aware
 * (`typeof()` per column, so INTEGER 1 ≠ REAL 1.0 ≠ TEXT '1'), rowid included
 * so vector-rowid parity is covered. `excludeColumns` skips vector columns for
 * the non-embedding parity pass.
 */
function hashTable(
  db: DatabaseSync,
  table: string,
  excludeColumns: ReadonlySet<string>,
): { count: number; hash: string } {
  const allCols = tableColumns(db, table);
  const meta = db.prepare(
    `SELECT sql FROM sqlite_master WHERE type='table' AND name=?`,
  ).get(table) as { sql: string | null } | undefined;
  const { orderExpr, ridExpr } = orderByClause(table, meta?.sql ?? null, allCols);
  const cols = allCols.map(c => c.name).filter(c => !excludeColumns.has(c));
  const typeSel = cols.map(c => `typeof("${c}") AS "t_${c}"`).join(', ');
  const valSel = cols.map(c => `"${c}"`).join(', ');
  const rows = db.prepare(
    `SELECT ${ridExpr} AS "rid", ${valSel}, ${typeSel} FROM "${table}" ORDER BY ${orderExpr}`,
  ).all() as unknown as Array<Record<string, unknown>>;
  const h = createHash('sha256');
  for (const row of rows) {
    h.update(`R${String(row['rid'])};`);
    for (const c of cols) {
      hashSqlValue(h, String(row[`t_${c}`]), row[c]);
    }
  }
  return { count: rows.length, hash: h.digest('hex') };
}

/**
 * Full logical fingerprint of a database: schema (sqlite_master + PRAGMA
 * user_version) plus every row of every logical table, type-aware and
 * rowid-ordered, INCLUDING the original vector bytes. Deterministic for
 * identical content; any committed row/vector/schema change flips it.
 */
export function computeSourceFingerprint(sourcePath: string): string {
  const db = new DatabaseSync(sourcePath, { readOnly: true });
  try {
    const h = createHash('sha256');
    const master = db.prepare(
      `SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name`,
    ).all() as unknown as Array<Record<string, unknown>>;
    for (const r of master) {
      h.update(`${String(r['type'])}|${String(r['name'])}|${String(r['tbl_name'])}|`);
      hashSqlValue(h, r['sql'] == null ? 'null' : 'text', r['sql']);
      h.update('\n');
    }
    const uv = db.prepare('PRAGMA user_version').get() as { user_version: number };
    h.update(`user_version=${uv.user_version};`);
    for (const table of listLogicalTables(db)) {
      h.update(`T:${table};`);
      const { count, hash } = hashTable(db, table, EMPTY_SET);
      h.update(`n=${count};h=${hash};`);
    }
    return h.digest('hex');
  } finally {
    db.close();
  }
}

const EMPTY_SET: ReadonlySet<string> = new Set();

// ── Vector-bearing table plans ─────────────────────────────────────

interface VectorTablePlan {
  table: string;
  /** Qualified rowid expression for streaming order/cursor (e.g. `e.rowid`). */
  rowidExpr: string;
  /** SELECT list providing `rid` + `txt` (JS-computed text when needed). */
  selectSql: string;
  /** FROM/JOIN clause (without the WHERE). */
  fromSql: string;
  /** When set, `txt` is computed in JS from these raw columns. */
  jsText?: (row: Record<string, unknown>) => string;
  /** Non-embedding columns excluded from parity hashing. */
  excludeFromParity: ReadonlySet<string>;
}

const VECTOR_PLANS: readonly VectorTablePlan[] = [
  {
    table: 'memory_embeddings',
    rowidExpr: 'e.rowid',
    selectSql: 'SELECT e.rowid AS rid, m.content AS txt',
    fromSql: 'FROM memory_embeddings e JOIN memories m ON m.id = e.id',
    excludeFromParity: new Set(['embedding']),
  },
  {
    table: 'session_prompts',
    rowidExpr: 'rowid',
    selectSql: 'SELECT rowid AS rid, raw_prompt AS txt',
    fromSql: 'FROM session_prompts',
    excludeFromParity: new Set(['embedding']),
  },
  {
    table: 'session_responses',
    rowidExpr: 'rowid',
    selectSql: 'SELECT rowid AS rid, full_response AS txt',
    fromSql: 'FROM session_responses',
    excludeFromParity: new Set(['embedding']),
  },
  {
    table: 'session_tool_calls',
    rowidExpr: 'rowid',
    // Matches storeSessionToolCall: resultSummary || toolName (JS truthiness).
    selectSql:
      `SELECT rowid AS rid, tool_name, result_summary AS txt, ` +
      `CASE WHEN result_summary IS NOT NULL AND result_summary != '' THEN 1 ELSE 0 END AS has_summary`,
    fromSql: 'FROM session_tool_calls',
    jsText: (row) =>
      (row['has_summary'] === 1 ? String(row['txt']) : String(row['tool_name'])),
    excludeFromParity: new Set(['embedding']),
  },
  {
    table: 'session_summaries',
    rowidExpr: 'rowid',
    // Matches storeSessionSummary: `${summary} ${decisions.join(' ')}`.trim().
    selectSql: 'SELECT rowid AS rid, summary, decisions AS txt',
    fromSql: 'FROM session_summaries',
    jsText: (row) => {
      let decisions: string[] = [];
      if (row['txt'] != null) {
        try { decisions = JSON.parse(String(row['txt'])) as string[]; } catch { decisions = []; }
      }
      return `${String(row['summary'])} ${decisions.join(' ')}`.trim();
    },
    excludeFromParity: new Set(['embedding']),
  },
];

// ── Preflight ──────────────────────────────────────────────────────

interface PreflightResult {
  /** memory rows without a memory_embeddings row (hard reject). */
  missingVectorRows: number;
  /** memory_embeddings rows without a memory row (hard reject). */
  orphanVectorRows: number;
  tables: Record<string, number>;
}

function preflight(sourcePath: string): PreflightResult {
  const db = new DatabaseSync(sourcePath, { readOnly: true });
  try {
    const hasMemories = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='memories'`,
    ).get();
    if (!hasMemories) {
      throw new Error(`Source '${sourcePath}' has no memories table — not a PMC database.`);
    }
    const missing = db.prepare(
      `SELECT COUNT(*) AS n FROM memories m LEFT JOIN memory_embeddings e ON e.id = m.id WHERE e.id IS NULL`,
    ).get() as { n: number };
    const orphan = db.prepare(
      `SELECT COUNT(*) AS n FROM memory_embeddings e LEFT JOIN memories m ON m.id = e.id WHERE m.id IS NULL`,
    ).get() as { n: number };
    const tables: Record<string, number> = {};
    for (const plan of VECTOR_PLANS) {
      const exists = db.prepare(
        `SELECT name FROM sqlite_master WHERE type='table' AND name=?`,
      ).get(plan.table);
      tables[plan.table] = exists
        ? (db.prepare(`SELECT COUNT(*) AS n FROM ${plan.table}`).get() as { n: number }).n
        : 0;
    }
    return { missingVectorRows: missing.n, orphanVectorRows: orphan.n, tables };
  } finally {
    db.close();
  }
}

// ── Snapshot ───────────────────────────────────────────────────────

function snapshotSource(sourcePath: string, targetPath: string): { method: string; walBytesCaptured: number } {
  const src = new DatabaseSync(sourcePath, { readOnly: true });
  try {
    let walBytesCaptured = 0;
    try {
      walBytesCaptured = statSync(`${resolve(sourcePath)}-wal`).size;
    } catch { /* no WAL file */ }
    // Bound parameter — no SQL-string path interpolation. VACUUM INTO refuses
    // to overwrite an existing file, giving a second overwrite guard.
    const targetCanonical = join(
      realpathSync.native(dirname(resolve(targetPath))),
      resolve(targetPath).split(/[\\/]/).pop()!,
    );
    src.prepare('VACUUM INTO ?').run(targetCanonical);
    return { method: SNAPSHOT_METHOD, walBytesCaptured };
  } finally {
    src.close();
  }
}

// ── Migration state (target metadata) ──────────────────────────────

function ensureStateTables(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${MIGRATION_STATE_TABLE} (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      status TEXT NOT NULL,
      source_path TEXT NOT NULL,
      source_fingerprint TEXT NOT NULL,
      identity_json TEXT NOT NULL,
      started_at TEXT NOT NULL,
      completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS ${MIGRATION_PROGRESS_TABLE} (
      table_name TEXT PRIMARY KEY,
      last_rowid INTEGER NOT NULL,
      total INTEGER NOT NULL,
      embedded INTEGER NOT NULL
    );
  `);
}

interface StateRow {
  status: string;
  source_path: string;
  source_fingerprint: string;
  identity_json: string;
  started_at: string;
}

function readState(db: DatabaseSync): StateRow | null {
  const table = db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table' AND name=?`,
  ).get(MIGRATION_STATE_TABLE);
  if (!table) return null;
  return (db.prepare(`SELECT * FROM ${MIGRATION_STATE_TABLE} WHERE id = 1`).get() as StateRow | undefined) ?? null;
}

function readProgress(db: DatabaseSync, table: string): { lastRowid: number; embedded: number } {
  const row = db.prepare(
    `SELECT last_rowid, embedded FROM ${MIGRATION_PROGRESS_TABLE} WHERE table_name = ?`,
  ).get(table) as { last_rowid: number; embedded: number } | undefined;
  return { lastRowid: row?.last_rowid ?? 0, embedded: row?.embedded ?? 0 };
}

/**
 * Path-based wrapper over the shared runtime guard: throws when the target
 * carries a non-COMPLETE migration stamp (INCOMPLETE/ERROR/unknown/missing
 * row). A DB without migration metadata is not a migration target and passes.
 */
export function assertNoIncompleteMigration(dbPath: string): void {
  const db = new DatabaseSync(dbPath, { readOnly: true });
  try {
    assertMigrationStateComplete(db);
  } finally {
    db.close();
  }
}

// ── Vector validation ──────────────────────────────────────────────

interface VectorAudit {
  checked: number;
  maxNormDeviation: number | null;
  failures: string[];
}

function auditVectors(db: DatabaseSync, dims: number, normalize: boolean): VectorAudit {
  const audit: VectorAudit = { checked: 0, maxNormDeviation: normalize ? 0 : null, failures: [] };
  const expectedBytes = dims * 4;
  for (const plan of VECTOR_PLANS) {
    const exists = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name=?`,
    ).get(plan.table);
    if (!exists) continue;
    const rows = db.prepare(
      `SELECT rowid AS rid, embedding FROM ${plan.table} WHERE embedding IS NOT NULL`,
    ).all() as unknown as Array<{ rid: number; embedding: Buffer | Uint8Array }>;
    for (const row of rows) {
      audit.checked++;
      const buf = Buffer.isBuffer(row.embedding) ? row.embedding : Buffer.from(row.embedding);
      if (buf.byteLength !== expectedBytes) {
        audit.failures.push(`${plan.table}#${row.rid}: ${buf.byteLength} bytes != ${expectedBytes}`);
        continue;
      }
      const view = new Float32Array(buf.buffer, buf.byteOffset, dims);
      let sumSq = 0;
      let finite = true;
      for (let i = 0; i < dims; i++) {
        const x = view[i];
        if (!Number.isFinite(x)) { finite = false; break; }
        sumSq += x * x;
      }
      if (!finite) {
        audit.failures.push(`${plan.table}#${row.rid}: non-finite component`);
        continue;
      }
      if (normalize) {
        const dev = Math.abs(Math.sqrt(sumSq) - 1);
        if (dev > (audit.maxNormDeviation ?? 0)) audit.maxNormDeviation = dev;
        if (dev > NORM_TOLERANCE) {
          audit.failures.push(`${plan.table}#${row.rid}: L2 norm deviation ${dev} > ${NORM_TOLERANCE}`);
        }
      }
    }
  }
  return audit;
}

// ── Parity verification ────────────────────────────────────────────

function verifyParity(sourcePath: string, targetPath: string): { tablesChecked: number; mismatch: string | null } {
  const src = new DatabaseSync(sourcePath, { readOnly: true });
  const tgt = new DatabaseSync(targetPath, { readOnly: true });
  try {
    const srcTables = listLogicalTables(src);
    const tgtTables = listLogicalTables(tgt).filter(t => !METADATA_TABLES.has(t));
    if (JSON.stringify(srcTables) !== JSON.stringify(tgtTables)) {
      return {
        tablesChecked: 0,
        mismatch: `table set differs: source=${JSON.stringify(srcTables)} target=${JSON.stringify(tgtTables)}`,
      };
    }
    let checked = 0;
    for (const table of srcTables) {
      const exclude = table === 'embedding_meta' ? EMPTY_SET : excludeFor(table);
      const s = hashTable(src, table, exclude);
      const t = hashTable(tgt, table, exclude);
      checked++;
      if (s.count !== t.count || s.hash !== t.hash) {
        return {
          tablesChecked: checked,
          mismatch: `parity mismatch in '${table}': source(count=${s.count},hash=${s.hash.slice(0, 16)}) ` +
            `target(count=${t.count},hash=${t.hash.slice(0, 16)})`,
        };
      }
    }
    return { tablesChecked: checked, mismatch: null };
  } finally {
    src.close();
    tgt.close();
  }
}

function excludeFor(table: string): ReadonlySet<string> {
  const plan = VECTOR_PLANS.find(p => p.table === table);
  return plan ? plan.excludeFromParity : EMPTY_SET;
}

// ── Main entry point ───────────────────────────────────────────────

export async function runReembed(opts: ReembedOptions): Promise<ReembedResult> {
  const startedAt = new Date().toISOString();
  const t0 = Date.now();
  const batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;
  const mode: ReembedResult['mode'] = opts.verifyOnly ? 'verify-only' : opts.resume ? 'resume' : 'migrate';
  const timings = { preflight: 0, snapshot: 0, fingerprint: 0, reembed: 0, validate: 0, stamp: 0, total: 0 };
  const result: ReembedResult = {
    status: 'ERROR',
    mode,
    startedAt,
    finishedAt: '',
    source: { path: resolve(opts.sourcePath), fingerprint: null },
    target: { path: resolve(opts.targetPath) },
    identity: null,
    batchSize,
    tables: {},
    counts: { totalRows: 0, totalEmbedded: 0 },
    timings,
    verification: null,
    reportPath: opts.reportPath ? resolve(opts.reportPath) : null,
    error: null,
  };

  try {
    // ── Path validation (both fresh and resume runs) ──
    assertDistinctPaths(opts.sourcePath, opts.targetPath);

    let targetExists = true;
    try { realpathSync.native(resolve(opts.targetPath)); } catch { targetExists = false; }

    if (opts.verifyOnly) {
      if (!targetExists) throw new Error(`Verify-only: target '${opts.targetPath}' does not exist.`);
    } else if (opts.resume) {
      if (!targetExists) throw new Error(`Resume: target '${opts.targetPath}' does not exist.`);
    } else if (targetExists) {
      throw new Error(
        `Target already exists: '${opts.targetPath}'. Refusing to overwrite — use --resume ` +
        `(must belong to the same source snapshot + embedding identity) or pick a new target.`,
      );
    }

    // ── Preflight (source, readOnly) — reject before any snapshot ──
    const tp = Date.now();
    const pf = preflight(opts.sourcePath);
    if (pf.missingVectorRows > 0) {
      throw new Error(
        `Preflight rejected: ${pf.missingVectorRows} memory row(s) missing vector rows in ` +
        `'memory_embeddings'. Re-vectorization never alters non-vector fields; repair the ` +
        `source (re-store those memories) before migrating.`,
      );
    }
    if (pf.orphanVectorRows > 0) {
      throw new Error(
        `Preflight rejected: ${pf.orphanVectorRows} orphan vector row(s) in 'memory_embeddings' ` +
        `without a memory row (unknown embed text).`,
      );
    }
    result.tables = Object.fromEntries(Object.entries(pf.tables).map(([k, v]) => [k, { total: v, embedded: 0 }]));
    result.counts.totalRows = Object.values(pf.tables).reduce((a, b) => a + b, 0);
    timings.preflight = Date.now() - tp;

    // ── Source fingerprint (drift baseline) ──
    const tf = Date.now();
    const fingerprint = computeSourceFingerprint(opts.sourcePath);
    result.source.fingerprint = fingerprint;
    timings.fingerprint = Date.now() - tf;

    // ── Snapshot (fresh run only) ──
    if (!opts.verifyOnly && !opts.resume) {
      const ts = Date.now();
      result.snapshot = snapshotSource(opts.sourcePath, opts.targetPath);
      timings.snapshot = Date.now() - ts;
    }

    // ── Embedder identity ──
    let identity: EmbeddingIdentity | null = null;
    if (!opts.verifyOnly) {
      if (!opts.embedder) throw new Error('An embedder is required unless --verify-only.');
      await opts.embedder.initialize();
      identity = opts.embedder.identity?.() ?? null;
      if (!identity) {
        throw new Error(
          'The embedder exposes no identity(); re-vectorization requires a stampable identity ' +
          '(model/dims/pooling/normalize/roles/chunk policy).',
        );
      }
      result.identity = identity;
    }

    // ── Open target + state handling ──
    const tgt = new DatabaseSync(resolve(opts.targetPath));
    try {
      if (opts.verifyOnly) {
        const state = readState(tgt);
        if (!state || state.status !== 'COMPLETE') {
          throw new Error(
            `Verify-only: migration state is ${state ? state.status : 'ABSENT'} — the target is ` +
            `INCOMPLETE and must not be treated as migrated. Resume with --resume.`,
          );
        }
        identity = JSON.parse(state.identity_json) as EmbeddingIdentity;
        result.identity = identity;
      } else {
        ensureStateTables(tgt);
        const state = readState(tgt);
        if (opts.resume) {
          if (!state) throw new Error('Resume: target has no migration state metadata.');
          if (state.status === 'COMPLETE') {
            throw new Error('Resume: target is already COMPLETE. Use --verify-only for checks.');
          }
          if (state.source_fingerprint !== fingerprint) {
            throw new Error(
              `Source drift detected: the source logical fingerprint changed since the snapshot ` +
              `(recorded ${state.source_fingerprint.slice(0, 16)}…, current ${fingerprint.slice(0, 16)}…). ` +
              `Refusing to mix a newer source state into the existing snapshot.`,
            );
          }
          const recorded = JSON.parse(state.identity_json) as EmbeddingIdentity;
          if (!sameIdentity(recorded, identity!)) {
            throw new Error(
              `Embedding identity mismatch on resume: snapshot was re-vectorized for ` +
              `${recorded.model}/${recorded.dimensions} but the current embedder is ` +
              `${identity!.model}/${identity!.dimensions}.`,
            );
          }
        } else {
          tgt.exec('BEGIN IMMEDIATE');
          try {
            tgt.prepare(
              `INSERT INTO ${MIGRATION_STATE_TABLE}
                 (id, status, source_path, source_fingerprint, identity_json, started_at, completed_at)
               VALUES (1, 'INCOMPLETE', ?, ?, ?, ?, NULL)`,
            ).run(resolve(opts.sourcePath), fingerprint, JSON.stringify(identity), startedAt);
            tgt.exec('COMMIT');
          } catch (err) {
            try { tgt.exec('ROLLBACK'); } catch { /* ignore */ }
            throw err;
          }
        }

        // ── Streaming re-embedding with per-batch atomic checkpoints ──
        const tr = Date.now();
        for (const plan of VECTOR_PLANS) {
          const exists = tgt.prepare(
            `SELECT name FROM sqlite_master WHERE type='table' AND name=?`,
          ).get(plan.table);
          if (!exists) continue;
          const total = (tgt.prepare(`SELECT COUNT(*) AS n FROM ${plan.table}`).get() as { n: number }).n;
          let { lastRowid, embedded } = readProgress(tgt, plan.table);
          for (;;) {
            const rows = tgt.prepare(
              `${plan.selectSql} ${plan.fromSql} WHERE ${plan.rowidExpr} > ? ORDER BY ${plan.rowidExpr} LIMIT ?`,
            ).all(lastRowid, batchSize) as unknown as Array<Record<string, unknown>>;
            if (rows.length === 0) break;

            const texts = rows.map(row => (plan.jsText ? plan.jsText(row) : String(row['txt'])));
            const vectors: number[][] = [];
            for (const text of texts) {
              vectors.push(await opts.embedder!.embed(text, 'passage'));
            }
            for (let i = 0; i < vectors.length; i++) {
              if (vectors[i].length !== identity!.dimensions) {
                throw new Error(
                  `Embedder returned ${vectors[i].length}-dim vector for ${plan.table}#${String(rows[i]['rid'])}; ` +
                  `identity declares ${identity!.dimensions}.`,
                );
              }
              if (vectors[i].length === 0) {
                throw new Error(`Embedder returned an empty (pending) vector for ${plan.table}#${String(rows[i]['rid'])}.`);
              }
            }

            lastRowid = Number(rows[rows.length - 1]['rid']);
            embedded += rows.length;
            // ONE transaction: all vector UPDATEs + the durable checkpoint.
            tgt.exec('BEGIN IMMEDIATE');
            try {
              const upd = tgt.prepare(`UPDATE ${plan.table} SET embedding = ? WHERE rowid = ?`);
              for (let i = 0; i < rows.length; i++) {
                upd.run(embeddingToBlob(vectors[i]), Number(rows[i]['rid']));
              }
              tgt.prepare(
                `INSERT INTO ${MIGRATION_PROGRESS_TABLE} (table_name, last_rowid, total, embedded)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT(table_name) DO UPDATE SET
                   last_rowid = excluded.last_rowid, total = excluded.total, embedded = excluded.embedded`,
              ).run(plan.table, lastRowid, total, embedded);
              tgt.exec('COMMIT');
            } catch (err) {
              try { tgt.exec('ROLLBACK'); } catch { /* ignore */ }
              throw err;
            }
            result.tables[plan.table] = { total, embedded };
            // Best-effort atomic sidecar checkpoint AFTER the DB commit: a
            // failure here cannot repeat or lose committed vector data.
            await writeSidecar(result, opts.reportPath, fingerprint, identity);
          }
          result.tables[plan.table] = { total, embedded };
        }
        timings.reembed = Date.now() - tr;
        result.counts.totalEmbedded = Object.values(result.tables).reduce((a, t) => a + t.embedded, 0);
      }

      // ── Validation (all vectors + parity + drift + integrity) ──
      const tv = Date.now();
      const idim = identity!.dimensions;
      const audit = auditVectors(tgt, idim, identity!.normalize);
      const parity = verifyParity(opts.sourcePath, opts.targetPath);
      const currentFingerprint = computeSourceFingerprint(opts.sourcePath);
      const sourceDrift = currentFingerprint !== fingerprint;
      const quick = tgt.prepare('PRAGMA quick_check').get() as { quick_check: string };
      const verification: VerificationResult = {
        ok: audit.failures.length === 0 && parity.mismatch === null && !sourceDrift && quick.quick_check === 'ok',
        quickCheck: quick.quick_check,
        sourceDrift,
        parityTablesChecked: parity.tablesChecked,
        vectorsChecked: audit.checked,
        maxNormDeviation: audit.maxNormDeviation,
      };
      result.verification = verification;
      timings.validate = Date.now() - tv;
      if (!verification.ok) {
        const reasons = [
          ...(audit.failures.length ? [`vector validation: ${audit.failures.slice(0, 3).join('; ')}${audit.failures.length > 3 ? ` (+${audit.failures.length - 3} more)` : ''}`] : []),
          ...(parity.mismatch ? [parity.mismatch] : []),
          ...(sourceDrift ? ['source drift detected during migration'] : []),
          ...(quick.quick_check !== 'ok' ? [`integrity: ${quick.quick_check}`] : []),
        ].join(' | ');
        throw new Error(`Verification failed — target kept INCOMPLETE and resumable: ${reasons}`);
      }

      // ── Stamp identity + COMPLETE only after full validation ──
      if (!opts.verifyOnly) {
        const tst = Date.now();
        tgt.exec('BEGIN IMMEDIATE');
        try {
          writeEmbeddingIdentity(tgt, identity!);
          tgt.prepare(
            `UPDATE ${MIGRATION_STATE_TABLE} SET status = 'COMPLETE', completed_at = ? WHERE id = 1`,
          ).run(new Date().toISOString());
          tgt.exec('COMMIT');
        } catch (err) {
          try { tgt.exec('ROLLBACK'); } catch { /* ignore */ }
          throw err;
        }
        timings.stamp = Date.now() - tst;
      }

      result.status = 'COMPLETE';
    } finally {
      tgt.close();
    }
  } catch (err) {
    result.error = err instanceof Error ? err.message : String(err);
  } finally {
    if (opts.embedder?.cleanup) {
      try { await opts.embedder.cleanup(); } catch { /* disposal is best-effort */ }
    }
    timings.total = Date.now() - t0;
    result.finishedAt = new Date().toISOString();
  }

  await writeSidecar(result, opts.reportPath, result.source.fingerprint, result.identity);
  return result;
}

// ── Helpers ────────────────────────────────────────────────────────

function embeddingToBlob(vec: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vec.length * 4);
  const view = new Float32Array(buf.buffer, buf.byteOffset, vec.length);
  view.set(vec);
  return buf;
}

function sameIdentity(a: EmbeddingIdentity, b: EmbeddingIdentity): boolean {
  return (
    a.model === b.model &&
    a.dimensions === b.dimensions &&
    a.pooling === b.pooling &&
    a.normalize === b.normalize &&
    a.roleAware === b.roleAware &&
    (a.chunkPolicy?.version ?? null) === (b.chunkPolicy?.version ?? null)
  );
}

/**
 * Atomic sidecar report (tmp + rename). Contains counts, hashes, timings and
 * identity only — never raw memory/prompt/response text.
 */
async function writeSidecar(
  result: ReembedResult,
  reportPath: string | undefined,
  fingerprint: string | null,
  identity: EmbeddingIdentity | null,
): Promise<void> {
  if (!reportPath) return;
  const payload = {
    tool: 'embedding-reindex',
    version: 1,
    status: result.status,
    mode: result.mode,
    startedAt: result.startedAt,
    finishedAt: result.finishedAt,
    source: { path: result.source.path, fingerprint },
    target: { path: result.target.path },
    ...(result.snapshot ? { snapshot: result.snapshot } : {}),
    identity,
    batchSize: result.batchSize,
    tables: result.tables,
    counts: result.counts,
    timings: result.timings,
    verification: result.verification,
    error: result.error,
  };
  const tmp = `${reportPath}.tmp`;
  try {
    await writeFile(tmp, JSON.stringify(payload, null, 2), 'utf8');
    await rename(tmp, reportPath);
  } catch (err) {
    // Best-effort observability: DB progress is authoritative; never fail the
    // migration because the sidecar could not be written.
    process.stderr.write(
      `[embedding-reindex] warning: could not write report sidecar '${reportPath}': ` +
      `${err instanceof Error ? err.message : String(err)}\n`,
    );
  }
}
