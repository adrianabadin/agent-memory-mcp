import { createHash, randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';
import type { Embedder } from '../types.js';
import {
  embeddingToBlob,
  blobToEmbedding,
  cosineDistance,
  rrf,
  escapeFtsQuery,
} from '../shared/vector-utils.js';

export interface ErrorRecord {
  id: string;
  projectId: string;
  message: string;
  stack: string | null;
  rootCause: string | null;
  fix: string;
  files: string[] | null;
  tags: string[];
  source: 'auto' | 'manual';
  createdAt: string;
  contentHash: string;
}

export interface RecordErrorRequest {
  projectId: string;
  message: string;
  stack?: string;
  rootCause?: string;
  fix: string;
  files?: string[];
  tags?: string[];
  source: 'auto' | 'manual';
}

export interface ErrorSearchResult {
  error: ErrorRecord;
  projectName: string;
  score: number;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS errors (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  message      TEXT NOT NULL,
  stack        TEXT,
  root_cause   TEXT,
  fix          TEXT NOT NULL,
  files        TEXT,
  tags         TEXT NOT NULL DEFAULT '[]',
  source       TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  content_hash TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS errors_content_hash_project
  ON errors(content_hash, project_id);

CREATE TABLE IF NOT EXISTS error_embeddings (
  id        TEXT PRIMARY KEY REFERENCES errors(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS errors_fts USING fts5(
  id UNINDEXED,
  message,
  root_cause,
  fix,
  tags,
  content='errors',
  content_rowid='rowid',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS errors_ai AFTER INSERT ON errors BEGIN
  INSERT INTO errors_fts(rowid, id, message, root_cause, fix, tags)
    VALUES (new.rowid, new.id, new.message, new.root_cause, new.fix, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS errors_ad AFTER DELETE ON errors BEGIN
  INSERT INTO errors_fts(errors_fts, rowid, id, message, root_cause, fix, tags)
    VALUES ('delete', old.rowid, old.id, old.message, old.root_cause, old.fix, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS errors_au AFTER UPDATE ON errors BEGIN
  INSERT INTO errors_fts(errors_fts, rowid, id, message, root_cause, fix, tags)
    VALUES ('delete', old.rowid, old.id, old.message, old.root_cause, old.fix, old.tags);
  INSERT INTO errors_fts(rowid, id, message, root_cause, fix, tags)
    VALUES (new.rowid, new.id, new.message, new.root_cause, new.fix, new.tags);
END;
`;

function rowToRecord(row: Record<string, unknown>): ErrorRecord {
  return {
    id: String(row['id']),
    projectId: String(row['project_id']),
    message: String(row['message']),
    stack: row['stack'] != null ? String(row['stack']) : null,
    rootCause: row['root_cause'] != null ? String(row['root_cause']) : null,
    fix: String(row['fix']),
    files: (() => {
      try {
        return JSON.parse(String(row['files'] ?? 'null'));
      } catch {
        return null;
      }
    })(),
    tags: (() => {
      try {
        return JSON.parse(String(row['tags'] ?? '[]'));
      } catch {
        return [];
      }
    })(),
    source: String(row['source']) as 'auto' | 'manual',
    createdAt: String(row['created_at']),
    contentHash: String(row['content_hash']),
  };
}

export class ErrorStore {
  constructor(
    private readonly db: DatabaseSync,
    private readonly embedder: Embedder,
  ) {}

  static computeContentHash(message: string, rootCause: string | null | undefined): string {
    return createHash('sha256')
      .update(message.trim())
      .update('\0')
      .update((rootCause ?? '').trim())
      .digest('hex');
  }

  ensureSchema(): void {
    this.db.exec(SCHEMA);
  }

  async record(req: RecordErrorRequest): Promise<ErrorRecord> {
    const id = randomUUID();
    const now = new Date().toISOString();
    const contentHash = ErrorStore.computeContentHash(req.message, req.rootCause);
    const tags = JSON.stringify(req.tags ?? []);
    const files = req.files ? JSON.stringify(req.files) : null;

    const embeddingText = [req.message, req.rootCause, req.fix].filter(Boolean).join('\n');
    const embedding = await this.embedder.embed(embeddingText, 'passage');

    this.db.exec('BEGIN');
    try {
      this.db.prepare(`
        INSERT INTO errors (id, project_id, message, stack, root_cause, fix, files, tags, source, created_at, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash, project_id) DO UPDATE SET
          fix        = excluded.fix,
          stack      = COALESCE(excluded.stack, errors.stack),
          files      = COALESCE(excluded.files, errors.files),
          tags       = excluded.tags
      `).run(
        id,
        req.projectId,
        req.message,
        req.stack ?? null,
        req.rootCause ?? null,
        req.fix,
        files,
        tags,
        req.source,
        now,
        contentHash,
      );

      const row = this.db.prepare(
        `SELECT id FROM errors WHERE content_hash = ? AND project_id = ?`,
      ).get(contentHash, req.projectId) as { id: string };

      this.db.prepare(
        `INSERT OR REPLACE INTO error_embeddings (id, embedding) VALUES (?, ?)`,
      ).run(row.id, embeddingToBlob(embedding));

      this.db.exec('COMMIT');

      return rowToRecord(
        this.db.prepare(`SELECT * FROM errors WHERE id = ?`).get(row.id) as Record<string, unknown>,
      );
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  async search(query: string, projectId?: string, limit = 10): Promise<ErrorSearchResult[]> {
    if (!query.trim()) return [];

    let ftsResults: Array<{ id: string; score: number }> = [];
    try {
      const escaped = escapeFtsQuery(query);
      type FtsRow = { id: string; fts_score: number };
      const ftsRows = this.db.prepare(`
        SELECT e.id, -bm25(errors_fts) AS fts_score
        FROM errors_fts
        JOIN errors e ON e.id = errors_fts.id
        WHERE errors_fts MATCH ?
        ${projectId ? 'AND e.project_id = ?' : ''}
        ORDER BY bm25(errors_fts)
        LIMIT ?
      `).all(
        ...(projectId ? [escaped, projectId, limit * 3] : [escaped, limit * 3]),
      ) as unknown as FtsRow[];
      ftsResults = ftsRows.map(r => ({ id: r.id, score: r.fts_score }));
    } catch {
      // FTS error (e.g. empty table) — continue with semantic only
    }

    const queryVec = await this.embedder.embed(query, 'query');
    const embeddingRows = this.db.prepare(`
      SELECT e.id, ee.embedding
      FROM errors e
      JOIN error_embeddings ee ON ee.id = e.id
      ${projectId ? 'WHERE e.project_id = ?' : ''}
    `).all(
      ...(projectId ? [projectId] : []),
    ) as unknown as { id: string; embedding: Buffer }[];

    const semanticResults = embeddingRows
      .map(r => ({ id: r.id, score: cosineDistance(queryVec, blobToEmbedding(r.embedding)) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit * 3);

    const merged = rrf([ftsResults, semanticResults]).slice(0, limit);
    if (merged.length === 0) return [];

    const scoreMap = new Map(merged.map(m => [m.id, m.score]));

    const rows = merged
      .map(({ id }) =>
        this.db.prepare(`
          SELECT e.*, p.name AS project_name
          FROM errors e
          JOIN projects p ON p.id = e.project_id
          WHERE e.id = ?
        `).get(id) as Record<string, unknown> | undefined,
      )
      .filter((r): r is Record<string, unknown> => r != null);

    return rows.map(row => ({
      error: rowToRecord(row),
      projectName: String(row['project_name']),
      score: scoreMap.get(String(row['id'])) ?? 0,
    }));
  }

  list(projectId?: string, limit = 100): ErrorRecord[] {
    const rows = projectId
      ? this.db.prepare(
          `SELECT * FROM errors WHERE project_id = ? ORDER BY created_at DESC LIMIT ?`,
        ).all(projectId, limit)
      : this.db.prepare(
          `SELECT * FROM errors ORDER BY created_at DESC LIMIT ?`,
        ).all(limit);
    return (rows as unknown as Record<string, unknown>[]).map(rowToRecord);
  }
}
