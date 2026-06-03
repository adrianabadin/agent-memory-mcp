import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';
import type {
  Embedder,
  Memory,
  MemoryCategory,
  MemoryStats,
  MemoryStore,
  PruneOptions,
  PruneResult,
  SearchFilters,
  SearchMode,
  SearchResult,
  StoreRequest,
  UpdateRequest,
} from './types.js';
import { ConcurrentUpdateError, MemoryNotFoundError } from './errors.js';
import {
  computeDecayFactor,
  parseDecayHalfLife,
  importanceMultiplier,
  EVERGREEN_TAGS,
} from './memory-store.js';

// ── Schema ─────────────────────────────────────────────────────────

const SCHEMA_SQL = `
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  category TEXT NOT NULL,
  tags TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  id UNINDEXED,
  content,
  tags,
  tokenize='unicode61',
  content='memories',
  content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
  id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, id, content, tags) VALUES ('delete', old.rowid, old.id, old.content, old.tags);
  INSERT INTO memories_fts(rowid, id, content, tags) VALUES (new.rowid, new.id, new.content, new.tags);
END;
`;

// ── Row type (internal) ───────────────────────────────────────────

interface MemoryRow {
  id: string;
  content: string;
  category: string;
  tags: string;
  created_at: string;
  updated_at: string;
  access_count: number;
  last_accessed_at: string;
  version: number;
}

// ── Utilities ─────────────────────────────────────────────────────

function embeddingToBlob(vec: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vec.length * 4);
  const view = new Float32Array(buf.buffer, buf.byteOffset, vec.length);
  view.set(vec);
  return buf;
}

function blobToEmbedding(blob: Buffer | Uint8Array): number[] {
  const buffer = Buffer.isBuffer(blob) ? blob : Buffer.from(blob);
  const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  return Array.from(view);
}

function cosineDistance(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return magA && magB ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
}

function rrf(
  lists: Array<Array<{ id: string; score: number }>>,
  k = 60,
): Array<{ id: string; score: number }> {
  const scores = new Map<string, number>();
  for (const list of lists) {
    list.forEach(({ id }, rank) => {
      scores.set(id, (scores.get(id) ?? 0) + 1 / (k + rank + 1));
    });
  }
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}

function rowToMemory(row: MemoryRow): Memory {
  return {
    id: row.id,
    content: row.content,
    category: row.category as MemoryCategory,
    tags: JSON.parse(row.tags),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    version: row.version ?? 1,
  };
}

function isEvergreen(row: MemoryRow): boolean {
  try {
    const tags: string[] = JSON.parse(row.tags);
    return tags.some(t => EVERGREEN_TAGS.has(t));
  } catch {
    return false;
  }
}

function escapeFtsQuery(query: string): string {
  // Quote each token independently so multi-word queries match any token (any order)
  return query
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map(t => `"${t.replace(/"/g, '""')}"`)
    .join(' ');
}

const DECAY_HALF_LIFE_DAYS = parseDecayHalfLife(process.env.MEMORY_DECAY_HALF_LIFE);
const MS_PER_DAY = 86_400_000;

function applyDecay(row: MemoryRow, rawScore: number): number {
  if (DECAY_HALF_LIFE_DAYS <= 0 || isEvergreen(row)) return rawScore;
  const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
  return rawScore * computeDecayFactor(row.updated_at, effectiveHalfLife);
}

// ── SqliteMemoryStore ─────────────────────────────────────────────

export class SqliteMemoryStore implements MemoryStore {
  private db!: DatabaseSync;

  constructor(
    private readonly dbPath: string,
    private readonly embedder: Embedder,
  ) {}

  async initialize(): Promise<void> {
    this.db = new DatabaseSync(this.dbPath);
    this.db.exec(SCHEMA_SQL);
  }

  close(): void {
    this.db.close();
  }

  // ── Storage ──────────────────────────────────────────────────────

  async store(request: StoreRequest & { id?: string }): Promise<Memory> {
    const now = new Date().toISOString();
    const id = (request as any).id ?? randomUUID();
    const tags = JSON.stringify(request.tags ?? []);
    const embedding = await this.embedder.embed(request.content);

    this.db.exec('BEGIN');
    try {
      this.db.prepare(
        `INSERT INTO memories
          (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           content        = excluded.content,
           category       = excluded.category,
           tags           = excluded.tags,
           updated_at     = excluded.updated_at`,
      ).run(id, request.content, request.category, tags, now, now, 0, now, 1);

      this.db.prepare(
        `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
      ).run(id, embeddingToBlob(embedding));

      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }

    return {
      id,
      content: request.content,
      category: request.category,
      tags: request.tags ?? [],
      createdAt: now,
      updatedAt: now,
      version: 1,
    };
  }

  async storeBatch(memories: StoreRequest[]): Promise<Memory[]> {
    if (memories.length === 0) return [];

    // 1. Embed all memories first (async, before transaction)
    const embedded = await Promise.all(
      memories.map(async m => ({ m, vec: await this.embedder.embed(m.content) }))
    );

    // 2. Write all in one transaction (synchronous db calls)
    const now = new Date().toISOString();
    const results: Memory[] = [];

    this.db.exec('BEGIN');
    try {
      for (const { m, vec } of embedded) {
        const id = (m as any).id ?? randomUUID();
        const tags = JSON.stringify(m.tags ?? []);
        const createdAt = (m as any).created_at ?? (m as any).createdAt ?? now;
        const updatedAt = (m as any).updated_at ?? (m as any).updatedAt ?? now;
        const accessCount = (m as any).access_count ?? 0;
        const lastAccessedAt = (m as any).last_accessed_at ?? now;
        const version = (m as any).version ?? 1;

        this.db.prepare(
          `INSERT INTO memories
            (id, content, category, tags, created_at, updated_at, access_count, last_accessed_at, version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             content        = excluded.content,
             category       = excluded.category,
             tags           = excluded.tags,
             updated_at     = excluded.updated_at`,
        ).run(id, m.content, m.category, tags, createdAt, updatedAt, accessCount, lastAccessedAt, version);

        this.db.prepare(
          `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
        ).run(id, embeddingToBlob(vec));

        results.push({
          id,
          content: m.content,
          category: m.category,
          tags: m.tags ?? [],
          createdAt,
          updatedAt,
          version,
        });
      }
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }

    return results;
  }

  // ── Access tracking ──────────────────────────────────────────────

  touchAccessed(id: string): void {
    // best-effort: no error if id doesn't exist
    this.db.prepare(
      'UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?',
    ).run(new Date().toISOString(), id);
  }

  // ── Search ────────────────────────────────────────────────────────

  async search(
    query: string,
    mode: SearchMode = 'hybrid',
    filters: SearchFilters = {},
  ): Promise<SearchResult[]> {
    const limit = filters.limit ?? 10;

    switch (mode) {
      case 'semantic':
        return this.semanticSearch(query, filters, limit);
      case 'keyword':
        return this.keywordSearch(query, filters, limit);
      case 'hybrid':
        return this.hybridSearch(query, filters, limit);
    }
  }

  private keywordSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
    trackAccess: boolean = true,
  ): SearchResult[] {
    if (!query.trim()) return [];

    try {
      const escaped = escapeFtsQuery(query);
      // bm25() returns negative values: lower (more negative) = better match
      // We negate to get positive score where higher = better
      const rows = this.db.prepare(`
        SELECT m.*, -bm25(memories_fts) AS fts_score
        FROM memories_fts
        JOIN memories m ON m.id = memories_fts.id
        WHERE memories_fts MATCH ?
        ORDER BY bm25(memories_fts)
        LIMIT ?
      `).all(escaped, limit * 3) as (MemoryRow & { fts_score: number })[];

      return this.applyFiltersAndDecay(rows, filters, (r) => r.fts_score ?? 1, limit, trackAccess);
    } catch {
      return [];
    }
  }

  private async semanticSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
    trackAccess: boolean = true,
  ): Promise<SearchResult[]> {
    const queryVec = await this.embedder.embed(query);
    return this.vectorSearch(queryVec, filters, limit, null, trackAccess);
  }

  private vectorSearch(
    queryVec: number[],
    filters: SearchFilters,
    limit: number,
    excludeId: string | null,
    trackAccess: boolean = true,
  ): SearchResult[] {
    // Fetch all embeddings + row data and compute cosine in-process
    const rows = this.db.prepare(
      `SELECT m.*, e.embedding
       FROM memories m
       JOIN memory_embeddings e ON e.id = m.id`,
    ).all() as (MemoryRow & { embedding: Buffer | Uint8Array })[];

    const scored = rows
      .filter(r => r.id !== excludeId)
      .map(r => ({
        row: r as MemoryRow,
        rawScore: cosineDistance(queryVec, blobToEmbedding(r.embedding)),
      }))
      .sort((a, b) => b.rawScore - a.rawScore);

    // Build scoreMap once (O(N)) before calling applyFiltersAndDecay
    const scoreMap = new Map(scored.map(s => [s.row.id, s.rawScore]));
    return this.applyFiltersAndDecay(
      scored.map(s => s.row),
      filters,
      (r) => scoreMap.get(r.id) ?? 0,
      limit,
      trackAccess,
    );
  }

  private async hybridSearch(
    query: string,
    filters: SearchFilters,
    limit: number,
  ): Promise<SearchResult[]> {
    // Pass trackAccess=false to avoid double-incrementing access_count
    const [keywordResults, semanticResults] = await Promise.all([
      Promise.resolve(this.keywordSearch(query, filters, limit * 2, false)),
      this.semanticSearch(query, filters, limit * 2, false),
    ]);

    if (keywordResults.length === 0 && semanticResults.length === 0) return [];

    // Build merged result set via RRF
    const kwList = keywordResults.map(r => ({ id: r.memory.id, score: r.score }));
    const semList = semanticResults.map(r => ({ id: r.memory.id, score: r.score }));

    const merged = rrf([kwList, semList]);
    const idToMemory = new Map<string, Memory>();
    for (const r of [...keywordResults, ...semanticResults]) {
      idToMemory.set(r.memory.id, r.memory);
    }

    const finalResults = merged
      .slice(0, limit)
      .filter(({ id }) => idToMemory.has(id))
      .map(({ id, score }) => ({ memory: idToMemory.get(id)!, score }));

    // Touch access exactly once per final result
    for (const { memory } of finalResults) {
      this.touchAccessed(memory.id);
    }

    // Fallback: if one list was empty, fall back to the other (already tracked above)
    if (finalResults.length === 0) {
      const fallback = keywordResults.length > 0 ? keywordResults : semanticResults;
      for (const { memory } of fallback.slice(0, limit)) {
        this.touchAccessed(memory.id);
      }
      return fallback.slice(0, limit);
    }

    return finalResults;
  }

  private applyFiltersAndDecay(
    rows: MemoryRow[],
    filters: SearchFilters,
    scoreFor: (r: MemoryRow) => number,
    limit: number,
    trackAccess: boolean = true,
  ): SearchResult[] {
    let filtered = rows;

    if (filters.category) {
      filtered = filtered.filter(r => r.category === filters.category);
    }
    if (filters.after) {
      filtered = filtered.filter(r => r.created_at >= filters.after!);
    }
    if (filters.before) {
      filtered = filtered.filter(r => r.created_at <= filters.before!);
    }
    if (filters.tags && filters.tags.length > 0) {
      filtered = filtered.filter(r => {
        const tags: string[] = JSON.parse(r.tags);
        return filters.tags!.some(t => tags.includes(t));
      });
    }

    const results = filtered
      .map(r => ({
        memory: rowToMemory(r),
        score: applyDecay(r, scoreFor(r)),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    // Track access (best-effort); skip when caller handles it (e.g. hybridSearch)
    if (trackAccess) {
      for (const { memory } of results) {
        this.touchAccessed(memory.id);
      }
    }

    return results;
  }

  // ── findRelated() ─────────────────────────────────────────────────

  async findRelated(memoryId: string, limit: number = 5): Promise<SearchResult[]> {
    const embRow = this.db.prepare(
      `SELECT embedding FROM memory_embeddings WHERE id = ?`,
    ).get(memoryId) as { embedding: Buffer | Uint8Array } | undefined;

    if (!embRow) throw new MemoryNotFoundError(memoryId);

    const queryVec = blobToEmbedding(embRow.embedding);
    return this.vectorSearch(queryVec, {}, limit, memoryId);
  }

  // ── listRecent() ──────────────────────────────────────────────────

  async listRecent(limit: number = 10, category?: MemoryCategory): Promise<Memory[]> {
    let rows: MemoryRow[];
    if (category) {
      rows = this.db.prepare(
        `SELECT * FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?`,
      ).all(category, limit) as MemoryRow[];
    } else {
      rows = this.db.prepare(
        `SELECT * FROM memories ORDER BY created_at DESC LIMIT ?`,
      ).all(limit) as MemoryRow[];
    }
    return rows.map(rowToMemory);
  }

  // ── update() with OCC ─────────────────────────────────────────────

  /**
   * Update a memory by id. Reads the current version and uses it as the
   * expected version for the OCC check. Use `updateWithVersion` to provide
   * an explicit expected version for testing or true concurrent update scenarios.
   */
  async update(id: string, updates: UpdateRequest): Promise<Memory> {
    const existing = this.db.prepare(
      'SELECT version FROM memories WHERE id = ?',
    ).get(id) as { version: number } | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    return this._updateWithVersion(id, updates, existing.version);
  }

  /**
   * Update with an explicit expected version for OCC. Throws ConcurrentUpdateError
   * if the current version does not match expectedVersion.
   */
  private async _updateWithVersion(
    id: string,
    updates: UpdateRequest,
    expectedVersion: number,
  ): Promise<Memory> {
    const existing = this.db.prepare(
      `SELECT * FROM memories WHERE id = ?`,
    ).get(id) as MemoryRow | undefined;

    if (!existing) throw new MemoryNotFoundError(id);

    const now = new Date().toISOString();
    const newContent = updates.content ?? existing.content;
    const newCategory = updates.category ?? existing.category;
    const newTags = updates.tags ? JSON.stringify(updates.tags) : existing.tags;

    const result = this.db.prepare(
      `UPDATE memories
       SET content = ?, category = ?, tags = ?, updated_at = ?, version = version + 1
       WHERE id = ? AND version = ?`,
    ).run(newContent, newCategory, newTags, now, id, expectedVersion) as { changes: number };

    if (result.changes === 0) {
      // Row exists but version doesn't match — concurrent writer updated it
      const current = this.db.prepare(
        `SELECT version FROM memories WHERE id = ?`,
      ).get(id) as { version: number } | undefined;

      if (!current) throw new MemoryNotFoundError(id);
      throw new ConcurrentUpdateError(id, expectedVersion, current.version);
    }

    // Re-embed if content changed (use !== undefined to handle empty string content)
    if (updates.content !== undefined) {
      const embedding = await this.embedder.embed(newContent);
      this.db.prepare(
        `INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)`,
      ).run(id, embeddingToBlob(embedding));
    }

    const updated = this.db.prepare(
      `SELECT * FROM memories WHERE id = ?`,
    ).get(id) as MemoryRow;

    return rowToMemory(updated);
  }

  // ── delete() ─────────────────────────────────────────────────────

  async delete(id: string): Promise<void> {
    const result = this.db.prepare(
      `DELETE FROM memories WHERE id = ?`,
    ).run(id) as { changes: number };

    if (result.changes === 0) throw new MemoryNotFoundError(id);
  }

  // ── stats() ──────────────────────────────────────────────────────

  async stats(): Promise<MemoryStats> {
    const rows = this.db.prepare(
      `SELECT * FROM memories`,
    ).all() as MemoryRow[];

    if (rows.length === 0) {
      return {
        totalMemories: 0,
        byCategory: {},
        oldestMemory: null,
        newestMemory: null,
        neverAccessed: 0,
        belowPruneThreshold: 0,
        avgAccessCount: 0,
        mostAccessed: [],
      };
    }

    const byCategory: Record<string, number> = {};
    let neverAccessed = 0;
    let belowPruneThreshold = 0;
    let totalAccessCount = 0;
    const accessCounts: { id: string; content: string; count: number }[] = [];

    for (const row of rows) {
      byCategory[row.category] = (byCategory[row.category] ?? 0) + 1;

      const ac = row.access_count ?? 0;
      totalAccessCount += ac;
      if (ac === 0) neverAccessed++;

      accessCounts.push({
        id: row.id,
        content: row.content.slice(0, 80),
        count: ac,
      });

      if (!isEvergreen(row)) {
        const effHL = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
        const ageDays = Math.max(
          0,
          (Date.now() - new Date(row.updated_at).getTime()) / MS_PER_DAY,
        );
        const strength = effHL > 0 ? Math.pow(0.5, ageDays / effHL) : 0;
        if (strength < 0.05 || (ac === 0 && ageDays > 90)) {
          belowPruneThreshold++;
        }
      }
    }

    const sorted = [...rows].sort((a, b) => a.created_at.localeCompare(b.created_at));
    accessCounts.sort((a, b) => b.count - a.count);

    return {
      totalMemories: rows.length,
      byCategory,
      oldestMemory: sorted[0]?.created_at ?? null,
      newestMemory: sorted.at(-1)?.created_at ?? null,
      neverAccessed,
      belowPruneThreshold,
      avgAccessCount: rows.length > 0
        ? Math.round((totalAccessCount / rows.length) * 10) / 10
        : 0,
      mostAccessed: accessCounts.slice(0, 5),
    };
  }

  // ── prune() ──────────────────────────────────────────────────────

  async prune(options: PruneOptions = {}): Promise<PruneResult> {
    const { dryRun = true, minStrength = 0.05, maxDormantDays = 90 } = options;

    const rows = this.db.prepare(`SELECT * FROM memories`).all() as MemoryRow[];
    const candidates: PruneResult['candidates'] = [];

    for (const row of rows) {
      if (isEvergreen(row)) continue;

      const accessCount = row.access_count ?? 0;
      const effectiveHalfLife = DECAY_HALF_LIFE_DAYS * importanceMultiplier(row as unknown as Record<string, unknown>);
      const ageMs = Date.now() - new Date(row.updated_at).getTime();
      const ageDays = Math.max(0, ageMs / MS_PER_DAY);
      const strength = effectiveHalfLife > 0
        ? Math.pow(0.5, ageDays / effectiveHalfLife)
        : 0;

      let reason = '';
      if (strength < minStrength) {
        reason = `strength ${strength.toFixed(4)} < ${minStrength}`;
      } else if (accessCount === 0 && ageDays > maxDormantDays) {
        reason = `never accessed, ${Math.floor(ageDays)} days old > ${maxDormantDays}`;
      }

      if (reason) {
        candidates.push({
          id: row.id,
          content: row.content.slice(0, 120),
          strength,
          reason,
        });
      }
    }

    let pruned = 0;
    if (!dryRun && candidates.length > 0) {
      for (const c of candidates) {
        await this.delete(c.id);
        pruned++;
      }
    }

    return {
      pruned,
      inspected: rows.length,
      dryRun,
      candidates,
    };
  }
}
