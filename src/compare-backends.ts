// src/compare-backends.ts
//
// T-010: Core comparison logic for validating migration completeness.
// Opens two SQLite databases (Engram + PMC), runs identical search queries,
// and reports overlap, drift, and semantic similarity.

import { DatabaseSync } from 'node:sqlite';
import { blobToEmbedding, cosineDistance } from './shared/vector-utils.js';

// ── Types ──────────────────────────────────────────────────────────

export interface ComparisonOptions {
  engramDbPath: string;
  pmcDbPath: string;
  queries?: string[];
  windowMinutes?: number | null;
}

export interface TopResult {
  id: string;
  content: string;
  score: number;
}

export interface ComparisonQueryResult {
  query: string;
  engram_top5: TopResult[];
  pmc_top5: TopResult[];
  matched_in_both: number;
}

export interface ComparisonResult {
  total_engram_memories: number;
  total_pmc_memories: number;
  overlap_count: number;
  engram_only_count: number;
  pmc_only_count: number;
  avg_cosine_matched: number;
  results: ComparisonQueryResult[];
}

// ── Constants ──────────────────────────────────────────────────────

const DEFAULT_QUERIES = ['auth', 'config', 'fix', 'decision', 'error'];
const COSINE_MATCH_THRESHOLD = 0.7;
const TOP_K = 5;

// ── Internal helpers ───────────────────────────────────────────────

interface MemoryRow {
  id: string;
  content: string;
  embedding: Buffer | Uint8Array | null;
  created_at: string;
}

function queryEngram(dbPath: string, query: string, windowMinutes: number | null): MemoryRow[] {
  const db = new DatabaseSync(dbPath, { open: false });
  db.open();

  try {
    let sql = `SELECT id, content, embedding, created_at FROM observations WHERE deleted_at IS NULL AND type != 'session_summary'`;
    const params: (string | number)[] = [];

    if (windowMinutes !== null && windowMinutes !== undefined) {
      const cutoff = new Date(Date.now() - windowMinutes * 60_000).toISOString();
      sql += ` AND created_at > ?`;
      params.push(cutoff);
    }

    // FTS5 doesn't exist on Engram observations table — use LIKE for keyword search
    if (query && query.trim()) {
      sql += ` AND (content LIKE ? OR title LIKE ?)`;
      const pattern = `%${query}%`;
      params.push(pattern, pattern);
    }

    sql += ` ORDER BY created_at DESC LIMIT ?`;
    params.push(TOP_K * 3); // Fetch more than needed for ranking

    const rows = db.prepare(sql).all(...params) as Array<{
      id: number;
      content: string;
      embedding: Buffer | null;
      created_at: string;
    }>;

    return rows.map(r => ({
      id: String(r.id),
      content: r.content,
      embedding: r.embedding,
      created_at: r.created_at,
    }));
  } finally {
    db.close();
  }
}

function queryPmc(dbPath: string, query: string, windowMinutes: number | null): MemoryRow[] {
  const db = new DatabaseSync(dbPath, { open: false });
  db.open();

  try {
    let sql = `SELECT m.id, m.content, e.embedding, m.created_at
               FROM memories m
               LEFT JOIN memory_embeddings e ON e.id = m.id
               WHERE m.status = 'active'`;
    const params: (string | number)[] = [];

    if (windowMinutes !== null && windowMinutes !== undefined) {
      const cutoff = new Date(Date.now() - windowMinutes * 60_000).toISOString();
      sql += ` AND m.created_at > ?`;
      params.push(cutoff);
    }

    if (query && query.trim()) {
      sql += ` AND m.content LIKE ?`;
      params.push(`%${query}%`);
    }

    sql += ` ORDER BY m.created_at DESC LIMIT ?`;
    params.push(TOP_K * 3);

    const rows = db.prepare(sql).all(...params) as Array<{
      id: string;
      content: string;
      embedding: Buffer | null;
      created_at: string;
    }>;

    return rows.map(r => ({
      id: r.id,
      content: r.content,
      embedding: r.embedding,
      created_at: r.created_at,
    }));
  } finally {
    db.close();
  }
}

function countMemories(dbPath: string, windowMinutes: number | null, isEngram: boolean): number {
  const db = new DatabaseSync(dbPath, { open: false });
  db.open();

  try {
    let sql: string;
    const params: (string | number)[] = [];

    if (isEngram) {
      sql = `SELECT COUNT(*) as cnt FROM observations WHERE deleted_at IS NULL AND type != 'session_summary'`;
      if (windowMinutes !== null && windowMinutes !== undefined) {
        const cutoff = new Date(Date.now() - windowMinutes * 60_000).toISOString();
        sql += ` AND created_at > ?`;
        params.push(cutoff);
      }
    } else {
      sql = `SELECT COUNT(*) as cnt FROM memories WHERE status = 'active'`;
      if (windowMinutes !== null && windowMinutes !== undefined) {
        const cutoff = new Date(Date.now() - windowMinutes * 60_000).toISOString();
        sql += ` AND created_at > ?`;
        params.push(cutoff);
      }
    }

    const row = db.prepare(sql).get(...params) as { cnt: number };
    return row.cnt;
  } finally {
    db.close();
  }
}

// ── Public API ─────────────────────────────────────────────────────

/**
 * Compare two backends (Engram + PMC) by running identical queries and
 * reporting overlap, drift, and semantic similarity.
 */
export async function compareBackends(options: ComparisonOptions): Promise<ComparisonResult> {
  const {
    engramDbPath,
    pmcDbPath,
    queries = DEFAULT_QUERIES,
    windowMinutes = null,
  } = options;

  const totalEngram = countMemories(engramDbPath, windowMinutes, true);
  const totalPmc = countMemories(pmcDbPath, windowMinutes, false);

  let totalOverlap = 0;
  let totalEngramOnly = 0;
  let totalPmcOnly = 0;
  let totalCosineMatched = 0;
  let matchCount = 0;

  const queryResults: ComparisonQueryResult[] = [];

  for (const query of queries) {
    const engramRows = queryEngram(engramDbPath, query, windowMinutes);
    const pmcRows = queryPmc(pmcDbPath, query, windowMinutes);

    // For ranking, we need query embeddings — use a simple heuristic
    // since we don't have an embedder in the CLI context.
    // Instead, rank by FTS match order (already sorted by relevance).
    const engramTop5: TopResult[] = engramRows.slice(0, TOP_K).map(r => ({
      id: r.id,
      content: r.content.slice(0, 200),
      score: 1.0, // Placeholder — real cosine requires embedder
    }));

    const pmcTop5: TopResult[] = pmcRows.slice(0, TOP_K).map(r => ({
      id: r.id,
      content: r.content.slice(0, 200),
      score: 1.0,
    }));

    // Compare embeddings directly for matched pairs
    const matched = countMatchesDirect(engramRows.slice(0, TOP_K), pmcRows.slice(0, TOP_K));
    totalOverlap += matched.overlap;
    totalEngramOnly += matched.engramOnly;
    totalPmcOnly += matched.pmcOnly;
    totalCosineMatched += matched.avgCosine;
    matchCount += matched.pairs;

    queryResults.push({
      query,
      engram_top5: engramTop5,
      pmc_top5: pmcTop5,
      matched_in_both: matched.overlap,
    });
  }

  return {
    total_engram_memories: totalEngram,
    total_pmc_memories: totalPmc,
    overlap_count: totalOverlap,
    engram_only_count: totalEngramOnly,
    pmc_only_count: totalPmcOnly,
    avg_cosine_matched: matchCount > 0 ? totalCosineMatched / matchCount : 0,
    results: queryResults,
  };
}

/** Direct cosine comparison between engram and PMC rows. */
function countMatchesDirect(
  engramRows: MemoryRow[],
  pmcRows: MemoryRow[],
): { overlap: number; engramOnly: number; pmcOnly: number; avgCosine: number; pairs: number } {
  const matchedPmc = new Set<number>();
  let overlap = 0;
  let totalCosine = 0;
  let pairs = 0;

  for (const eRow of engramRows) {
    if (!eRow.embedding) continue;
    const eVec = blobToEmbedding(eRow.embedding);
    let bestSim = 0;
    let bestIdx = -1;

    for (let i = 0; i < pmcRows.length; i++) {
      if (matchedPmc.has(i)) continue;
      const pRow = pmcRows[i];
      if (!pRow.embedding) continue;
      const pVec = blobToEmbedding(pRow.embedding);
      const sim = cosineDistance(eVec, pVec);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }

    if (bestSim >= COSINE_MATCH_THRESHOLD && bestIdx >= 0) {
      overlap++;
      matchedPmc.add(bestIdx);
      totalCosine += bestSim;
      pairs++;
    }
  }

  const engramOnly = engramRows.length - overlap;
  const pmcOnly = pmcRows.length - matchedPmc.size;

  return { overlap, engramOnly, pmcOnly, avgCosine: totalCosine, pairs };
}
