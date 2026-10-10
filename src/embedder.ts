import { join } from 'path';
import type {
  ChunkPolicyIdentity,
  Embedder,
  EmbeddingIdentity,
  EmbeddingRole,
} from './types.js';
import { EmbeddingCache } from './embedding-cache.js';

export const DEFAULT_MODEL = 'Xenova/bge-m3';

/** Multilingual E5 small: 384 dims, mean pooling, L2-normalized, role-aware. */
export const E5_MODEL = 'Xenova/multilingual-e5-small';

/**
 * Role prefixes. Applied at INFERENCE ONLY (the stored/logical text never
 * carries the prefix; the embedding cache is keyed on the raw text + role).
 */
export const E5_ROLE_PREFIXES: Readonly<Record<EmbeddingRole, string>> = {
  query: 'query: ',
  passage: 'passage: ',
};

/**
 * Full-text chunking policy for role-aware (E5) embedding. The `version`
 * participates in the model identity (DB guard + cache identity); bump it when
 * chunking or aggregation semantics change so stale vectors are rejected.
 */
export const E5_CHUNK_POLICY: ChunkPolicyIdentity = {
  version: 'e5-fulltext-v1',
  maxTokens: 512,
  overlapTokens: 64,
};

export interface TokenChunkSpan {
  start: number; // inclusive token index
  end: number;   // exclusive token index
}

/**
 * Plan overlapping token chunks covering [0, tokenCount) completely.
 *
 * Contract (tested exactly):
 *   - every span has (end - start) <= budget;
 *   - spans[0].start === 0 and spans[last].end === tokenCount (no truncation);
 *   - consecutive spans overlap by exactly `overlap` tokens;
 *   - every token index is covered at least once (complete coverage);
 *   - `overlap` must be < `budget`, `budget` must be >= 1.
 *
 * @param tokenCount number of content tokens (specials/prefix excluded)
 * @param budget     max content tokens per chunk (already net of prefix+specials)
 * @param overlap    tokens shared between consecutive chunks
 */
export function planTokenChunks(tokenCount: number, budget: number, overlap: number): TokenChunkSpan[] {
  if (!Number.isInteger(tokenCount) || tokenCount < 0) {
    throw new Error(`planTokenChunks: invalid tokenCount ${tokenCount}`);
  }
  if (!Number.isInteger(budget) || budget < 1) {
    throw new Error(`planTokenChunks: invalid budget ${budget}`);
  }
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= budget) {
    throw new Error(`planTokenChunks: overlap must satisfy 0 <= overlap < budget (got ${overlap} vs budget ${budget})`);
  }
  if (tokenCount === 0) return [];
  if (tokenCount <= budget) return [{ start: 0, end: tokenCount }];

  const spans: TokenChunkSpan[] = [];
  let start = 0;
  while (true) {
    const end = Math.min(start + budget, tokenCount);
    spans.push({ start, end });
    if (end === tokenCount) break;
    // Overlap: back up `overlap` tokens from the chunk end. Since overlap <
    // budget this always advances forward by (budget - overlap) > 0 tokens.
    start = end - overlap;
  }
  return spans;
}

/**
 * Sequential, token-count-weighted, L2-normalized aggregation of per-chunk
 * vectors into ONE vector: normalize(sum(w_i * v_i) / sum(w_i)).
 */
export function aggregateChunkVectors(vectors: number[][], weights: number[]): number[] {
  if (vectors.length === 0 || weights.length === 0) {
    throw new Error('aggregateChunkVectors: empty input');
  }
  if (vectors.length !== weights.length) {
    throw new Error('aggregateChunkVectors: vectors/weights length mismatch');
  }
  const dim = vectors[0].length;
  if (dim === 0 || vectors.some(v => v.length !== dim)) {
    throw new Error('aggregateChunkVectors: mismatched or empty vector dimensions');
  }
  const agg = new Array<number>(dim).fill(0);
  let wSum = 0;
  vectors.forEach((v, i) => {
    const w = weights[i];
    wSum += w;
    for (let d = 0; d < dim; d++) agg[d] += w * v[d];
  });
  const mean = agg.map(v => v / wSum);
  const norm = Math.sqrt(mean.reduce((s, v) => s + v * v, 0));
  if (norm === 0) return mean; // degenerate zero vector — keep as-is
  return mean.map(v => v / norm);
}

/** Minimal tokenizer surface required for token-accurate chunking. */
export interface EmbedderTokenizer {
  encode(text: string, opts?: { add_special_tokens?: boolean }): number[];
  decode(ids: number[], opts?: { skip_special_tokens?: boolean }): string;
}

/**
 * Injection points for tests (mock tokenizer/extractor — no network) and for
 * the project model-weight cache (explicit `weightCacheDir`, applied via the
 * SAME ESM transformers module options used by `pipeline()` — never by
 * mutating process-global env/paths).
 */
export interface EmbedderRuntimeOptions {
  tokenizer?: EmbedderTokenizer;
  extractor?: ExtractorPipeline;
  weightCacheDir?: string;
}

/** True for role-aware E5-family models (query:/passage: prefixes + chunking). */
export function isRoleAwareModel(model: string): boolean {
  return model.toLowerCase().includes('e5');
}

export class TransformersEmbedder implements Embedder {
  private extractor: ExtractorPipeline | null = null;
  private tokenizer: EmbedderTokenizer | null = null;
  private readonly modelName: string;
  private readonly embeddingDimensions: number;
  private readonly poolingMode: string;
  private readonly cache: EmbeddingCache | undefined;
  private readonly roleAware: boolean;
  private readonly runtime: EmbedderRuntimeOptions;

  constructor(
    modelName: string = DEFAULT_MODEL,
    dimensions?: number,
    pooling?: string,
    cache?: EmbeddingCache,
    runtime: EmbedderRuntimeOptions = {},
  ) {
    this.modelName = modelName;
    this.embeddingDimensions = dimensions ?? this.inferDimensions(modelName);
    this.poolingMode = pooling ?? this.inferPooling(modelName);
    this.cache = cache;
    this.roleAware = isRoleAwareModel(modelName);
    this.runtime = runtime;
    this.tokenizer = runtime.tokenizer ?? null;
  }

  async initialize(): Promise<void> {
    if (this.runtime.extractor) {
      this.extractor = this.runtime.extractor;
    } else {
      const { pipeline } = await import('@huggingface/transformers');
      const opts = this.runtime.weightCacheDir
        ? { cache_dir: this.runtime.weightCacheDir }
        : undefined;
      this.extractor = await pipeline('feature-extraction', this.modelName, opts) as ExtractorPipeline;
    }
    if (this.roleAware && !this.tokenizer) {
      const { AutoTokenizer } = await import('@huggingface/transformers');
      const opts = this.runtime.weightCacheDir
        ? { cache_dir: this.runtime.weightCacheDir }
        : undefined;
      // PreTrainedTokenizer satisfies the minimal EmbedderTokenizer surface
      // structurally (sync encode/decode).
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelName, opts) as unknown as EmbedderTokenizer;
    }
  }

  async embed(text: string, role: EmbeddingRole = 'passage'): Promise<number[]> {
    const qualifier = this.cacheQualifier(role);
    if (this.cache) {
      const cached = await this.cache.get(text, qualifier);
      if (cached) return cached;
    }
    const result = this.roleAware
      ? await this.embedRoleAware(text, role)
      : await this.runModel(text);
    if (this.cache) {
      await this.cache.set(text, result, qualifier);
    }
    return result;
  }

  async embedBatch(texts: string[], role: EmbeddingRole = 'passage'): Promise<number[][]> {
    if (!this.roleAware) {
      // BGE/single-pass batch path unchanged.
      if (!this.cache) {
        return this.runModelBatch(texts);
      }
      // Per-text cache lookup; batch only the misses
      const results: (number[] | null)[] = await Promise.all(texts.map(t => this.cache!.get(t)));
      const missIndices = results.map((r, i) => r === null ? i : -1).filter(i => i !== -1);

      if (missIndices.length > 0) {
        const missTexts = missIndices.map(i => texts[i]);
        const missVectors = await this.runModelBatch(missTexts);
        for (let k = 0; k < missIndices.length; k++) {
          results[missIndices[k]] = missVectors[k];
          await this.cache!.set(missTexts[k], missVectors[k]);
        }
      }
      return results as number[][];
    }
    // Role-aware: each text goes through the chunked role path sequentially.
    const vectors: number[][] = [];
    for (const text of texts) {
      vectors.push(await this.embed(text, role));
    }
    return vectors;
  }

  /**
   * Stable identity: model + dims + pooling + normalization + role-awareness
   * + chunk-policy version. Part of the DB open-time guard and cache identity.
   */
  identity(): EmbeddingIdentity {
    return {
      model: this.modelName,
      dimensions: this.embeddingDimensions,
      pooling: this.poolingMode,
      normalize: true,
      roleAware: this.roleAware,
      chunkPolicy: this.roleAware ? E5_CHUNK_POLICY : null,
    };
  }

  dimensions(): number {
    return this.embeddingDimensions;
  }

  /** Cache identity: model / role / pooling / chunk policy (never the prefix). */
  private cacheQualifier(role: EmbeddingRole): string {
    const id = this.identity();
    const rolePart = id.roleAware ? role : 'raw';
    return [
      `model=${id.model}`,
      `dims=${id.dimensions}`,
      `pooling=${id.pooling}`,
      `norm=${id.normalize}`,
      `chunk=${id.chunkPolicy?.version ?? 'none'}`,
      `role=${rolePart}`,
    ].join('|');
  }

  /**
   * Token-accurate full-text embedding:
   *   1. tokenize the raw text (no special tokens);
   *   2. plan overlapping chunks so prefixTokens + chunkTokens + specialTokens
   *      <= maxTokens for EVERY chunk (no silent truncation, complete coverage);
   *   3. embed chunks sequentially with the role prefix (inference only);
   *   4. aggregate token-weighted and L2-normalize into one vector.
   */
  private async embedRoleAware(text: string, role: EmbeddingRole): Promise<number[]> {
    const tokenizer = this.requireTokenizer();
    const prefix = E5_ROLE_PREFIXES[role];
    const tokens = tokenizer.encode(text, { add_special_tokens: false });
    if (tokens.length === 0) {
      // Nothing to cover; embed the prefix alone (still a valid vector).
      return this.runModel(prefix);
    }
    const prefixTokens = tokenizer.encode(prefix, { add_special_tokens: false }).length;
    const specialTokens = tokenizer.encode('', { add_special_tokens: true }).length;
    const budget = E5_CHUNK_POLICY.maxTokens - prefixTokens - specialTokens;
    if (budget < 1) {
      throw new Error(
        `E5 chunk budget exhausted: maxTokens=${E5_CHUNK_POLICY.maxTokens} prefix=${prefixTokens} specials=${specialTokens}`,
      );
    }
    const spans = planTokenChunks(tokens.length, budget, E5_CHUNK_POLICY.overlapTokens);
    if (spans.length === 1) {
      return this.runModel(prefix + tokenizer.decode(tokens, { skip_special_tokens: true }));
    }
    const vectors: number[][] = [];
    const weights: number[] = [];
    for (const span of spans) {
      const chunkText = tokenizer.decode(tokens.slice(span.start, span.end), { skip_special_tokens: true });
      vectors.push(await this.runModel(prefix + chunkText));
      weights.push(span.end - span.start);
    }
    return aggregateChunkVectors(vectors, weights);
  }

  private async runModel(text: string): Promise<number[]> {
    const extractor = this.requireExtractor();
    const output = await extractor(text, { pooling: this.poolingMode as 'cls' | 'mean', normalize: true });
    return Array.from(output.data as Float32Array);
  }

  private async runModelBatch(texts: string[]): Promise<number[][]> {
    const extractor = this.requireExtractor();
    const output = await extractor(texts, { pooling: this.poolingMode as 'cls' | 'mean', normalize: true });
    const data = output.data as Float32Array;
    const dim = this.dimensions();
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      vectors.push(Array.from(data.slice(i * dim, (i + 1) * dim)));
    }
    return vectors;
  }

  private requireExtractor(): ExtractorPipeline {
    if (!this.extractor) {
      throw new Error('Embedder not initialised. Call initialize() first.');
    }
    return this.extractor;
  }

  private requireTokenizer(): EmbedderTokenizer {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialised. Call initialize() first.');
    }
    return this.tokenizer;
  }

  private inferDimensions(model: string): number {
    if (model.includes('bge-m3')) return 1024;
    if (model.includes('bge-large')) return 1024;
    if (model.includes('bge-base')) return 768;
    if (model.includes('gte-large')) return 1024;
    if (model.includes('gte-base')) return 768;
    if (model.includes('e5-large')) return 1024;
    if (model.includes('e5-base')) return 768;
    if (model.includes('e5-small')) return 384;
    if (model.includes('MiniLM-L12')) return 384;
    return 384;
  }

  private inferPooling(model: string): string {
    if (model.includes('bge-')) return 'cls';
    return 'mean';
  }
}

interface ExtractorPipeline {
  (text: string | string[], options: { pooling: string; normalize: boolean }): Promise<{
    data: Float32Array;
  }>;
}

// ── Env-configured embedder factories ──────────────────────────────────

/** Vector-cache directory for the PROJECT embedder (EMBEDDING_CACHE_PATH). */
export function projectEmbedderCacheDir(): string | undefined {
  return process.env.EMBEDDING_CACHE_PATH;
}

/**
 * Vector-cache directory for the GLOBAL error-store embedder. Derived under
 * the project cache path so global entries never collide with project entries.
 */
export function globalEmbedderCacheDir(): string | undefined {
  const base = process.env.EMBEDDING_CACHE_PATH;
  return base ? join(base, 'global-bge') : undefined;
}

/**
 * Project embedder from the standard env vars (EMBEDDING_MODEL /
 * EMBEDDING_DIMENSIONS / EMBEDDING_POOLING / EMBEDDING_CACHE_PATH), plus the
 * optional explicit model-WEIGHT cache path (EMBEDDING_WEIGHT_CACHE_PATH),
 * applied as the per-call `cache_dir` option of the same ESM transformers
 * module the pipeline uses — never via process-global side effects.
 */
export function createProjectEmbedder(): TransformersEmbedder {
  const modelName = process.env.EMBEDDING_MODEL ?? DEFAULT_MODEL;
  const dimensions = process.env.EMBEDDING_DIMENSIONS
    ? parseInt(process.env.EMBEDDING_DIMENSIONS, 10)
    : undefined;
  const pooling = process.env.EMBEDDING_POOLING ?? undefined;
  const cacheDir = projectEmbedderCacheDir();
  const cache = cacheDir ? new EmbeddingCache(cacheDir, modelName) : undefined;
  const weightCacheDir = process.env.EMBEDDING_WEIGHT_CACHE_PATH || undefined;
  return new TransformersEmbedder(modelName, dimensions, pooling, cache, { weightCacheDir });
}

/**
 * Global error-store embedder: ALWAYS the BGE-M3 1024/cls default, independent
 * of the project model. The global DB (~/.pmc/global.db) stores 1024-dim BGE
 * vectors; inheriting a project E5 embedder would silently corrupt it.
 */
export function createGlobalEmbedder(): TransformersEmbedder {
  const cacheDir = globalEmbedderCacheDir();
  const cache = cacheDir ? new EmbeddingCache(cacheDir, DEFAULT_MODEL) : undefined;
  // No weight-cache override here: the global scope keeps default behavior.
  return new TransformersEmbedder(DEFAULT_MODEL, undefined, undefined, cache);
}
