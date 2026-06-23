import type { Embedder } from './types.js';
import type { EmbeddingCache } from './embedding-cache.js';

const DEFAULT_MODEL = 'Xenova/bge-m3';

export class TransformersEmbedder implements Embedder {
  private extractor: ExtractorPipeline | null = null;
  private readonly modelName: string;
  private readonly embeddingDimensions: number;
  private readonly poolingMode: string;
  private readonly cache: EmbeddingCache | undefined;

  constructor(
    modelName: string = DEFAULT_MODEL,
    dimensions?: number,
    pooling?: string,
    cache?: EmbeddingCache,
  ) {
    this.modelName = modelName;
    this.embeddingDimensions = dimensions ?? this.inferDimensions(modelName);
    this.poolingMode = pooling ?? this.inferPooling(modelName);
    this.cache = cache;
  }

  async initialize(): Promise<void> {
    const { pipeline } = await import('@huggingface/transformers');
    this.extractor = await pipeline('feature-extraction', this.modelName) as ExtractorPipeline;
  }

  async embed(text: string): Promise<number[]> {
    if (this.cache) {
      const cached = await this.cache.get(text);
      if (cached) return cached;
    }
    const result = await this.runModel(text);
    if (this.cache) {
      await this.cache.set(text, result);
    }
    return result;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
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

  dimensions(): number {
    return this.embeddingDimensions;
  }

  private requireExtractor(): ExtractorPipeline {
    if (!this.extractor) {
      throw new Error('Embedder not initialised. Call initialize() first.');
    }
    return this.extractor;
  }

  private inferDimensions(model: string): number {
    if (model.includes('bge-m3')) return 1024;
    if (model.includes('bge-large')) return 1024;
    if (model.includes('bge-base')) return 768;
    if (model.includes('gte-large')) return 1024;
    if (model.includes('gte-base')) return 768;
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
