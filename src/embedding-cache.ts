import { createHash } from 'crypto';
import { readFile, writeFile, rename, mkdir } from 'fs/promises';
import { join } from 'path';

export class EmbeddingCache {
  constructor(
    private readonly cacheDir: string,
    private readonly modelName: string,
    /**
     * Optional identity qualifier (pooling/dimensions/chunk policy/role …).
     * When provided by the embedder it participates in the cache key so
     * vectors computed under different embedding identities never collide.
     */
    private readonly identity?: string,
  ) {}

  private key(text: string, qualifier?: string): string {
    return createHash('sha1')
      .update(this.modelName + '\u0000' + (this.identity ?? '') + '\u0000' + (qualifier ?? '') + '\u0000' + text)
      .digest('hex');
  }

  private filePath(text: string, qualifier?: string): string {
    return join(this.cacheDir, `${this.key(text, qualifier)}.bin`);
  }

  async get(text: string, qualifier?: string): Promise<number[] | null> {
    try {
      const buf = await readFile(this.filePath(text, qualifier));
      // Stored as little-endian float32 bytes
      const floats = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      return Array.from(floats);
    } catch {
      return null;
    }
  }

  async set(text: string, vector: number[], qualifier?: string): Promise<void> {
    try {
      await mkdir(this.cacheDir, { recursive: true });
      const finalPath = this.filePath(text, qualifier);
      const tmpPath = `${finalPath}.tmp`;
      const floats = new Float32Array(vector);
      await writeFile(tmpPath, Buffer.from(floats.buffer));
      await rename(tmpPath, finalPath);
    } catch {
      // Best-effort cache write — never propagate
    }
  }
}
