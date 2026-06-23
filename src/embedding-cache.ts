import { createHash } from 'crypto';
import { readFile, writeFile, rename, mkdir } from 'fs/promises';
import { join } from 'path';

export class EmbeddingCache {
  constructor(
    private readonly cacheDir: string,
    private readonly modelName: string,
  ) {}

  private key(text: string): string {
    return createHash('sha1')
      .update(this.modelName + ':' + text)
      .digest('hex');
  }

  private filePath(text: string): string {
    return join(this.cacheDir, `${this.key(text)}.bin`);
  }

  async get(text: string): Promise<number[] | null> {
    try {
      const buf = await readFile(this.filePath(text));
      // Stored as little-endian float32 bytes
      const floats = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      return Array.from(floats);
    } catch {
      return null;
    }
  }

  async set(text: string, vector: number[]): Promise<void> {
    try {
      await mkdir(this.cacheDir, { recursive: true });
      const finalPath = this.filePath(text);
      const tmpPath = `${finalPath}.tmp`;
      const floats = new Float32Array(vector);
      await writeFile(tmpPath, Buffer.from(floats.buffer));
      await rename(tmpPath, finalPath);
    } catch {
      // Best-effort cache write — never propagate
    }
  }
}
