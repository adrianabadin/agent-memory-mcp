export class ConcurrentUpdateError extends Error {
  constructor(
    public readonly id: string,
    public readonly expectedVersion: number,
    public readonly actualVersion: number | null,
  ) {
    super(
      `Concurrent update conflict on memory ${id}: ` +
      `expected version ${expectedVersion}, ` +
      `actual ${actualVersion == null ? 'deleted by another writer' : `version ${actualVersion}`}`,
    );
    this.name = 'ConcurrentUpdateError';
  }
}

export class MemoryNotFoundError extends Error {
  constructor(public readonly id: string) {
    super(`Memory not found: ${id}`);
    this.name = 'MemoryNotFoundError';
  }
}

export class EmbeddingDimensionMismatchError extends Error {
  constructor(
    public readonly dbPath: string,
    public readonly expectedDimensions: number,
    public readonly actualDimensions: number | null,
  ) {
    super(
      actualDimensions == null
        ? `Embedding dimension mismatch in LanceDB at ${dbPath}: could not determine existing vector column dimensions for expected ${expectedDimensions}. Rebuild the database before using this embedding model.`
        : `Embedding dimension mismatch in LanceDB at ${dbPath}: expected ${expectedDimensions} dimensions but found ${actualDimensions}. Rebuild the database before using this embedding model.`,
    );
    this.name = 'EmbeddingDimensionMismatchError';
  }
}
