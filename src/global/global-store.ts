import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import type { Embedder, MergeResult } from '../types.js';
import { ProjectRegistry, type ProjectRecord, type ProjectMetadata } from './project-registry.js';
import { ErrorStore, type ErrorRecord, type RecordErrorRequest, type ErrorSearchResult } from './error-store.js';

export type { ProjectRecord, ProjectMetadata, ErrorRecord, RecordErrorRequest, ErrorSearchResult };

export interface GlobalStore {
  registerProject(metadata: ProjectMetadata): ProjectRecord;
  syncMetadata(projectId: string, metadata: Partial<Omit<ProjectMetadata, 'name' | 'rootPath'>>): ProjectRecord;
  recordError(req: RecordErrorRequest): Promise<ErrorRecord>;
  searchErrors(query: string, projectId?: string, limit?: number): Promise<ErrorSearchResult[]>;
  mergeProjects(sourceId: string, targetId: string, dryRun?: boolean): Promise<MergeResult>;
  close(): void;
}

export class SqliteGlobalStore implements GlobalStore {
  private db!: DatabaseSync;
  private registry!: ProjectRegistry;
  private errors!: ErrorStore;

  constructor(
    private readonly dbPath: string,
    private readonly embedder: Embedder,
  ) {}

  async initialize(): Promise<void> {
    mkdirSync(dirname(this.dbPath), { recursive: true });
    this.db = new DatabaseSync(this.dbPath);
    this.db.exec('PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;');
    this.registry = new ProjectRegistry(this.db);
    this.registry.ensureSchema();
    this.errors = new ErrorStore(this.db, this.embedder);
    this.errors.ensureSchema();
  }

  registerProject(metadata: ProjectMetadata): ProjectRecord {
    return this.registry.register(metadata);
  }

  syncMetadata(
    projectId: string,
    metadata: Partial<Omit<ProjectMetadata, 'name' | 'rootPath'>>,
  ): ProjectRecord {
    return this.registry.updateMetadata(projectId, metadata);
  }

  async recordError(req: RecordErrorRequest): Promise<ErrorRecord> {
    return this.errors.record(req);
  }

  async searchErrors(query: string, projectId?: string, limit = 10): Promise<ErrorSearchResult[]> {
    return this.errors.search(query, projectId, limit);
  }

  // Stub: full implementation lands in a later PR (see tasks.md T-008).
  async mergeProjects(sourceId: string, targetId: string, dryRun = true): Promise<MergeResult> {
    throw new Error(
      `mergeProjects not implemented in PR-1 (source=${sourceId}, target=${targetId}, dryRun=${dryRun})`,
    );
  }

  close(): void {
    try { this.db?.close(); } catch { /* ignore */ }
  }
}
