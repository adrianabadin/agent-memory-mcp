import { createHash } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

export interface ProjectRecord {
  id: string;
  name: string;
  rootPath: string;
  objective: string | null;
  stack: string | null;
  architecture: string | null;
  dependencies: string | null;
  minimap: string | null;
  createdAt: string;
  lastSyncedAt: string;
}

export interface ProjectMetadata {
  name: string;
  rootPath: string;
  objective?: string;
  stack?: unknown;
  architecture?: string;
  dependencies?: unknown;
  minimap?: unknown;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS projects (
  id              TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  root_path       TEXT NOT NULL,
  objective       TEXT,
  stack           TEXT,
  architecture    TEXT,
  dependencies    TEXT,
  minimap         TEXT,
  created_at      TEXT NOT NULL,
  last_synced_at  TEXT NOT NULL
);
`;

function rowToRecord(row: Record<string, unknown>): ProjectRecord {
  return {
    id: String(row['id']),
    name: String(row['name']),
    rootPath: String(row['root_path']),
    objective: row['objective'] != null ? String(row['objective']) : null,
    stack: row['stack'] != null ? String(row['stack']) : null,
    architecture: row['architecture'] != null ? String(row['architecture']) : null,
    dependencies: row['dependencies'] != null ? String(row['dependencies']) : null,
    minimap: row['minimap'] != null ? String(row['minimap']) : null,
    createdAt: String(row['created_at']),
    lastSyncedAt: String(row['last_synced_at']),
  };
}

export class ProjectRegistry {
  constructor(private readonly db: DatabaseSync) {}

  static computeId(rootPath: string): string {
    const normalised = rootPath.replace(/\\/g, '/').toLowerCase();
    return createHash('sha256').update(normalised).digest('hex');
  }

  ensureSchema(): void {
    this.db.exec(SCHEMA);
  }

  register(metadata: ProjectMetadata): ProjectRecord {
    const id = ProjectRegistry.computeId(metadata.rootPath);
    const now = new Date().toISOString();

    this.db.prepare(`
      INSERT INTO projects (id, name, root_path, objective, stack, architecture, dependencies, minimap, created_at, last_synced_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        name           = excluded.name,
        objective      = COALESCE(excluded.objective, projects.objective),
        stack          = COALESCE(excluded.stack, projects.stack),
        architecture   = COALESCE(excluded.architecture, projects.architecture),
        dependencies   = COALESCE(excluded.dependencies, projects.dependencies),
        minimap        = COALESCE(excluded.minimap, projects.minimap),
        last_synced_at = excluded.last_synced_at
    `).run(
      id,
      metadata.name,
      metadata.rootPath,
      metadata.objective ?? null,
      metadata.stack != null ? JSON.stringify(metadata.stack) : null,
      metadata.architecture ?? null,
      metadata.dependencies != null ? JSON.stringify(metadata.dependencies) : null,
      metadata.minimap != null ? JSON.stringify(metadata.minimap) : null,
      now,
      now,
    );

    return this.get(id)!;
  }

  updateMetadata(id: string, metadata: Partial<Omit<ProjectMetadata, 'name' | 'rootPath'>>): ProjectRecord {
    const now = new Date().toISOString();
    const fields: [string, unknown][] = [];

    if (metadata.objective !== undefined) fields.push(['objective', metadata.objective]);
    if (metadata.stack !== undefined) fields.push(['stack', JSON.stringify(metadata.stack)]);
    if (metadata.architecture !== undefined) fields.push(['architecture', metadata.architecture]);
    if (metadata.dependencies !== undefined) fields.push(['dependencies', JSON.stringify(metadata.dependencies)]);
    if (metadata.minimap !== undefined) fields.push(['minimap', JSON.stringify(metadata.minimap)]);
    fields.push(['last_synced_at', now]);

    const setClause = fields.map(([col]) => `${col} = ?`).join(', ');
    const values = fields.map(([, v]) => v);

    this.db.prepare(`UPDATE projects SET ${setClause} WHERE id = ?`).run(...(values as string[]), id);
    return this.get(id)!;
  }

  get(id: string): ProjectRecord | null {
    const row = this.db.prepare(`SELECT * FROM projects WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? rowToRecord(row) : null;
  }

  list(): ProjectRecord[] {
    const rows = this.db.prepare(`SELECT * FROM projects ORDER BY last_synced_at DESC`).all() as unknown as Record<string, unknown>[];
    return rows.map(rowToRecord);
  }
}
