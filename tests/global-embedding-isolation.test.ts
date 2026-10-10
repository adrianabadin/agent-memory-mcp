// tests/global-embedding-isolation.test.ts
//
// M1 (E5 small migration): the GLOBAL error store must keep its independent
// BGE-M3 1024 identity — never inheriting the project's E5 384 model — with
// a separate vector cache. Also covers:
//   - query/passage roles in the global error store callsites
//   - LazyEmbedder passthrough of roles + identity
//
// No network: embedders are built from identity factories and mocks only.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  createProjectEmbedder,
  createGlobalEmbedder,
  projectEmbedderCacheDir,
  globalEmbedderCacheDir,
  E5_MODEL,
  DEFAULT_MODEL,
} from '../src/embedder.js';
import { EmbeddingCache } from '../src/embedding-cache.js';
import { LazyEmbedder } from '../src/compose.js';
import { ErrorStore } from '../src/global/error-store.js';
import type { Embedder } from '../src/types.js';

const E5_ENV = {
  EMBEDDING_MODEL: E5_MODEL,
  EMBEDDING_DIMENSIONS: '384',
  EMBEDDING_POOLING: 'mean',
} as const;

describe('project vs global embedder isolation', () => {
  const saved = new Map<string, string | undefined>();
  const setEnv = (key: string, value: string | undefined) => {
    if (!saved.has(key)) saved.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  };
  const clearEnv = (key: string) => setEnv(key, undefined);

  beforeEach(() => {
    saved.clear();
  });
  afterEach(() => {
    for (const [k, v] of saved) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  });

  it('project embedder follows EMBEDDING_MODEL (E5384/mean)', () => {
    for (const [k, v] of Object.entries(E5_ENV)) setEnv(k, v);
    const project = createProjectEmbedder();
    expect(project.identity?.()).toMatchObject({
      model: E5_MODEL,
      dimensions: 384,
      pooling: 'mean',
      roleAware: true,
    });
    expect(project.dimensions()).toBe(384);
  });

  it('global embedder stays BGE-M3 1024/cls regardless of project env', () => {
    for (const [k, v] of Object.entries(E5_ENV)) setEnv(k, v);
    const global = createGlobalEmbedder();
    expect(global.identity?.()).toEqual({
      model: DEFAULT_MODEL,
      dimensions: 1024,
      pooling: 'cls',
      normalize: true,
      roleAware: false,
      chunkPolicy: null,
    });
    expect(global.dimensions()).toBe(1024);
  });

  it('global embedder defaults also apply when no env is set (unchanged behavior)', () => {
    for (const k of Object.keys(E5_ENV)) clearEnv(k);
    clearEnv('EMBEDDING_CACHE_PATH');
    expect(createGlobalEmbedder().identity?.()).toMatchObject({
      model: DEFAULT_MODEL,
      dimensions: 1024,
    });
  });

  it('uses separate vector cache locations for project and global embedders', () => {
    for (const [k, v] of Object.entries(E5_ENV)) setEnv(k, v);
    setEnv('EMBEDDING_CACHE_PATH', 'X:\\pmc-cache\\vec');

    const projectDir = projectEmbedderCacheDir();
    const globalDir = globalEmbedderCacheDir();
    expect(projectDir).toBe('X:\\pmc-cache\\vec');
    expect(globalDir).not.toBe(projectDir);
    expect(globalDir.startsWith(projectDir)).toBe(true); // nested, but distinct

    // No cache configured → no cache for either scope.
    clearEnv('EMBEDDING_CACHE_PATH');
    expect(projectEmbedderCacheDir()).toBeUndefined();
    expect(globalEmbedderCacheDir()).toBeUndefined();
  });

  it('cache keys do not collide between the separate project/global cache dirs', async () => {
    const base = await mkdtemp(join(tmpdir(), 'pmc-iso-'));
    try {
      const projectCache = new EmbeddingCache(join(base, 'project'), DEFAULT_MODEL);
      const globalCache = new EmbeddingCache(join(base, 'global-bge'), DEFAULT_MODEL);
      await projectCache.set('same text', [0.5, 0.5], 'qual');
      expect(await globalCache.get('same text', 'qual')).toBeNull();
      expect(await projectCache.get('same text', 'qual')).toEqual([0.5, 0.5]);
    } finally {
      await rm(base, { recursive: true, force: true });
    }
  });
});

// ── LazyEmbedder passthrough ─────────────────────────────────────────

function capturingEmbedder(
  log: Array<{ text: string; role?: string }>,
  identity?: () => import('../src/types.js').EmbeddingIdentity,
): Embedder {
  return {
    initialize: async () => {},
    embed: async (text: string, role?: 'query' | 'passage') => {
      log.push({ text, role });
      return [0.1, 0.2];
    },
    embedBatch: async (texts: string[]) => texts.map(() => [0.1, 0.2]),
    dimensions: () => 7,
    ...(identity ? { identity } : {}),
  };
}

describe('LazyEmbedder passthrough', () => {
  it('forwards roles to the inner embedder', async () => {
    const log: Array<{ text: string; role?: string }> = [];
    const lazy = new LazyEmbedder(capturingEmbedder(log));
    await lazy.embed('hello', 'query');
    await lazy.embed('world'); // default passage
    expect(log).toEqual([
      { text: 'hello', role: 'query' },
      { text: 'world', role: 'passage' },
    ]);
  });

  it('delegates identity() without initializing the model', () => {
    const stubIdentity = () => ({
      model: E5_MODEL,
      dimensions: 384,
      pooling: 'mean',
      normalize: true,
      roleAware: true,
      chunkPolicy: null,
    });
    const lazy = new LazyEmbedder(capturingEmbedder([], stubIdentity));
    expect(lazy.identity?.()).toEqual(stubIdentity());
    expect(lazy.isInitialized()).toBe(false);
    expect(lazy.dimensions()).toBe(7);
  });
});

// ── Global error store callsites (passage store / query search) ──────

describe('global error store roles', () => {
  it('embeds stored error docs as passage and search queries as query', async () => {
    const log: Array<{ text: string; role?: string }> = [];
    const db = new DatabaseSync(':memory:');
    // ErrorStore references projects(id); create the minimal global schema.
    db.exec('CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL);');
    db.exec(`INSERT INTO projects (id, name) VALUES ('p1', 'proj');`);
    const store = new ErrorStore(db, capturingEmbedder(log));
    store.ensureSchema();

    await store.record({
      projectId: 'p1',
      message: 'boom happened',
      rootCause: 'null pointer',
      fix: 'add a guard',
      source: 'auto',
    });
    expect(log.at(-1)).toEqual({ text: 'boom happened\nnull pointer\nadd a guard', role: 'passage' });

    await store.search('boom');
    expect(log.at(-1)).toEqual({ text: 'boom', role: 'query' });
  });
});
