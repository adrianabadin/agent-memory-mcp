import { describe, it, expect } from 'vitest';
import path from 'node:path';
import { handleGetMemoryProtocol, type MemoryProtocolDeps } from '../src/tools.js';

// ── Helpers ────────────────────────────────────────────────────────

function parseResult(result: { content: Array<{ type: string; text: string }>; isError?: boolean }) {
  return JSON.parse(result.content[0].text);
}

const DB_PATH = '/tmp/proj/.planning/project-memory-context/memory.db';
const EXPECTED_FLAG_PATH = path.resolve(
  '/tmp/proj',
  '.planning',
  'project-memory-context',
  'memory-protocol.json',
);

/**
 * Build injectable deps. `flagJson` controls what readFile returns:
 *   - undefined  -> readFile rejects (file missing)
 *   - string     -> readFile resolves with that raw text
 *   - object     -> readFile resolves with JSON.stringify(object)
 */
function makeDeps(opts: {
  flagJson?: string | Record<string, unknown>;
  dbExists?: boolean;
  dbPath?: string | undefined;
} = {}): MemoryProtocolDeps {
  const { flagJson, dbExists = true, dbPath = DB_PATH } = opts;

  return {
    readFile: (async () => {
      if (flagJson === undefined) {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      }
      return typeof flagJson === 'string' ? flagJson : JSON.stringify(flagJson);
    }) as MemoryProtocolDeps['readFile'],
    existsSync: ((p: string) => {
      if (p === dbPath) return dbExists;
      return false;
    }) as MemoryProtocolDeps['existsSync'],
    env: dbPath === undefined ? {} : { MEMORY_DB_PATH: dbPath },
  };
}

// ── Tests ──────────────────────────────────────────────────────────

describe('get_memory_protocol', () => {
  it('returns dual/default when no flag file exists', async () => {
    const handler = handleGetMemoryProtocol(makeDeps({ flagJson: undefined }));
    const result = await handler();

    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.version).toBe('dual');
    expect(data.source).toBe('default');
    expect(data.warning).toBeUndefined();
  });

  it('returns dual/flag when file holds a valid "dual" value', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'dual' } }),
    );
    const data = parseResult(await handler());
    expect(data.version).toBe('dual');
    expect(data.source).toBe('flag');
    expect(data.warning).toBeUndefined();
  });

  it('returns pmc-only/flag when file holds "pmc-only"', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'pmc-only' } }),
    );
    const data = parseResult(await handler());
    expect(data.version).toBe('pmc-only');
    expect(data.source).toBe('flag');
  });

  it('falls back to default with warning on invalid value', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'garbage' } }),
    );
    const data = parseResult(await handler());
    expect(data.version).toBe('dual');
    expect(data.source).toBe('default');
    expect(data.warning).toMatch(/Invalid flag value "garbage"/);
  });

  it('falls back to default with warning on malformed JSON', async () => {
    const handler = handleGetMemoryProtocol(makeDeps({ flagJson: '{ not json' }));
    const data = parseResult(await handler());
    expect(data.version).toBe('dual');
    expect(data.source).toBe('default');
    expect(data.warning).toMatch(/not valid JSON/);
  });

  it('reports pmcAvailable: false when MEMORY_DB_PATH does not exist', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'dual' }, dbExists: false }),
    );
    const data = parseResult(await handler());
    expect(data.pmcAvailable).toBe(false);
  });

  it('reports pmcAvailable: true when MEMORY_DB_PATH exists', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'dual' }, dbExists: true }),
    );
    const data = parseResult(await handler());
    expect(data.pmcAvailable).toBe(true);
    expect(data.pmcDbPath).toBe(DB_PATH);
  });

  it('infers project root and flag path from MEMORY_DB_PATH', async () => {
    const handler = handleGetMemoryProtocol(
      makeDeps({ flagJson: { memory_protocol_version: 'dual' } }),
    );
    const data = parseResult(await handler());
    // /tmp/proj/.planning/project-memory-context/memory.db -> /tmp/proj root
    expect(data.flagPath).toBe(EXPECTED_FLAG_PATH);
  });

  it('never throws and defaults safely when MEMORY_DB_PATH is unset', async () => {
    // Empty env: no MEMORY_DB_PATH. readFile always rejects (no flag file).
    const handler = handleGetMemoryProtocol({
      readFile: (async () => {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      }) as MemoryProtocolDeps['readFile'],
      existsSync: (() => false) as MemoryProtocolDeps['existsSync'],
      env: {},
    });
    const result = await handler();
    expect(result.isError).toBeUndefined();
    const data = parseResult(result);
    expect(data.version).toBe('dual');
    expect(data.source).toBe('default');
    expect(data.pmcAvailable).toBe(false);
    expect(data.pmcDbPath).toBe('');
  });
});
