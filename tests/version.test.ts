import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { packageMetadata } from '../src/version.js';

describe('package identity metadata', () => {
  it('stays synchronized with package.json', async () => {
    const packageJson = JSON.parse(
      await readFile(new URL('../package.json', import.meta.url), 'utf8')
    ) as { version: string };

    expect(packageMetadata).toEqual({
      version: packageJson.version,
      serverName: 'agent-memory',
    });
  });
});
