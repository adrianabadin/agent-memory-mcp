import { defineConfig, type Plugin } from 'vitest/config';
import { createRequire } from 'node:module';

// Vite strips the `node:` prefix and then fails to resolve `sqlite` as a file.
// This plugin intercepts the resolution and inlines the exports from node:sqlite.
const req = createRequire(import.meta.url);
const sqliteMod = req('node:sqlite') as Record<string, unknown>;
const sqliteExports = Object.keys(sqliteMod);

const nodeSqlitePlugin: Plugin = {
  name: 'node-sqlite-external',
  enforce: 'pre',
  resolveId(id: string) {
    if (id === 'node:sqlite' || id === 'sqlite') {
      return '\0virtual:node-sqlite';
    }
  },
  load(id: string) {
    if (id === '\0virtual:node-sqlite') {
      // Re-export each named export from a CJS require call inline.
      // Using a synchronous IIFE so no top-level await is needed.
      const lines = [
        `import { createRequire } from 'node:module';`,
        `const _req = createRequire(import.meta.url);`,
        `const _m = _req('node:sqlite');`,
        ...sqliteExports.map(k => `export const ${k} = _m.${k};`),
        `export default _m;`,
      ];
      return lines.join('\n');
    }
  },
};

export default defineConfig({
  plugins: [nodeSqlitePlugin],
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/index.ts'],
    },
  },
});
