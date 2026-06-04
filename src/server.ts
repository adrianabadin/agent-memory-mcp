import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { MemoryStore } from './types.js';
import type { GlobalStore } from './global/global-store.js';
import { registerTools } from './tools.js';
import { packageMetadata } from './version.js';

export function createServer(store: MemoryStore, globalStore?: GlobalStore): McpServer {
  const server = new McpServer({
    name: packageMetadata.serverName,
    version: packageMetadata.version,
  });

  registerTools(server, store, globalStore);

  return server;
}
