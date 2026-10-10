import { mkdir, writeFile, rename, unlink } from 'fs/promises';
import { join } from 'path';
import type {
  CompareResult,
  JudgeResult,
  Memory,
  MemoryCategory,
  MemoryStatus,
  MemoryStore,
  MemoryStats,
  PruneOptions,
  PruneResult,
  RelationType,
  SearchFilters,
  SearchMode,
  SearchResult,
  SessionForgetResult,
  SessionLedger,
  SessionMeta,
  SessionPrompt,
  SessionPurgeResult,
  SessionResponse,
  SessionSummary,
  SessionToolCall,
  StoreRequest,
  StoreResult,
  UpdateRequest,
} from './types.js';

// ── HardcopyMemoryStore ─────────────────────────────────────────
//
// Transparent decorator that mirrors all mutations to plain JSON
// files on disk. One file per memory, named {id}.json. Read
// operations delegate straight through — the hardcopy is write-only.
//
// Hardcopy errors are logged to stderr but never propagate.
// The primary store is the source of truth; the hardcopy is
// a human-readable escape hatch.

export class HardcopyMemoryStore implements MemoryStore {
  constructor(
    private readonly inner: MemoryStore,
    private readonly hardcopyPath: string,
  ) {}

  async initialize(): Promise<void> {
    await this.inner.initialize();
    await mkdir(this.hardcopyPath, { recursive: true });
  }

  // ── Mutations (mirrored to disk) ────────────────────────────

  async store(request: StoreRequest): Promise<Memory | StoreResult> {
    const result = await this.inner.store(request);
    // Extract Memory for hardcopy — StoreResult wraps it in .stored.
    // When detect_conflicts=true and candidates exist, stored is absent (W2 fix).
    const isStoreResult = (r: Memory | StoreResult): r is StoreResult =>
      'judgment_required' in r;
    const memory: Memory | undefined = isStoreResult(result)
      ? result.stored
      : result;
    if (memory) {
      await this.writeHardcopy(memory);
    }
    return result;
  }

  async storeBatch(requests: StoreRequest[]): Promise<Memory[]> {
    const memories = await this.inner.storeBatch(requests);
    await Promise.all(memories.map(m => this.writeHardcopy(m)));
    return memories;
  }

  async update(id: string, updates: UpdateRequest): Promise<Memory> {
    const memory = await this.inner.update(id, updates);
    await this.writeHardcopy(memory);
    return memory;
  }

  async delete(id: string): Promise<void> {
    await this.inner.delete(id);
    await this.deleteHardcopy(id);
  }

  // ── Reads (pass through) ────────────────────────────────────

  async search(query: string, mode: SearchMode, filters: SearchFilters): Promise<SearchResult[]> {
    return this.inner.search(query, mode, filters);
  }

  async findRelated(memoryId: string, limit: number): Promise<SearchResult[]> {
    return this.inner.findRelated(memoryId, limit);
  }

  async listRecent(limit: number, category?: MemoryCategory): Promise<Memory[]> {
    return this.inner.listRecent(limit, category);
  }

  async stats(): Promise<MemoryStats> {
    return this.inner.stats();
  }

  async prune(options: PruneOptions): Promise<PruneResult> {
    return this.inner.prune(options);
  }

  // ── New PR-1 surface: pure delegation ───────────────────────
  //
  // Session-ledger hardcopy mirroring is intentionally deferred to T-007
  // (PR-4); for PR-1 these methods simply delegate so the decorator
  // satisfies the MemoryStore interface and stays transparent.

  async updateMemoryStatus(id: string, status: MemoryStatus, reviewReason?: string): Promise<Memory> {
    const memory = await this.inner.updateMemoryStatus(id, status, reviewReason);
    await this.writeHardcopy(memory);
    return memory;
  }

  async upsertTopicAlias(topicKey: string, memoryId: string): Promise<void> {
    return this.inner.upsertTopicAlias(topicKey, memoryId);
  }

  async resolveTopic(topicKey: string): Promise<Memory | null> {
    return this.inner.resolveTopic(topicKey);
  }

  async storeSessionPrompt(sessionId: string, rawPrompt: string): Promise<SessionPrompt> {
    return this.inner.storeSessionPrompt(sessionId, rawPrompt);
  }

  async storeSessionResponse(sessionId: string, promptId: string, fullResponse: string): Promise<SessionResponse> {
    return this.inner.storeSessionResponse(sessionId, promptId, fullResponse);
  }

  async storeSessionToolCall(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    toolName: string;
    argsSafe?: string;
    resultSummary?: string;
    importance?: string;
  }): Promise<SessionToolCall> {
    return this.inner.storeSessionToolCall(params);
  }

  async storeSessionSummary(params: {
    sessionId: string;
    promptId?: string;
    responseId?: string;
    summary: string;
    decisions?: string[];
    keyTitles?: string[];
    memoryIds?: string[];
    artifactRefs?: string[];
  }): Promise<SessionSummary> {
    return this.inner.storeSessionSummary(params);
  }

  async getSession(sessionId: string, opts?: { limit?: number; includeSummary?: boolean; includeForgotten?: boolean }): Promise<SessionLedger> {
    return this.inner.getSession(sessionId, opts);
  }

  async listSessions(limit?: number): Promise<SessionMeta[]> {
    return this.inner.listSessions(limit);
  }

  async linkPromptToMemory(promptId: string, memoryId: string, relation: string, responseId?: string): Promise<void> {
    return this.inner.linkPromptToMemory(promptId, memoryId, relation, responseId);
  }

  async forgetSessionItem(id: string): Promise<SessionForgetResult> {
    return this.inner.forgetSessionItem(id);
  }

  async purgeSessionItem(id: string): Promise<SessionPurgeResult> {
    return this.inner.purgeSessionItem(id);
  }

  setSessionContext(sessionId: string, projectId: string): void {
    this.inner.setSessionContext(sessionId, projectId);
  }

  getSessionContext(): { sessionId: string; projectId: string } | null {
    return this.inner.getSessionContext();
  }

  // ── Symbol-linked memory context (v7) — pass through ─────────

  async attachSymbol(opts: { symbolKey: string; memoryId?: string; sessionItemId?: string }): Promise<{ id: string; symbolKey: string }> {
    return this.inner.attachSymbol(opts);
  }

  async getBySymbol(symbolKey: string, opts?: { sources?: string[]; limit?: number }): Promise<import('./types.js').SymbolLinkedResult[]> {
    return this.inner.getBySymbol(symbolKey, opts);
  }

  // ── Pin/unpin (v8) — pass through ──────────────────────────────

  async pin(id: string): Promise<Memory> {
    const memory = await this.inner.pin(id);
    await this.writeHardcopy(memory);
    return memory;
  }

  async unpin(id: string): Promise<Memory> {
    const memory = await this.inner.unpin(id);
    await this.writeHardcopy(memory);
    return memory;
  }

  // ── Conflict resolution (v9) — pass through ──────────────────────

  async compare(idA: string, idB: string): Promise<CompareResult> {
    return this.inner.compare(idA, idB);
  }

  async judge(judgmentId: string, relation: RelationType, reasoning?: string, confidence?: number): Promise<JudgeResult> {
    return this.inner.judge(judgmentId, relation, reasoning, confidence);
  }

  // ── Review queue (v9 T-008) — pass through (no hardcopy for ephemeral queue) ──

  async listReviewQueue(opts?: { projectId?: string; limit?: number }): Promise<{ count: number; items: Memory[] }> {
    return this.inner.listReviewQueue(opts);
  }

  // ── Private ─────────────────────────────────────────────────

  private safeIdPath(id: string): string | null {
    if (!/^[A-Za-z0-9._-]+$/.test(id) || id.includes('..')) return null;
    return join(this.hardcopyPath, `${id}.json`);
  }

  private async writeHardcopy(memory: Memory): Promise<void> {
    const finalPath = this.safeIdPath(memory.id);
    if (!finalPath) {
      console.error(`[hardcopy] Skipping unsafe memory id: ${JSON.stringify(memory.id)}`);
      return;
    }
    const tmpPath = `${finalPath}.tmp.${process.pid}`;
    try {
      await writeFile(tmpPath, JSON.stringify(memory, null, 2) + '\n');
      await rename(tmpPath, finalPath);
    } catch (err) {
      // Best-effort: clean up temp file, then ignore error
      try { await unlink(tmpPath); } catch {}
      console.error(`[hardcopy] Failed to write ${memory.id}:`, err);
    }
  }

  private async deleteHardcopy(id: string): Promise<void> {
    const filePath = this.safeIdPath(id);
    if (!filePath) {
      console.error(`[hardcopy] Skipping unsafe memory id: ${JSON.stringify(id)}`);
      return;
    }
    try {
      await unlink(filePath);
    } catch (err) {
      // File may not exist (e.g. hardcopy was enabled after the memory
      // was created). That's fine — don't log ENOENT noise.
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
        console.error(`[hardcopy] Failed to delete ${id}:`, err);
      }
    }
  }
}
