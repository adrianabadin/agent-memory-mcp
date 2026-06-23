// agent-memory-mcp/src/global/global-promotion.ts
//
// Promotion rules engine: evaluates memories against promotion criteria and
// creates audit trail entries in global_promotion_log.
//
// Promotion criteria:
//   • 2+ projects sharing same content_hash
//   • 2+ times in same project if critical/security/data-loss category
//   • Reused/revived 3+ times (memory_revive_reviews)
//
// Categories eligible: bugfix, pattern, architecture, security, performance.
// Confidence threshold: >= 0.75.

import { randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';

// ── Types ──────────────────────────────────────────────────────────

export interface PromotionRuleResult {
  matched: boolean;
  ruleName: string;
  sourceIds: string[];
  sourceProjects: string[];
  confidence: number;
  reason: string;
}

export interface PromotionResult {
  evaluated: number;
  promoted: number;
  skipped: number;
  errors: string[];
  promotedMemoryIds: string[];
}

export interface GlobalPromotionLogRecord {
  id: string;
  promoted_memory_id: string;
  source_ids: string;       // JSON array
  source_projects: string;  // JSON array
  rule_matched: string;
  confidence: number;
  model: string | null;
  reviewer_provenance: string | null;
  created_at: string;
}

export interface PromotionCandidate {
  memoryId: string;
  contentHash: string | null;
  category: string;
  sessionId: string | null;
}

// ── Critical categories eligible for promotion ─────────────────────

const CRITICAL_CATEGORIES = new Set([
  'bug-fix',
  'architecture',
  'security',
  'performance',
]);

// ── Promotion Rule Engine ──────────────────────────────────────────

export class PromotionRuleEngine {
  constructor(private readonly db: DatabaseSync) {}

  /**
   * Multi-project recurrence rule: same content_hash appears in 2+ distinct
   * session_id values (used as project proxy). Falls back to exact content
   * match when content_hash is null.
   */
  matchMultiProjectRule(memoryId: string): PromotionRuleResult {
    const row = this.db.prepare(
      'SELECT content, content_hash, session_id FROM memories WHERE id = ?',
    ).get(memoryId) as { content: string; content_hash: string | null; session_id: string | null } | undefined;

    if (!row) return noMatch('multi_project_occurrence', 'memory not found');

    // Prefer content_hash for performance; fall back to content match
    const matches = row.content_hash
      ? this.db.prepare(
          `SELECT id, session_id FROM memories
           WHERE content_hash = ? AND session_id IS NOT NULL AND session_id != ''`,
        ).all(row.content_hash) as Array<{ id: string; session_id: string }>
      : this.db.prepare(
          `SELECT id, session_id FROM memories
           WHERE content = ? AND session_id IS NOT NULL AND session_id != ''`,
        ).all(row.content) as Array<{ id: string; session_id: string }>;

    const uniqueProjects = [...new Set(matches.map(m => m.session_id))];
    if (uniqueProjects.length < 2) {
      return noMatch('multi_project_occurrence', `only ${uniqueProjects.length} project(s)`);
    }

    return {
      matched: true,
      ruleName: 'multi_project_occurrence',
      sourceIds: matches.map(m => m.id),
      sourceProjects: uniqueProjects,
      confidence: Math.min(0.75 + uniqueProjects.length * 0.05, 1.0),
      reason: `content found in ${uniqueProjects.length} projects`,
    };
  }

  /**
   * Critical-category repetition: 2+ memories in same project with a
   * critical category (security, performance, bug-fix, architecture).
   */
  matchCriticalCategoryRule(memoryId: string): PromotionRuleResult {
    const row = this.db.prepare(
      'SELECT category, session_id FROM memories WHERE id = ?',
    ).get(memoryId) as { category: string; session_id: string | null } | undefined;

    if (!row) return noMatch('critical_category_repetition', 'memory not found');

    if (!CRITICAL_CATEGORIES.has(row.category)) {
      return noMatch('critical_category_repetition', `category '${row.category}' not critical`);
    }

    const projectId = row.session_id ?? '';
    const matches = this.db.prepare(
      `SELECT id FROM memories WHERE category = ? AND session_id = ? AND session_id IS NOT NULL AND session_id != ''`,
    ).all(row.category, projectId) as Array<{ id: string }>;

    if (matches.length < 2) {
      return noMatch('critical_category_repetition', `only ${matches.length} occurrence(s)`);
    }

    return {
      matched: true,
      ruleName: 'critical_category_repetition',
      sourceIds: matches.map(m => m.id),
      sourceProjects: [projectId],
      confidence: 0.80,
      reason: `${matches.length} '${row.category}' memories in same project`,
    };
  }

  /**
   * Revival-based rule: memory accessed/revived 3+ times from deep decay
   * (shadow/gist_only) via memory_revive_reviews.
   */
  matchRevivalRule(memoryId: string): PromotionRuleResult {
    const reviews = this.db.prepare(
      `SELECT id FROM memory_revive_reviews WHERE memory_id = ? AND decision = 'revived'`,
    ).all(memoryId) as Array<{ id: string }>;

    if (reviews.length < 3) {
      return noMatch('reuse_revival_detection', `only ${reviews.length} revival(s)`);
    }

    return {
      matched: true,
      ruleName: 'reuse_revival_detection',
      sourceIds: reviews.map(r => r.id),
      sourceProjects: [],
      confidence: Math.min(0.75 + reviews.length * 0.03, 1.0),
      reason: `revived ${reviews.length} times`,
    };
  }

  /**
   * Run all promotion rules for a memory and execute promotions that
   * meet the confidence threshold (>= 0.75). Creates audit trail entries.
   */
  evaluateForPromotion(memoryId: string): PromotionResult {
    const result: PromotionResult = {
      evaluated: 1,
      promoted: 0,
      skipped: 0,
      errors: [],
      promotedMemoryIds: [],
    };

    const rules = [
      this.matchMultiProjectRule(memoryId),
      this.matchCriticalCategoryRule(memoryId),
      this.matchRevivalRule(memoryId),
    ];

    const CONFIDENCE_THRESHOLD = 0.75;

    this.db.exec('BEGIN');
    try {
      for (const ruleResult of rules) {
        if (!ruleResult.matched || ruleResult.confidence < CONFIDENCE_THRESHOLD) {
          if (ruleResult.matched) result.skipped++;
          continue;
        }

        // Check for duplicate promotion (same memory + same source_ids)
        const sourceIdsKey = JSON.stringify([...ruleResult.sourceIds].sort());
        const existing = this.db.prepare(
          'SELECT id FROM global_promotion_log WHERE promoted_memory_id = ? AND source_ids = ?',
        ).get(memoryId, sourceIdsKey) as { id: string } | undefined;

        if (existing) {
          result.skipped++;
          continue;
        }

        // Insert audit log entry
        const logId = randomUUID();
        const now = new Date().toISOString();
        this.db.prepare(
          `INSERT INTO global_promotion_log
            (id, promoted_memory_id, source_ids, source_projects, rule_matched, confidence, model, reviewer_provenance, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          logId,
          memoryId,
          sourceIdsKey,
          JSON.stringify(ruleResult.sourceProjects),
          ruleResult.ruleName,
          ruleResult.confidence,
          null, // model — future use
          null, // reviewer_provenance — future use
          now,
        );

        // Mark memory as non_forgettable
        this.db.prepare(
          'UPDATE memories SET non_forgettable = 1 WHERE id = ?',
        ).run(memoryId);

        result.promoted++;
        if (!result.promotedMemoryIds.includes(memoryId)) {
          result.promotedMemoryIds.push(memoryId);
        }
      }

      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      result.errors.push(String(e));
    }

    return result;
  }
}

// ── Helpers ────────────────────────────────────────────────────────

function noMatch(ruleName: string, reason: string): PromotionRuleResult {
  return {
    matched: false,
    ruleName,
    sourceIds: [],
    sourceProjects: [],
    confidence: 0,
    reason,
  };
}
