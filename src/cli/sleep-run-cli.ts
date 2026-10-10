// agent-memory-mcp/src/cli/sleep-run-cli.ts
//
// CLI entry point for sleep-run. Initializes the SQLite store and executes
// the one-shot maintenance pipeline.
import { resolve } from 'node:path';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { SqliteMemoryStore } from '../sqlite-store.js';
import { createProjectEmbedder } from '../embedder.js';
import { executeSleepRun, loadCheckpoint, validateReport, type SleepRunConfig } from '../sleep-run.js';
import { loadSleepConfig } from '../sleep-config.js';

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: sleep-run [options]');
    console.log('');
    console.log('Options:');
    console.log('  --db <path>          Path to memory database (required)');
    console.log('  --checkpoint <path>  Path to checkpoint report (default: ~/.config/opencode/pmc/sleep/run-checkpoint.json)');
    console.log('  --config <path>      Path to global config file');
    console.log('  --projects <ids>     Comma-separated project IDs for multi-project mode');
    console.log('  --resume             Load and display last checkpoint');
    console.log('  -h, --help           Show this help message');
    return 0;
  }

  const dbIdx = args.indexOf('--db');
  const checkpointIdx = args.indexOf('--checkpoint');
  const configIdx = args.indexOf('--config');
  const projectsIdx = args.indexOf('--projects');

  const dbPath = dbIdx !== -1 ? resolve(args[dbIdx + 1]) : undefined;
  const checkpointPath = checkpointIdx !== -1
    ? resolve(args[checkpointIdx + 1])
    : join(homedir(), '.config', 'opencode', 'pmc', 'sleep', 'run-checkpoint.json');
  const configPath = configIdx !== -1 ? args[configIdx + 1] : undefined;
  const projectIds = projectsIdx !== -1 ? args[projectsIdx + 1].split(',') : [];
  const resume = args.includes('--resume');

  // Resume mode: just show last checkpoint
  if (resume) {
    const checkpoint = await loadCheckpoint(checkpointPath);
    if (!checkpoint) {
      console.log('No checkpoint found.');
      return 0;
    }
    const errors = validateReport(checkpoint);
    if (errors.length > 0) {
      console.warn('Checkpoint has validation errors:', errors);
    }
    console.log(JSON.stringify(checkpoint, null, 2));
    return 0;
  }

  // Normal run mode: requires --db
  if (!dbPath) {
    console.error('Error: --db <path> is required (unless using --resume).');
    console.error('Usage: sleep-run --db <path-to-memory.db>');
    process.exit(1);
  }

  // Load sleep config for forgetting config
  const sleepConfig = await loadSleepConfig(configPath);

  // Initialize SQLite store with the PROJECT embedding policy (env-driven —
  // E5 prefixes + full-text chunking when the project is configured for E5).
  const embedder = createProjectEmbedder();
  await embedder.initialize();
  const store = new SqliteMemoryStore(dbPath, embedder);
  await store.initialize();
  const db = store.getDb();

  const runConfig: SleepRunConfig = {
    checkpointPath,
    forgettingConfig: sleepConfig.forgetting,
    projectIds,
  };

  console.log(`[sleep-run] Starting maintenance pipeline...`);
  console.log(`  DB: ${dbPath}`);
  console.log(`  Checkpoint: ${checkpointPath}`);
  if (projectIds.length > 0) {
    console.log(`  Projects: ${projectIds.join(', ')}`);
  }

  const report = await executeSleepRun(
    { store, db },
    runConfig,
  );

  // Print summary
  console.log(`\n[sleep-run] ${report.status.toUpperCase()}`);
  console.log(`  Decay: ${report.summary.decayEvaluated} evaluated, ${report.summary.decayTransitioned} transitioned`);
  console.log(`  Promotion: ${report.summary.promotionEvaluated} evaluated, ${report.summary.promotionPromoted} promoted`);
  console.log(`  Enrichment: ${report.summary.symbolsEnriched} enriched, ${report.summary.symbolsDeferred} deferred`);
  if (report.summary.pausedByActivity) {
    console.log('  Paused due to user activity.');
  }
  if (report.errors.length > 0) {
    console.log(`  Errors: ${report.errors.length}`);
    for (const err of report.errors) {
      console.log(`    - ${err}`);
    }
  }
  console.log(`  Report: ${checkpointPath}`);

  store.close();
  return report.status === 'failed' ? 1 : 0;
}

main().catch((err) => {
  console.error('[sleep-run] FATAL:', err.message);
  process.exit(1);
});
