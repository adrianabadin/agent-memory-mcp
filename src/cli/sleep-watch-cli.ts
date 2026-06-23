// agent-memory-mcp/src/cli/sleep-watch-cli.ts
//
// CLI entry point for sleep-watch. Loads config and starts the idle daemon.
import { resolve } from 'node:path';
import { main as startDaemon, getDaemonStatus, SleepWatchDaemon } from '../sleep-watch.js';
import { getDefaultConfigPath } from '../sleep-config.js';
import { join } from 'node:path';
import { homedir } from 'node:os';

const DEFAULT_CHECKPOINT = join(homedir(), '.config', 'opencode', 'pmc', 'sleep', 'daemon-checkpoint.json');

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: sleep-watch [options]');
    console.log('');
    console.log('Options:');
    console.log('  --config <path>      Path to global config file');
    console.log('  --checkpoint <path>  Path to daemon checkpoint');
    console.log('  --status             Show daemon status from checkpoint');
    console.log('  --stop               Stop the daemon (sends SIGTERM)');
    console.log('  --poll-ms <ms>       Polling interval in ms (default: 30000)');
    console.log('  -h, --help           Show this help message');
    return 0;
  }

  const configIdx = args.indexOf('--config');
  const checkpointIdx = args.indexOf('--checkpoint');
  const pollIdx = args.indexOf('--poll-ms');

  const configPath = configIdx !== -1 ? args[configIdx + 1] : undefined;
  const checkpointPath = checkpointIdx !== -1 ? resolve(args[checkpointIdx + 1]) : DEFAULT_CHECKPOINT;
  const pollMs = pollIdx !== -1 ? parseInt(args[pollIdx + 1], 10) : undefined;

  // Status command
  if (args.includes('--status')) {
    const status = await getDaemonStatus(checkpointPath);
    if (!status) {
      console.log('No daemon checkpoint found.');
      return 0;
    }
    console.log(JSON.stringify(status, null, 2));
    return 0;
  }

  // Stop command
  if (args.includes('--stop')) {
    const status = await getDaemonStatus(checkpointPath);
    if (!status) {
      console.log('No daemon checkpoint found.');
      return 0;
    }
    // Try to kill by PID (best effort)
    try {
      process.kill(status.daemonId as unknown as number, 'SIGTERM');
    } catch {
      // PID may not be directly killable; just report
    }
    console.log(`Stop signal sent to daemon ${status.daemonId}.`);
    await SleepWatchDaemon.deleteCheckpoint(checkpointPath);
    return 0;
  }

  // Start daemon
  console.log('[sleep-watch] Starting idle daemon...');
  console.log(`  Config: ${configPath ?? getDefaultConfigPath()}`);
  console.log(`  Checkpoint: ${checkpointPath}`);
  if (pollMs) console.log(`  Poll interval: ${pollMs}ms`);

  const daemon = await startDaemon({
    configPath,
    checkpointPath,
    pollIntervalMs: pollMs,
  });

  console.log(`[sleep-watch] Daemon running (id: ${daemon.getId()}). Press Ctrl+C to stop.`);

  // Keep process alive
  await new Promise<void>((resolve) => {
    process.on('SIGINT', async () => {
      console.log('\n[sleep-watch] Shutting down...');
      await daemon.stop();
      resolve();
    });
    process.on('SIGTERM', async () => {
      await daemon.stop();
      resolve();
    });
  });

  return 0;
}

main().catch((err) => {
  console.error('[sleep-watch] FATAL:', err.message);
  process.exit(1);
});
