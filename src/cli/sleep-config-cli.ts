// agent-memory-mcp/src/cli/sleep-config-cli.ts
//
// CLI entry point for sleep-config. Handles show/validate/set actions
// for the global sleep-mode configuration.
import {
  loadSleepConfig,
  saveSleepConfig,
  validateSleepConfig,
  getDefaultConfigPath,
  DEFAULT_SLEEP_CONFIG,
  type SleepConfig,
  type FullSleepConfig,
} from '../sleep-config.js';

async function main() {
  const args = process.argv.slice(2);
  const configIdx = args.indexOf('--config');
  const configPath = configIdx !== -1 ? args[configIdx + 1] : undefined;
  const filtered = args.filter((_, i) => i !== configIdx && i !== configIdx + 1);

  const action = filtered[0] ?? 'show';

  switch (action) {
    case 'show': {
      const config = await loadSleepConfig(configPath);
      console.log(JSON.stringify(config, null, 2));
      break;
    }

    case 'validate': {
      const config = await loadSleepConfig(configPath);
      const errors = validateSleepConfig(config.sleep);
      if (errors.length > 0) {
        console.error('Validation errors:');
        for (const err of errors) {
          console.error(`  - ${err}`);
        }
        process.exit(1);
      } else {
        console.log('Config is valid.');
      }
      break;
    }

    case 'set': {
      const key = filtered[1];
      const value = filtered[2];
      if (!key || value === undefined) {
        console.error('Usage: sleep-config set <key> <value>');
        process.exit(1);
      }

      const config = await loadSleepConfig(configPath);
      const sleepConfig: Record<string, unknown> = { ...(config.sleep as unknown as Record<string, unknown>) };

      // Parse value to correct type
      const defaultVal = (DEFAULT_SLEEP_CONFIG as unknown as Record<string, unknown>)[key];
      let parsed: unknown;
      if (typeof defaultVal === 'boolean') {
        parsed = value === 'true';
      } else if (typeof defaultVal === 'number') {
        parsed = Number(value);
        if (isNaN(parsed as number)) {
          console.error(`Error: "${key}" expects a number, got "${value}"`);
          process.exit(1);
        }
      } else {
        parsed = value;
      }

      sleepConfig[key] = parsed;

      // Validate before saving
      const errors = validateSleepConfig(sleepConfig as Partial<SleepConfig>);
      if (errors.length > 0) {
        console.error('Validation errors:');
        for (const err of errors) {
          console.error(`  - ${err}`);
        }
        process.exit(1);
      }

      const fullConfig: FullSleepConfig = {
        sleep: sleepConfig as unknown as SleepConfig,
        forgetting: config.forgetting,
      };
      await saveSleepConfig(fullConfig, configPath);
      console.log(`Set ${key} = ${JSON.stringify(parsed)}`);
      console.log(`Saved to ${configPath ?? getDefaultConfigPath()}`);
      break;
    }

    default:
      console.error(`Unknown action: ${action}`);
      console.error('Actions: show, validate, set');
      process.exit(1);
  }
}

main().catch((err) => {
  console.error('[sleep-config] FATAL:', err.message);
  process.exit(1);
});
