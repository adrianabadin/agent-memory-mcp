// src/cli/reembed-cli.ts
//
// M2 CLI: snapshot a live PMC backend database and re-vectorize it with the
// configured (E5) embedder — WITHOUT ever mutating the source.
//
//   node dist/cli/reembed-cli.js \
//     --source .planning/project-memory-context/memory-db.db \
//     --target .planning/project-memory-context/memory-db-e5.db \
//     --report .planning/project-memory-context/reembed-report.json
//
//   --resume        continue an interrupted migration (same snapshot+identity)
//   --verify-only   run all checks against an already-COMPLETE target
//   --batch-size N  streaming batch size (default 25)
//
// The embedder comes from the standard env (EMBEDDING_MODEL /
// EMBEDDING_DIMENSIONS / EMBEDDING_POOLING / EMBEDDING_CACHE_PATH /
// EMBEDDING_WEIGHT_CACHE_PATH), so the real run uses the already-cached
// multilingual-e5-small offline (HF_HUB_OFFLINE=1) — no downloads here.
//
// This CLI deliberately does NOT touch runtime config, the global backend,
// or the source database: cutover is M3, executed separately by the parent.
import { createProjectEmbedder } from '../embedder.js';
import { runReembed, DEFAULT_BATCH_SIZE } from '../embedding-reindex.js';

interface CliArgs {
  source?: string;
  target?: string;
  report?: string;
  batchSize: number;
  resume: boolean;
  verifyOnly: boolean;
  help: boolean;
}

export function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { batchSize: DEFAULT_BATCH_SIZE, resume: false, verifyOnly: false, help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '--source': args.source = argv[++i]; break;
      case '--target': args.target = argv[++i]; break;
      case '--report': args.report = argv[++i]; break;
      case '--batch-size': {
        const n = Number.parseInt(argv[++i], 10);
        if (!Number.isInteger(n) || n < 1 || n > 1000) {
          throw new Error(`--batch-size must be an integer in [1, 1000] (got '${argv[i]}').`);
        }
        args.batchSize = n;
        break;
      }
      case '--resume': args.resume = true; break;
      case '--verify-only': args.verifyOnly = true; break;
      case '--help': case '-h': args.help = true; break;
      default:
        if (a.startsWith('--')) throw new Error(`Unknown flag: ${a}`);
        break;
    }
  }
  return args;
}

const USAGE = `Usage: reembed-cli --source <db> --target <new-db> [options]

  --source <path>      Existing PMC database (opened READ-ONLY; never mutated).
  --target <path>      New sibling database to create (must not exist).
  --report <path>      Sidecar JSON report (counts/hashes/timings — no raw text).
  --batch-size <n>     Streaming batch size (default ${DEFAULT_BATCH_SIZE}).
  --resume             Continue an interrupted migration on the same target.
  --verify-only        Verify an existing COMPLETE target (no embedding).

Environment: EMBEDDING_MODEL / EMBEDDING_DIMENSIONS / EMBEDDING_POOLING /
EMBEDDING_CACHE_PATH / EMBEDDING_WEIGHT_CACHE_PATH (offline E5 cache).
`;

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  let args: CliArgs;
  try {
    args = parseArgs(argv);
  } catch (err) {
    process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n\n${USAGE}`);
    return 2;
  }
  if (args.help) {
    process.stdout.write(USAGE);
    return 0;
  }
  if (!args.source || !args.target) {
    process.stderr.write(`--source and --target are required.\n\n${USAGE}`);
    return 2;
  }

  // Embedder: env-configured project embedder (offline cached weights). The
  // CLI owns its lifecycle — initialize here, cleanup in runReembed's finally.
  const embedder = args.verifyOnly ? undefined : createProjectEmbedder();

  const result = await runReembed({
    sourcePath: args.source,
    targetPath: args.target,
    embedder,
    batchSize: args.batchSize,
    reportPath: args.report,
    resume: args.resume,
    verifyOnly: args.verifyOnly,
  });

  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  if (result.status !== 'COMPLETE') {
    process.stderr.write(`FAILED: ${result.error ?? 'unknown error'}\n`);
    return 1;
  }
  return 0;
}

// Direct execution entry (not imported as a module by tests).
const invokedDirectly =
  process.argv[1] !== undefined &&
  (process.argv[1].endsWith('reembed-cli.js') || process.argv[1].endsWith('reembed-cli.ts'));
if (invokedDirectly) {
  main().then(
    (code) => process.exit(code),
    (err) => {
      process.stderr.write(`${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
      process.exit(1);
    },
  );
}
