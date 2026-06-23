// agent-memory-mcp/src/sleep-watch.ts
//
// Sleep-mode daemon with idle detection, CPU pressure monitoring, and
// keep-awake lease management. Polls for idle eligibility and spawns
// sleep-run when conditions are met.
//
// Pipeline:
//   1. Load config (global only)
//   2. Acquire keep-awake lease
//   3. Poll idle/CPU every 30 seconds
//   4. When eligible, spawn sleep-run (detached)
//   5. Track running spawn PID
//   6. On signal, gracefully shutdown
//   7. Checkpoint/resume on crash

import { randomUUID } from 'node:crypto';
import { writeFile, readFile, mkdir, unlink } from 'node:fs/promises';
import { dirname } from 'node:path';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';
import type { SleepConfig, FullSleepConfig } from './sleep-config.js';
import { loadSleepConfig } from './sleep-config.js';
import { InMemoryKeepAwakeLeaseManager } from './sleep-run.js';

// ── Platform Abstraction ────────────────────────────────────────────

export interface PlatformOps {
  /** Returns seconds since last user input (keyboard/mouse). -1 if unavailable. */
  getIdleSeconds(): Promise<number>;
  /** Returns current CPU usage as 0-100. 0 if unavailable. */
  getCPUUsage(): Promise<number>;
  /** Returns true if machine is on AC power (not battery). */
  isPluggedIn(): Promise<boolean>;
}

// ── Daemon State ────────────────────────────────────────────────────

export type DaemonStatus = 'idle' | 'running' | 'paused' | 'stopping' | 'stopped';

export interface DaemonState {
  status: DaemonStatus;
  startedAt: string;
  lastIdleCheck: string | null;
  lastRunSpawnedAt: string | null;
  currentRunPid: number | null;
  currentRunStartedAt: string | null;
  totalRunsSpawned: number;
  totalRunsCompleted: number;
  pauseReason: string | null;
  idleSeconds: number;
  cpuUsage: number;
  isPluggedIn: boolean;
  leaseActive: boolean;
}

export interface DaemonCheckpoint {
  daemonId: string;
  state: DaemonState;
  config: SleepConfig;
  savedAt: string;
}

// ── Daemon Config ───────────────────────────────────────────────────

export interface SleepWatchConfig {
  /** Path to global config file. */
  configPath?: string;
  /** Polling interval in ms. Default: 30_000 (30 seconds) */
  pollIntervalMs?: number;
  /** Path to checkpoint file. */
  checkpointPath?: string;
  /** Path to sleep-run script. */
  sleepRunScript?: string;
  /** Platform operations (for testing). */
  platformOps?: PlatformOps;
}



// ── Eligibility Check ───────────────────────────────────────────────

export interface IdleEligibilityResult {
  eligible: boolean;
  reason: string;
  idleSeconds: number;
  cpuUsage: number;
  isPluggedIn: boolean;
}

/**
 * Check if the machine is eligible for a sleep-run based on idle time,
 * CPU usage, and power status.
 */
export function checkIdleEligibility(
  config: SleepConfig,
  idleSeconds: number,
  cpuUsage: number,
  isPluggedIn: boolean,
): IdleEligibilityResult {
  const idleThresholdSeconds = config.idleMinMinutes * 60;

  // Check power status
  if (config.onlyWhenPluggedIn && !isPluggedIn) {
    return {
      eligible: false,
      reason: 'not_plugged_in',
      idleSeconds,
      cpuUsage,
      isPluggedIn,
    };
  }

  // Check idle time
  if (idleSeconds >= 0 && idleSeconds < idleThresholdSeconds) {
    return {
      eligible: false,
      reason: 'not_idle',
      idleSeconds,
      cpuUsage,
      isPluggedIn,
    };
  }

  // Check CPU pressure
  if (cpuUsage > config.cpuHighPercent) {
    return {
      eligible: false,
      reason: 'cpu_pressure',
      idleSeconds,
      cpuUsage,
      isPluggedIn,
    };
  }

  return {
    eligible: true,
    reason: 'eligible',
    idleSeconds,
    cpuUsage,
    isPluggedIn,
  };
}

// ── Sleep-Watch Daemon ──────────────────────────────────────────────

export class SleepWatchDaemon {
  private config: FullSleepConfig;
  private daemonId: string;
  private platformOps: PlatformOps;
  private pollIntervalMs: number;
  private checkpointPath: string;
  private sleepRunScript: string;
  private leaseManager: InMemoryKeepAwakeLeaseManager;

  private state: DaemonState;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private leaseRenewalTimer: ReturnType<typeof setInterval> | null = null;
  private currentRunProcess: ChildProcess | null = null;
  private shutdownRequested = false;
  private stateChangeCallbacks: Array<(state: DaemonState) => void> = [];

  constructor(config: FullSleepConfig, options: SleepWatchConfig = {}) {
    this.config = config;
    this.daemonId = randomUUID();
    this.platformOps = options.platformOps ?? { getIdleSeconds: async () => -1, getCPUUsage: async () => 0, isPluggedIn: async () => true };
    this.pollIntervalMs = options.pollIntervalMs ?? 30_000;
    this.checkpointPath = options.checkpointPath ?? join(homedir(), '.config', 'opencode', 'pmc', 'sleep', 'daemon-checkpoint.json');
    this.sleepRunScript = options.sleepRunScript ?? 'sleep-run.mjs';
    this.leaseManager = new InMemoryKeepAwakeLeaseManager();

    this.state = {
      status: 'idle',
      startedAt: new Date().toISOString(),
      lastIdleCheck: null,
      lastRunSpawnedAt: null,
      currentRunPid: null,
      currentRunStartedAt: null,
      totalRunsSpawned: 0,
      totalRunsCompleted: 0,
      pauseReason: null,
      idleSeconds: 0,
      cpuUsage: 0,
      isPluggedIn: true,
      leaseActive: false,
    };
  }

  /** Register a callback for state changes. */
  onStateChange(callback: (state: DaemonState) => void): void {
    this.stateChangeCallbacks.push(callback);
  }

  /** Get current daemon state (readonly snapshot). */
  getState(): Readonly<DaemonState> {
    return { ...this.state };
  }

  /** Get daemon ID. */
  getId(): string {
    return this.daemonId;
  }

  /** Start the daemon. */
  async start(): Promise<void> {
    if (this.state.status !== 'idle' && this.state.status !== 'stopped') {
      throw new Error(`Cannot start daemon in state: ${this.state.status}`);
    }

    this.shutdownRequested = false;
    this.state.status = 'running';
    this.state.startedAt = new Date().toISOString();
    this.emitStateChange();

    // Acquire keep-awake lease
    const leaseTTL = this.config.sleep.keepAwakeLeaseMinutes * 60 * 1000;
    const acquired = this.leaseManager.acquire(`sleep-watch-${this.daemonId}`, leaseTTL);
    this.state.leaseActive = acquired;
    this.emitStateChange();

    // Start lease renewal timer (every 10 minutes)
    this.leaseRenewalTimer = setInterval(() => {
      this.renewLease();
    }, 10 * 60 * 1000);

    // Start polling
    this.pollTimer = setInterval(() => {
      this.poll().catch(() => { /* non-fatal */ });
    }, this.pollIntervalMs);

    // Initial poll
    await this.poll();
  }

  /** Stop the daemon gracefully. */
  async stop(): Promise<void> {
    if (this.state.status === 'stopped' || this.state.status === 'stopping') {
      return;
    }

    this.shutdownRequested = true;
    this.state.status = 'stopping';
    this.emitStateChange();

    // Clear timers
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.leaseRenewalTimer) {
      clearInterval(this.leaseRenewalTimer);
      this.leaseRenewalTimer = null;
    }

    // Wait for current run to finish (max 30 seconds)
    if (this.currentRunProcess) {
      const maxWait = 30_000;
      const start = Date.now();
      while (this.currentRunProcess && Date.now() - start < maxWait) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // Release lease
    this.leaseManager.release(`sleep-watch-${this.daemonId}`);
    this.state.leaseActive = false;

    this.state.status = 'stopped';
    this.state.currentRunPid = null;
    this.emitStateChange();

    // Save final checkpoint
    await this.saveCheckpoint();
  }

  /** Pause the daemon (e.g., on SIGUSR1). */
  pause(reason: string): void {
    if (this.state.status !== 'running') return;

    this.state.status = 'paused';
    this.state.pauseReason = reason;
    this.emitStateChange();
  }

  /** Resume the daemon (e.g., on SIGUSR2). */
  resume(): void {
    if (this.state.status !== 'paused') return;

    this.state.status = 'running';
    this.state.pauseReason = null;
    this.emitStateChange();
  }

  /** Single poll cycle: check eligibility and spawn if needed. */
  private async poll(): Promise<void> {
    if (this.shutdownRequested || this.state.status !== 'running') {
      return;
    }

    // Check if a run is still in flight
    if (this.currentRunProcess) {
      // Check if process exited
      const exited = this.currentRunProcess.exitCode !== null || this.currentRunProcess.killed;
      if (exited) {
        this.currentRunProcess = null;
        this.state.currentRunPid = null;
        this.state.currentRunStartedAt = null;
        this.state.totalRunsCompleted++;
        this.emitStateChange();
      } else {
        // Run still in progress — check CPU pressure
        const cpuUsage = await this.platformOps.getCPUUsage();
        if (cpuUsage > this.config.sleep.cpuHighPercent) {
          // CPU pressure — pause after current symbol
          // The sleep-run will checkpoint at symbol boundary
          this.state.cpuUsage = cpuUsage;
          this.emitStateChange();
        }
        return; // Don't spawn new run while one is in flight
      }
    }

    // Check idle eligibility
    const idleSeconds = await this.platformOps.getIdleSeconds();
    const cpuUsage = await this.platformOps.getCPUUsage();
    const isPluggedIn = await this.platformOps.isPluggedIn();

    this.state.idleSeconds = idleSeconds;
    this.state.cpuUsage = cpuUsage;
    this.state.isPluggedIn = isPluggedIn;
    this.state.lastIdleCheck = new Date().toISOString();
    this.emitStateChange();

    const eligibility = checkIdleEligibility(
      this.config.sleep,
      idleSeconds,
      cpuUsage,
      isPluggedIn,
    );

    if (!eligibility.eligible) {
      return;
    }

    // Spawn sleep-run
    await this.spawnSleepRun();
  }

  /** Spawn a detached sleep-run process. */
  private async spawnSleepRun(): Promise<void> {
    const runId = randomUUID();

    try {
      const child = spawn('node', [this.sleepRunScript], {
        detached: true,
        stdio: 'ignore',
        shell: false,
        env: {
          ...process.env,
          SLEEP_RUN_ID: runId,
          SLEEP_DAEMON_ID: this.daemonId,
        },
      });

      child.unref();

      this.currentRunProcess = child;
      this.state.currentRunPid = child.pid ?? null;
      this.state.currentRunStartedAt = new Date().toISOString();
      this.state.lastRunSpawnedAt = new Date().toISOString();
      this.state.totalRunsSpawned++;
      this.emitStateChange();

      // Handle child exit
      child.on('exit', () => {
        this.currentRunProcess = null;
        this.state.currentRunPid = null;
        this.state.currentRunStartedAt = null;
        this.state.totalRunsCompleted++;
        this.emitStateChange();
      });

      child.on('error', () => {
        this.currentRunProcess = null;
        this.state.currentRunPid = null;
        this.state.currentRunStartedAt = null;
        this.emitStateChange();
      });
    } catch {
      // Spawn failure — non-fatal
    }
  }

  /** Renew the keep-awake lease. */
  private renewLease(): void {
    const leaseTTL = this.config.sleep.keepAwakeLeaseMinutes * 60 * 1000;
    const renewed = this.leaseManager.renew(`sleep-watch-${this.daemonId}`, leaseTTL);
    this.state.leaseActive = renewed;
    this.emitStateChange();
  }

  /** Emit state change to callbacks. */
  private emitStateChange(): void {
    for (const cb of this.stateChangeCallbacks) {
      try {
        cb({ ...this.state });
      } catch {
        // Non-fatal
      }
    }
  }

  /** Save checkpoint to disk. */
  async saveCheckpoint(): Promise<void> {
    const checkpoint: DaemonCheckpoint = {
      daemonId: this.daemonId,
      state: { ...this.state },
      config: this.config.sleep,
      savedAt: new Date().toISOString(),
    };

    try {
      await mkdir(dirname(this.checkpointPath), { recursive: true });
      await writeFile(this.checkpointPath, JSON.stringify(checkpoint, null, 2), 'utf-8');
    } catch {
      // Checkpoint persistence failure is non-fatal
    }
  }

  /** Load checkpoint from disk. Returns null if not found or invalid. */
  static async loadCheckpoint(checkpointPath: string): Promise<DaemonCheckpoint | null> {
    try {
      const raw = await readFile(checkpointPath, 'utf-8');
      return JSON.parse(raw) as DaemonCheckpoint;
    } catch {
      return null;
    }
  }

  /** Delete checkpoint file. */
  static async deleteCheckpoint(checkpointPath: string): Promise<void> {
    try {
      await unlink(checkpointPath);
    } catch {
      // Non-fatal
    }
  }
}

// ── Status Reporting ────────────────────────────────────────────────

export interface SleepWatchStatus {
  daemonId: string;
  status: DaemonStatus;
  startedAt: string;
  uptimeMs: number;
  lastIdleCheck: string | null;
  idleSeconds: number;
  cpuUsage: number;
  isPluggedIn: boolean;
  leaseActive: boolean;
  currentRunPid: number | null;
  currentRunStartedAt: string | null;
  totalRunsSpawned: number;
  totalRunsCompleted: number;
  pauseReason: string | null;
  config: SleepConfig;
}

/**
 * Get status of a running daemon from its checkpoint.
 */
export async function getDaemonStatus(checkpointPath: string): Promise<SleepWatchStatus | null> {
  const checkpoint = await SleepWatchDaemon.loadCheckpoint(checkpointPath);
  if (!checkpoint) return null;

  const uptimeMs = Date.now() - new Date(checkpoint.state.startedAt).getTime();

  return {
    daemonId: checkpoint.daemonId,
    status: checkpoint.state.status,
    startedAt: checkpoint.state.startedAt,
    uptimeMs,
    lastIdleCheck: checkpoint.state.lastIdleCheck,
    idleSeconds: checkpoint.state.idleSeconds,
    cpuUsage: checkpoint.state.cpuUsage,
    isPluggedIn: checkpoint.state.isPluggedIn,
    leaseActive: checkpoint.state.leaseActive,
    currentRunPid: checkpoint.state.currentRunPid,
    currentRunStartedAt: checkpoint.state.currentRunStartedAt,
    totalRunsSpawned: checkpoint.state.totalRunsSpawned,
    totalRunsCompleted: checkpoint.state.totalRunsCompleted,
    pauseReason: checkpoint.state.pauseReason,
    config: checkpoint.config,
  };
}

// ── Main Entry Point ────────────────────────────────────────────────

export interface SleepWatchMainOptions {
  configPath?: string;
  checkpointPath?: string;
  sleepRunScript?: string;
  platformOps?: PlatformOps;
  pollIntervalMs?: number;
}

/**
 * Main entry point for the sleep-watch daemon.
 * Loads config, creates daemon, sets up signal handlers, and starts.
 */
export async function main(options: SleepWatchMainOptions = {}): Promise<SleepWatchDaemon> {
  const config = await loadSleepConfig(options.configPath);
  const checkpointPath = options.checkpointPath ?? join(homedir(), '.config', 'opencode', 'pmc', 'sleep', 'daemon-checkpoint.json');

  const daemon = new SleepWatchDaemon(config, {
    configPath: options.configPath,
    checkpointPath,
    sleepRunScript: options.sleepRunScript,
    platformOps: options.platformOps,
    pollIntervalMs: options.pollIntervalMs,
  });

  // Setup signal handlers
  const setupSignalHandlers = () => {
    // SIGINT/SIGTERM: graceful shutdown
    const gracefulShutdown = async () => {
      await daemon.stop();
      process.exit(0);
    };

    process.on('SIGINT', gracefulShutdown);
    process.on('SIGTERM', gracefulShutdown);

    // SIGUSR1: pause
    process.on('SIGUSR1', () => {
      daemon.pause('signal');
    });

    // SIGUSR2: resume
    process.on('SIGUSR2', () => {
      daemon.resume();
    });
  };

  // Only setup signal handlers if we're the main process
  if (process.argv[1]?.includes('sleep-watch')) {
    setupSignalHandlers();
  }

  await daemon.start();

  // Save checkpoint periodically
  const checkpointInterval = setInterval(() => {
    daemon.saveCheckpoint().catch(() => { /* non-fatal */ });
  }, 60_000); // Every minute

  // Clean up on stop
  daemon.onStateChange((state) => {
    if (state.status === 'stopped') {
      clearInterval(checkpointInterval);
    }
  });

  return daemon;
}
