import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));

describe("package lifecycle (GitHub install builds dist/)", () => {
  it("declares a prepare script that runs the build", () => {
    expect(pkg.scripts.prepare).toBe("npm run build");
  });

  it("keeps the compiled bin entry points after build", () => {
    expect(pkg.bin["agent-memory-mcp"]).toBe("dist/index.js");
    expect(existsSync(path.join(root, pkg.bin["agent-memory-mcp"]))).toBe(true);
  });

  it("exposes the compiled reembed CLI entry", () => {
    expect(existsSync(path.join(root, "dist/cli/reembed-cli.js"))).toBe(true);
  });
});
