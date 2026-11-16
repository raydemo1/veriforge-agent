import { AclSandbox, tempWriteSid, workspaceWriteSid } from "@deepseek-ai/dsh-sandbox-windows-acl";
import { realpathSync } from "node:fs";
import { isAbsolute, relative, sep } from "node:path";

const [policyJson, command, ...args] = process.argv.slice(2);
let sandbox: AclSandbox | undefined;
try {
  const policy = JSON.parse(policyJson!);
  const workspace = realpathSync.native(policy.cwd);
  const mode = policy.mode === "read-only" ? "read-only" : "workspace-write";
  const declaredRoots: string[] = policy.write_paths ?? [];
  const declaredExtraRoots: string[] = policy.extra_write_paths ?? [];
  if (mode === "read-only" && (declaredRoots.length > 0 || declaredExtraRoots.length > 0)) {
    throw new Error("Read-only mode cannot carry write grants");
  }
  const roots: string[] = declaredRoots.map((path: string) => realpathSync.native(path));
  for (const root of roots) {
    const child = relative(workspace, root);
    if (isAbsolute(child) || child === ".." || child.startsWith(`..${sep}`)) {
      throw new Error("Writable directories must remain inside the workspace");
    }
  }
  // Extra directories are a per-call capability that the permission policy
  // validated and the approver granted; they are not workspace write_paths.
  const extraRoots: string[] = declaredExtraRoots
    .map((path: string) => realpathSync.native(path));
  for (const root of extraRoots) {
    if (!isAbsolute(root)) {
      throw new Error("Additional writable directories must be absolute");
    }
  }
  for (const path of declaredExtraRoots) {
    if (realpathSync.native(path) !== path) {
      throw new Error("Approved write path changed after approval");
    }
  }
  const granted = [...new Set([...roots, ...extraRoots])];
  const temporary = mode === "workspace-write" ? realpathSync.native(policy.temp) : undefined;
  // Grants are standing ACL edits, so each directory keeps one stable SID:
  // any combination of directories reuses the same ACEs, and a token only
  // carries the SIDs of the directories this command declared.
  sandbox = new AclSandbox({
    writableDirs: granted,
    tempDir: temporary ?? null,
    writeSidsByDir: mode === "workspace-write"
      ? granted.map(root => workspaceWriteSid(`veriforge:write-path:${root.toLowerCase()}`))
      : undefined,
    tempWriteSid: temporary === undefined ? undefined : tempWriteSid(temporary),
    mode,
  });
  await sandbox.init();
  const child = sandbox.spawn({ command: command!, args, cwd: workspace, stdio: "inherit" });
  process.exitCode = (await child.wait()).exitCode;
} catch (error) {
  console.error(`native-sandbox: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 127;
} finally {
  try {
    sandbox?.dispose();
  } catch (error) {
    console.error(`native-sandbox: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 127;
  }
}
