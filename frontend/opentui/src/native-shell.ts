import { AclSandbox, tempWriteSid, workspaceWriteSid } from "@deepseek-ai/dsh-sandbox-windows-acl";
import { realpathSync } from "node:fs";
import { isAbsolute, relative, sep } from "node:path";

const [policyJson, command, ...args] = process.argv.slice(2);
let sandbox: AclSandbox | undefined;
try {
  const policy = JSON.parse(policyJson!);
  const workspace = realpathSync.native(policy.cwd);
  const roots: string[] = policy.write_paths.map((path: string) => realpathSync.native(path));
  for (const root of roots) {
    const child = relative(workspace, root);
    if (isAbsolute(child) || child === ".." || child.startsWith(`..${sep}`)) {
      throw new Error("Writable directories must remain inside the workspace");
    }
  }
  const temporary = realpathSync.native(policy.temp);
  // Grants are standing ACL edits, so each directory keeps one stable SID:
  // any combination of directories reuses the same ACEs, and a token only
  // carries the SIDs of the directories this command declared.
  sandbox = new AclSandbox({
    writableDirs: roots,
    tempDir: temporary,
    writeSidsByDir: roots.map(root => workspaceWriteSid(`veriforge:write-path:${root.toLowerCase()}`)),
    tempWriteSid: tempWriteSid(temporary),
    mode: "workspace-write",
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
