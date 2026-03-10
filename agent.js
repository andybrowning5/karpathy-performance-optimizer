/**
 * Performance Optimizer — autonomous code performance agent (Deep Agent, Node.js).
 *
 * Uses deepagentsjs with LocalShellBackend to:
 * 1. Explore a codebase and run baseline benchmarks
 * 2. Identify performance bottlenecks (O(n^2), redundant work, etc.)
 * 3. Implement optimizations one at a time
 * 4. Verify with tests — keep if they pass, revert if they don't
 *
 * Runs via the Primordial NDJSON protocol (stdin/stdout) or standalone.
 */
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { createInterface } from "readline";

const WORKSPACE = process.env.WORKSPACE || process.cwd();

const SYSTEM_PROMPT = `You are the Performance Optimizer, an autonomous agent that improves code performance.

## Your Mission
Given a codebase with performance tests, find and fix performance bottlenecks.
Make the failing performance tests pass while keeping all correctness tests green.

## Workflow

1. **Explore** — Use ls, read_file, glob, grep to understand the codebase structure.
2. **Baseline** — Run the test suite with \`execute\` to see which tests pass/fail.
   Use: \`python -m pytest -v 2>&1\` to run all tests.
3. **Analyze** — Read the failing performance tests to understand what they measure.
   Read the source code to identify the algorithmic bottleneck.
4. **Plan** — Use write_todos to track which functions need optimization and what algorithm to use.
5. **For each optimization:**
   a. Read the function you want to optimize.
   b. Implement ONE focused optimization (e.g., replace O(n^2) with O(n) using a set/dict).
   c. Run correctness tests: \`python -m pytest test_correctness.py -v 2>&1\`
   d. If correctness passes, run performance tests: \`python -m pytest test_performance.py -v 2>&1\`
   e. If BOTH pass: commit with \`git add -A && git commit -m "opt: description"\`.
   f. If EITHER fails: revert with \`git checkout -- .\` and try a different approach.
   g. Update your todos with the result.

## Rules
- Fix ONE function per iteration. Don't batch changes.
- Always run correctness tests BEFORE performance tests.
- If a correctness test fails, your optimization is WRONG — revert immediately.
- Use edit_file for surgical changes (preferred). Only use write_file for new files.
- Always read a file before editing it.
- Common optimizations:
  - O(n^2) membership check → use a set for O(1) lookups
  - O(n^2) counting → use a dict/Counter for O(n) counting
  - Repeated list concatenation → use list.append() or list.extend()
  - Recomputing sums → use sliding window
  - Sorting when merging → use two-pointer merge
- When all performance tests pass, summarize what you optimized and the speedups.
- If a function is already fast enough, skip it.

## Important
- The workspace is at ${WORKSPACE}
- Python is available via \`python\` or \`python3\`
- Use \`git\` to track changes and enable easy reverts`;

// --- NDJSON Protocol ---

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function waitForMessage() {
  return new Promise((resolve) => {
    let resolved = false;
    const rl = createInterface({ input: process.stdin });
    rl.on("line", (line) => {
      line = line.trim();
      if (!line || resolved) return;
      try {
        const msg = JSON.parse(line);
        resolved = true;
        rl.close();
        resolve(msg.type === "shutdown" ? null : msg);
      } catch {
        // ignore malformed lines
      }
    });
    rl.on("close", () => {
      if (!resolved) resolve(null);
    });
  });
}

function parseObjective(content) {
  let objective = content;
  let verifyCmd = "python -m pytest -v 2>&1";

  if (content.includes("|")) {
    const [obj, rest] = content.split("|", 2);
    objective = obj.trim();
    const verify = rest.trim();
    verifyCmd = verify.toLowerCase().startsWith("verify:")
      ? verify.slice(7).trim()
      : verify;
  }

  return { objective, verifyCmd };
}

async function runAgent(objective, verifyCmd, onActivity) {
  if (onActivity) onActivity("Creating backend...");

  const backend = await LocalShellBackend.create({
    rootDir: WORKSPACE,
    virtualMode: false,
    inheritEnv: true,
    timeout: 120,
  });

  if (onActivity) onActivity("Creating deep agent...");

  const agent = createDeepAgent({
    model: new ChatAnthropic({
      model: "claude-opus-4-6",
      temperature: 0,
    }),
    systemPrompt: SYSTEM_PROMPT,
    backend,
  });

  const task = [
    `## Objective`,
    objective,
    ``,
    `## Verify Command`,
    `\`${verifyCmd}\``,
    ``,
    `## Workspace`,
    WORKSPACE,
    ``,
    `Start by exploring the workspace and running the full test suite to see`,
    `the baseline. Then optimize each failing function one at a time, verifying`,
    `after each change.`,
  ].join("\n");

  if (onActivity) onActivity("Invoking agent (this takes several minutes)...");

  // Use invoke with a heartbeat timer to keep the connection alive
  let heartbeatCount = 0;
  const heartbeat = setInterval(() => {
    heartbeatCount++;
    if (onActivity) onActivity(`Agent working... (${heartbeatCount * 30}s)`);
  }, 30_000);

  try {
    const result = await agent.invoke(
      { messages: [new HumanMessage(task)] },
      { recursionLimit: 200 }
    );

    const messages = result.messages || [];
    if (messages.length > 0) {
      const last = messages[messages.length - 1];
      if (last.content) {
        return typeof last.content === "string"
          ? last.content
          : JSON.stringify(last.content);
      }
    }
    return "Agent completed with no final message.";
  } finally {
    clearInterval(heartbeat);
  }
}

// --- Entry Points ---

async function runPrimordial() {
  send({ type: "ready" });

  const msg = await waitForMessage();
  if (!msg) process.exit(0);

  const mid = msg.message_id || "";
  const content = msg.content || "";
  const { objective, verifyCmd } = parseObjective(content);

  send({
    type: "activity",
    tool: "setup",
    description: `Starting performance optimization: ${objective}`,
  });

  try {
    const result = await runAgent(objective, verifyCmd, (desc) => {
      send({ type: "activity", tool: "agent", description: desc });
    });

    send({
      type: "response",
      content: result,
      message_id: mid,
      done: true,
    });
  } catch (err) {
    send({
      type: "error",
      error: `Agent failed: ${err.message || err}`,
      message_id: mid,
    });
  }
}

async function runStandalone() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log(
      "Usage: node agent.js 'optimize performance | verify: python -m pytest -v'"
    );
    process.exit(1);
  }

  const content = args.join(" ");
  const { objective, verifyCmd } = parseObjective(content);
  const result = await runAgent(objective, verifyCmd, (desc) => {
    console.error(`  > ${desc}`);
  });
  console.log("\n=== Result ===\n");
  console.log(result);
}

// Detect mode
if (process.argv.length > 2) {
  runStandalone().catch(console.error);
} else if (process.stdin.isTTY) {
  console.log("Performance Optimizer — Deep Agent (Node.js)");
  console.log(
    "Usage: node agent.js 'optimize performance | verify: python -m pytest -v'"
  );
  console.log("   Or pipe via Primordial NDJSON protocol.");
} else {
  runPrimordial().catch((err) => {
    send({ type: "error", error: `Fatal: ${err.message || err}` });
    process.exit(1);
  });
}
