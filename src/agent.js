/**
 * Coding agent — Karpathy-style iterative worker.
 *
 * Conversational + autonomous. Answers questions naturally.
 * When given a task, enters an iterative loop:
 * edit -> test -> keep/revert -> report -> next.
 *
 * Language-agnostic: detects the project's language, test runner,
 * and build system automatically.
 *
 * Uses @anthropic-ai/sdk directly with custom coding tools.
 * Streams tool uses back to the host for Claude Code-style logging.
 */
import Anthropic from "@anthropic-ai/sdk";
import { createInterface } from "readline";
import { execSync } from "child_process";
import {
  readFileSync, writeFileSync, mkdirSync, existsSync,
  appendFileSync, readdirSync,
} from "fs";
import { join, resolve, dirname } from "path";

const WORKSPACE = process.env.WORKSPACE || process.cwd();
const MODEL = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-5-20250929";

const anthropic = new Anthropic();

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function log(text) {
  process.stderr.write(text + "\n");
}

// --- Conversational Memory ---

const MEMORY_DIR = "/home/user/data/memory";
const CONVERSATIONS_FILE = join(MEMORY_DIR, "conversations.jsonl");

function initMemory() {
  mkdirSync(MEMORY_DIR, { recursive: true });
  if (!existsSync(CONVERSATIONS_FILE)) writeFileSync(CONVERSATIONS_FILE, "");
}

function saveTurn(userMsg, assistantMsg) {
  const entry = { timestamp: new Date().toISOString(), user: userMsg, assistant: assistantMsg };
  appendFileSync(CONVERSATIONS_FILE, JSON.stringify(entry) + "\n");
}

function getRecentConversations(limit = 5) {
  try {
    const lines = readFileSync(CONVERSATIONS_FILE, "utf-8").trim().split("\n").filter(Boolean);
    return lines.slice(-limit).map((l) => JSON.parse(l));
  } catch { return []; }
}

function getContext() {
  const recent = getRecentConversations();
  if (!recent.length) return "";
  const parts = ["### Recent coding sessions"];
  for (const c of recent) {
    parts.push(`[${c.timestamp}] Task: ${c.user.slice(0, 150)}`);
    parts.push(`Result: ${c.assistant.slice(0, 200)}`);
  }
  return parts.join("\n");
}

// --- Shell execution ---

function shellExec(command, timeout = 120) {
  try {
    return execSync(command, {
      cwd: WORKSPACE,
      timeout: timeout * 1000,
      maxBuffer: 10 * 1024 * 1024,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    }) || "(no output)";
  } catch (err) {
    const out = [err.stdout, err.stderr].filter(Boolean).join("\n").trim();
    return `Exit code: ${err.status ?? "unknown"}${out ? "\n" + out : ""}`;
  }
}

// --- Tools ---

const tools = [
  {
    name: "execute",
    description: "Run a shell command in the workspace. Use for running tests, builds, git, installing packages, etc. Commands run with a 120-second timeout. Always append 2>&1 to capture both stdout and stderr.",
    input_schema: {
      type: "object",
      properties: {
        command: { type: "string", description: "Shell command to execute" },
      },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description: "Read the contents of a file. Always read a file before editing it.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to workspace root" },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to a file, creating parent directories if needed. Use for new files or complete rewrites.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to workspace root" },
        content: { type: "string", description: "Full file content to write" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "edit_file",
    description: "Make a surgical edit to a file by replacing an exact string match. The old_string must appear exactly once in the file. Include enough surrounding context to make the match unique.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to workspace root" },
        old_string: { type: "string", description: "Exact string to find (must be unique in the file)" },
        new_string: { type: "string", description: "Replacement string" },
      },
      required: ["path", "old_string", "new_string"],
    },
  },
  {
    name: "ls",
    description: "List directory contents with file type indicators (d = directory, - = file).",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Directory path relative to workspace root (default: '.')" },
      },
    },
  },
  {
    name: "glob",
    description: "Find files matching a name pattern. Excludes node_modules and .git directories.",
    input_schema: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "File name pattern (e.g. '*.py', '*.test.js', 'Makefile')" },
      },
      required: ["pattern"],
    },
  },
  {
    name: "grep",
    description: "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.",
    input_schema: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Search pattern (regex supported)" },
        path: { type: "string", description: "Directory or file to search in (default: '.')" },
      },
      required: ["pattern"],
    },
  },
];

const toolHandlers = {
  execute: ({ command }) => shellExec(command),

  read_file: ({ path }) => {
    try {
      return readFileSync(resolve(WORKSPACE, path), "utf-8");
    } catch (e) {
      return `Error: ${e.message}`;
    }
  },

  write_file: ({ path, content }) => {
    try {
      const fullPath = resolve(WORKSPACE, path);
      mkdirSync(dirname(fullPath), { recursive: true });
      writeFileSync(fullPath, content);
      return `Wrote ${content.length} bytes to ${path}`;
    } catch (e) {
      return `Error: ${e.message}`;
    }
  },

  edit_file: ({ path, old_string, new_string }) => {
    try {
      const fullPath = resolve(WORKSPACE, path);
      const content = readFileSync(fullPath, "utf-8");
      if (!content.includes(old_string)) return `Error: old_string not found in ${path}`;
      const count = content.split(old_string).length - 1;
      if (count > 1) return `Error: old_string matches ${count} times in ${path} — add more context to make it unique`;
      writeFileSync(fullPath, content.replace(old_string, new_string));
      return `Edited ${path}`;
    } catch (e) {
      return `Error: ${e.message}`;
    }
  },

  ls: ({ path } = {}) => {
    try {
      const fullPath = resolve(WORKSPACE, path || ".");
      const entries = readdirSync(fullPath, { withFileTypes: true });
      return entries.map((e) => `${e.isDirectory() ? "d" : "-"} ${e.name}`).join("\n") || "(empty)";
    } catch (e) {
      return `Error: ${e.message}`;
    }
  },

  glob: ({ pattern }) => {
    const escaped = pattern.replace(/'/g, "'\\''");
    return shellExec(`find . -name '${escaped}' -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | sort | head -100`);
  },

  grep: ({ pattern, path }) => {
    const escaped = pattern.replace(/'/g, "'\\''");
    const target = path || ".";
    return shellExec(`grep -rn '${escaped}' '${target}' --exclude-dir=node_modules --exclude-dir=.git 2>/dev/null | head -50`);
  },
};

// --- Tool activity summarizer ---

function summarizeTool(name, input) {
  if (!input) return name;
  if (name === "execute") return `$ ${(input.command || "").slice(0, 120)}`;
  if (name === "read_file") return `read ${input.path || ""}`;
  if (name === "write_file") return `write ${input.path || ""}`;
  if (name === "edit_file") return `edit ${input.path || ""}`;
  if (name === "ls") return `ls ${input.path || "."}`;
  if (name === "glob") return `glob ${input.pattern || ""}`;
  if (name === "grep") return `grep ${input.pattern || ""}`;
  return name;
}

// --- System Prompt ---

const SYSTEM_PROMPT = `You are a coding agent. You live in a workspace at ${WORKSPACE}.

## Personality
You are conversational. If the user asks a question, answer it. If they ask you to look at something, look and report back. When given a task (optimize, fix bugs, refactor, add features), enter the work loop below.

## Work Loop
When given a coding task, work like an autonomous researcher:

1. **Explore** — ls, read_file, glob, grep to understand the project structure, language, and tooling.
2. **Detect** — Figure out the language, test runner, build system, and how to verify changes:
   - Python: \`python -m pytest -v 2>&1\`, \`python -m unittest discover -v 2>&1\`
   - Node.js: \`npm test 2>&1\`, \`npx jest --verbose 2>&1\`, \`npx vitest run 2>&1\`
   - Go: \`go test ./... -v 2>&1\`
   - Rust: \`cargo test 2>&1\`
   - Or whatever the project uses — check package.json scripts, Makefile, CI config, etc.
3. **Baseline** — Run the tests to see what passes and what fails.
4. **LOOP** — For each change:
   a. Make ONE focused change (one function, one bug, one improvement).
   b. Run the tests.
   c. If tests PASS → \`git add -A && git commit -m "description"\`
   d. If tests FAIL → \`git checkout -- .\` and try a different approach.
   e. Report what you changed and the result.
   f. Move to the next issue — don't stop, don't ask.

## Rules
- ONE change per iteration. Never batch unrelated changes.
- Always run tests after each change. If they fail, revert immediately.
- Read files before editing them. Use edit_file for surgical changes.
- When done, give a summary of everything you changed and the results.
- Git is available. Commit after each successful change so reverts are clean.
- NEVER STOP to ask "should I continue?" — keep going until the task is done.
- If you run out of ideas or everything passes, summarize and stop.`;

// --- Agentic loop ---

async function handleMessage(content, messageId) {
  const memoryContext = getContext();
  const enrichedContent = memoryContext
    ? `## Memory context from past sessions\n${memoryContext}\n\n## User message\n${content}`
    : content;

  const messages = [{ role: "user", content: enrichedContent }];

  while (true) {
    const resp = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 200000,
      system: SYSTEM_PROMPT,
      tools,
      messages,
    });

    // Emit activity for each tool use
    for (const block of resp.content) {
      if (block.type === "tool_use") {
        send({
          type: "activity",
          tool: block.name,
          description: summarizeTool(block.name, block.input),
          message_id: messageId,
        });
      }
    }

    // If no tool use, return final text
    if (resp.stop_reason === "end_turn" || !resp.content.some((b) => b.type === "tool_use")) {
      const result = resp.content.filter((b) => b.type === "text").map((b) => b.text).join("") || "";
      saveTurn(content, result);
      return result;
    }

    // Process tool calls
    messages.push({ role: "assistant", content: resp.content });

    const toolResults = [];
    for (const block of resp.content) {
      if (block.type !== "tool_use") continue;
      const handler = toolHandlers[block.name];
      let result;
      try {
        result = handler ? await handler(block.input) : `Error: unknown tool ${block.name}`;
      } catch (e) {
        result = `Error: ${e.message}`;
      }
      toolResults.push({ type: "tool_result", tool_use_id: block.id, content: result });
    }
    messages.push({ role: "user", content: toolResults });
  }
}

// --- Primordial Protocol ---

function main() {
  initMemory();
  send({ type: "ready" });
  log("Coding Agent ready");

  const rl = createInterface({ input: process.stdin, terminal: false });

  rl.on("line", async (line) => {
    line = line.trim();
    if (!line) return;

    let msg;
    try { msg = JSON.parse(line); } catch { return; }

    if (msg.type === "shutdown") {
      log("Shutting down");
      rl.close();
      return;
    }

    if (msg.type === "message") {
      const mid = msg.message_id;
      try {
        const result = await handleMessage(msg.content, mid);
        send({ type: "response", content: result, message_id: mid, done: true });
      } catch (e) {
        log(`Error: ${e.message}`);
        send({ type: "error", error: e.message, message_id: mid });
        send({ type: "response", content: `Something went wrong: ${e.message}`, message_id: mid, done: true });
      }
    }
  });
}

main();
