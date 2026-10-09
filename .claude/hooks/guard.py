#!/usr/bin/env python3
"""PreToolUse guard for Claude Code (works in auto mode; hooks run before the classifier).

- Blocks any Bash command that references a .env file (read, write, delete, source...)
- Blocks `git push` that targets a protected branch, or runs while on one
- Asks for approval before Edit/Write/NotebookEdit touches a path outside the project

Fails closed: any internal error blocks the call.
Install at ~/.claude/hooks/guard.py
"""
import json
import os
import re
import subprocess
import sys

PROTECTED_BRANCHES = {"main", "master"}  # add "develop", "production", ... if needed

# .env, foo.env, .env.local, .env.production ... but not .envrc / .environment
ENV_RE = re.compile(r"\.env(\.[A-Za-z0-9_-]+)?(?![\w.-])")


def deny(msg):
    print(f"Blocked by guard.py: {msg}", file=sys.stderr)
    sys.exit(2)


def ask(msg):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": msg,
    }}))
    sys.exit(0)


def current_branch(cwd):
    try:
        r = subprocess.run(["git", "-C", cwd, "symbolic-ref", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def check_file_tool(inp, cwd, project):
    raw = inp.get("file_path") or inp.get("notebook_path")
    if not raw:
        return
    path = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(cwd, raw))
    if not inside(path, project):
        ask(f"{path} is outside the project directory ({project})")


def check_bash(cmd, cwd):
    if ENV_RE.search(cmd):
        deny("command references a .env file")

    if re.search(r"\bgit\b", cmd) and re.search(r"\bpush\b", cmd):
        tokens = {t.lstrip("+") for t in re.split(r"[\s'\"=:/\\;&|()]+", cmd) if t}
        hit = PROTECTED_BRANCHES & tokens
        if hit:
            deny(f"push targets protected branch {sorted(hit)[0]}; push a feature branch instead")
        if tokens & {"--all", "--mirror"}:
            deny("--all/--mirror pushes can update protected branches")
        branch = current_branch(cwd)
        if branch in PROTECTED_BRANCHES:
            deny(f"currently on {branch}; create a feature branch first")


def main():
    d = json.load(sys.stdin)
    tool = d.get("tool_name", "")
    inp = d.get("tool_input") or {}
    cwd = d.get("cwd") or os.getcwd()
    project = os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    if tool == "Bash":
        check_bash(inp.get("command", ""), cwd)
    elif tool in ("Edit", "Write", "NotebookEdit"):
        check_file_tool(inp, cwd, project)


try:
    main()
except Exception as e:  # SystemExit is not an Exception, so deny()/ask() pass through
    deny(f"internal error ({e}); failing closed")