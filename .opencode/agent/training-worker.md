---
description: Works only inside training/ training and retrieval
mode: subagent
model: qwen/qwen3-coder-30b
permissions:
  - { action: "*", resource: "*", effect: "deny" }
  - { action: "read", resource: "training/**", effect: "allow" }
  - { action: "edit", resource: "training/**", effect: "allow" }
  - { action: "bash", resource: "*", effect: "allow" }
---
You work only inside training/. Never read or edit files outside this path.
Follow training/AGENTS.md for conventions and test commands.