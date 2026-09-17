---
description: Works only inside serving/
mode: subagent
model: qwen/qwen3-coder-30b
permissions:
  - { action: "*", resource: "*", effect: "deny" }
  - { action: "read", resource: "serving/**", effect: "allow" }
  - { action: "edit", resource: "serving/**", effect: "allow" }
  - { action: "bash", resource: "*", effect: "allow" }
---
You work only inside serving/. Never read or edit files outside this path.
Follow serving/AGENTS.md for conventions and test commands.