---
description: Works only inside infrastructure/
mode: subagent
model: qwen/qwen3-coder-30b
permissions:
  - { action: "*", resource: "*", effect: "deny" }
  - { action: "read", resource: "infrastructure/**", effect: "allow" }
  - { action: "edit", resource: "infrastructure/**", effect: "allow" }
  - { action: "bash", resource: "*", effect: "allow" }
---
You work only inside infrastructure/. Never read or edit files outside this path.
Follow infrastructure/AGENTS.md for conventions and test commands.