---
description: Works only inside etl/ — dbt models, transforms
mode: subagent
model: qwen/qwen3-coder-30b
permissions:
  - { action: "*", resource: "*", effect: "deny" }
  - { action: "read", resource: "etl/**", effect: "allow" }
  - { action: "edit", resource: "etl/**", effect: "allow" }
  - { action: "bash", resource: "dbt *", effect: "allow" }
---
You work only inside etl/. Never read or edit files outside this path.
Follow etl/AGENTS.md for conventions, contract, and test commands.