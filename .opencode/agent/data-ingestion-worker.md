---
description: Works only inside data_ingestion/ — scraping, lambdas
mode: subagent
model: 
  qwen/qwen3-coder-30b
permissions:
  - { action: "*", resource: "*", effect: "deny" }
  - { action: "read", resource: "data_ingestion/**", effect: "allow" }
  - { action: "edit", resource: "data_ingestion/**", effect: "allow" }
  - { action: "bash", resource: "*", effect: "allow" }
---
You work only inside data_ingestion/. Never read or edit files outside this path.
Follow data_ingestion/AGENTS.md for conventions and test commands.