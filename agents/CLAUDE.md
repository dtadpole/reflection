# Agent Definitions

Agents are organized as `agents/<agent_name>/<variant>/`. The initial variant is `base`.

Each variant folder contains these files:

## Required Files

- **agent.md** — Agent definition with sections:
  - `# Agent Name` — title
  - `## Description` — one-line (used as AgentDefinition.description)
  - `## System Prompt` — full prompt (becomes AgentDefinition.prompt)
  - `## Input Format` — what the agent receives
  - `## Output Format` — what the agent must produce
  - `## Examples` — few-shot examples

- **config.yaml** — Agent configuration:
  ```yaml
  model: sonnet                       # sonnet, opus, haiku
  temperature: 0.7
  max_turns: 10
  tools: [Read, Bash]                # Built-in Claude tools
  custom_tools: [code_executor]      # agenix custom tools
  ```

## Optional Files

- **tools.md** — Detailed tool specifications
- **logic.py** — Hardened Python logic (parsing, validation, helpers)

## Conventions

- Keep prompts focused and specific
- Use structured I/O formats (JSON) for pipeline compatibility
- Move stabilized prompt patterns to logic.py
- Each agent specifies its own model — use cheaper models for simpler tasks
