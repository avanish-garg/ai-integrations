# Temporal AI Integrations

Plugins that connect AI agent frameworks and SDKs to [Temporal](https://temporal.io) durable
execution. Each plugin is its own package with its own dependencies, tests, version and release
cadence, laid out as `<language>/<integration>/`.

| Plugin | Package | Root API | Maturity |
|---|---|---|---|
| [`python/deepagents`](python/deepagents) | `temporalio-deepagents` (migrating) | `temporalio.contrib.deepagents` | Experimental |
| [`python/google_adk`](python/google_adk) | `temporalio-google-adk` (migrating) | `temporalio.contrib.google_adk_agents` | Preview |
| [`python/google_genai`](python/google_genai) | `temporalio-google-genai` (migrating) | `temporalio.contrib.google_genai` | Experimental |
| [`python/langgraph`](python/langgraph) | `temporalio-langgraph` (migrating) | `temporalio.contrib.langgraph` | Experimental |
| [`python/langsmith`](python/langsmith) | `temporalio-langsmith` (migrating) | `temporalio.contrib.langsmith` | Experimental |
| [`python/mcp`](python/mcp) | [`temporalio-mcp`](https://pypi.org/project/temporalio-mcp/) | `temporalio.mcp` | Experimental |
| [`python/openai_agents`](python/openai_agents) | [`temporalio-openai-agents`](https://pypi.org/project/temporalio-openai-agents/) | `temporalio.openai_agents` | GA |
| [`python/strands_agents`](python/strands_agents) | `temporalio-strands-agents` (migrating) | `temporalio.contrib.strands` | Experimental |

More plugins are migrating here from the SDK repositories; see the target table in
[`AGENTS.md`](AGENTS.md).

## Install

```
$ uv add temporalio-mcp
$ uv add temporalio-openai-agents
```

## Develop

```
$ cd python/openai_agents
$ make sync    # non-editable install into .venv (see AGENTS.md for why)
$ make lint
$ make test    # provider calls use deterministic local models and transports
```

`make help` lists every target. Conventions, CI design, release process and migration procedure
are in [`AGENTS.md`](AGENTS.md); contributor workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[MIT](LICENSE). Each plugin directory carries an identical copy so every published package ships the license text.
