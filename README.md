# Temporal AI Integrations

Plugins that connect AI agent frameworks and SDKs to [Temporal](https://temporal.io) durable
execution. Each plugin is its own package with its own dependencies, tests, version and release
cadence, laid out as `<language>/<integration>/`.

| Plugin | Package | Root API | Maturity |
|---|---|---|---|
| [`python/deepagents`](python/deepagents) | `temporalio-deepagents` | `temporalio.deepagents` | Experimental |
| [`python/google_adk`](python/google_adk) | `temporalio-google-adk` | `temporalio.google_adk` | Preview |
| [`python/google_genai`](python/google_genai) | `temporalio-google-genai` | `temporalio.google_genai` | Experimental |
| [`python/langgraph`](python/langgraph) | `temporalio-langgraph` | `temporalio.langgraph` | Experimental |
| [`python/langsmith`](python/langsmith) | `temporalio-langsmith` | `temporalio.langsmith` | Experimental |
| [`python/mcp`](python/mcp) | [`temporalio-mcp`](https://pypi.org/project/temporalio-mcp/) | `temporalio.mcp` | Experimental |
| [`python/openai_agents`](python/openai_agents) | [`temporalio-openai-agents`](https://pypi.org/project/temporalio-openai-agents/) | `temporalio.openai_agents` | GA |
| [`python/strands_agents`](python/strands_agents) | `temporalio-strands-agents` | `temporalio.strands_agents` | Experimental |

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
