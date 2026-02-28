# App — Agent Pipeline

This folder contains the LangGraph agent and supporting tools that extract product data and ingest into ClickHouse.

## Run Locally (Inside Container)

1. `docker compose exec agent python main.py`

## Key Files

- `main.py` entrypoint to run the end‑to‑end pipeline
- `extract_data_agent.py` LangGraph workflow
- `tools.py` tool implementations (Excel parsing, translation, ClickHouse ingest)
- `schema.json` extraction schema used by the agent
- `prompts.json` LLM prompts

## Configuration

The app reads configuration from environment variables. See repo‑root `README.md` for the full list.

