# Codebase — Agent Pipeline

## Technical Documentation

### **See the Techical Documentation for Agentic AI here :**

`https://sonalka24.github.io/Agentic_AI/`

This folder contains the LangGraph agent and supporting tools that extract product data and ingest into ClickHouse.

## Run Locally (Inside Container)

1. `docker compose exec agent python3 -m extract_data_agent`


## Key Files

- `extract_data_agent.py` LangGraph workflow and main entrypoint for the end-to-end pipeline
- `main.py` compatibility wrapper that forwards to `extract_data_agent.py`
- `tools.py` tool implementations (Excel parsing, translation, ClickHouse ingest)
- `extract_data_agent/json/schema.json` extraction schema used by the agent
- `extract_data_agent/json/prompts.json` LLM prompts

## Configuration

The codebase reads configuration from environment variables. See repo‑root `README.md` for the full list.
