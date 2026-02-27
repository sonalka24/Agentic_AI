# Agentic AI — PLM Migration (Agent AI)

An end‑to‑end PLM migration pipeline that pulls product data from MinIO, extracts structured facts with a LangGraph agent, and ingests results into ClickHouse. Metabase is included for quick exploration.


## Technical Documentation

### **See the Techical Documentation for Agentic AI here :**

`https://sonalka24.github.io/Agentic_AI/`


## What’s Inside

- Python agentic AI development code in `codebase/`
- ClickHouse for analytics storage
- MinIO as the data lake
- Metabase for dashboards
- Docker Compose for local orchestration

## Project Layout

- `codebase/` Python source for developing agents and tools
- `environment_setup/` Code for setting up the enviornment.

## Notes

- ClickHouse is initialized to create the `plm` database only.
- For setting up the environment, see `environment_setup/README.md`.
- Details about the codebase, see `codebase/README.md`.
