# Agentic AI — PLM Migration (Agent AI)

An end‑to‑end PLM migration pipeline that pulls product data from MinIO, extracts structured facts with a LangGraph agent, and ingests results into ClickHouse. Metabase is included for quick exploration.

## What’s Inside

- Python agent pipeline in `app/`
- ClickHouse for analytics storage
- MinIO as the data lake
- Metabase for dashboards
- Docker Compose for local orchestration

## Quick Start (Local)

1. `cp .env.example .env` (if you have one) or create `.env`
2. `docker compose up -d --build`
3. `docker compose exec agent python main.py`

If you only want the services:

1. `docker compose up -d clickhouse minio metabase`

## Services

- ClickHouse HTTP: `http://localhost:8123`
- ClickHouse Native: `localhost:9000`
- MinIO: `http://localhost:9001` (console), `http://localhost:9002` (API)
- Metabase: `http://localhost:3000`

## Environment Variables

These are the primary knobs used by the app and containers:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `MINIO_ENDPOINT` (default `minio:9000`)
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET` (default `plm`)
- `MINIO_PREFIX` (default `synthetic_data/`)
- `CLICKHOUSE_HOST` (default `clickhouse`)
- `CLICKHOUSE_PORT` (default `8123`)
- `CLICKHOUSE_USER`
- `CLICKHOUSE_PASSWORD`
- `CLICKHOUSE_DB` (default `plm`)

## Data Flow

1. Data is seeded into MinIO at startup.
2. The agent downloads Excel files from MinIO.
3. The agent extracts structured facts and images.
4. Results are ingested into ClickHouse.

## Project Layout

- `app/` Python source for agents and tools
- `Dockerfiles/` service Dockerfiles
- `scripts/` container entrypoints
- `synthetic_data/` sample input data
- `compose.yaml` local orchestration
- `clickhouse-init.sql` creates the `plm` database

## Notes

- ClickHouse is initialized to create the `plm` database only.
- The agent uses `app/schema.json` to guide extraction.

## Troubleshooting

- If MinIO seeding looks empty, restart MinIO and check `scripts/minio-entrypoint.sh`.
- If ClickHouse auth fails, verify `CLICKHOUSE_USER` and `CLICKHOUSE_PASSWORD` in `.env`.
