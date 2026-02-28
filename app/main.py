from minio import Minio
from urllib3.exceptions import HTTPError

from extract_data_agent import build_agent
from config import load_config
from tools import Toolset


def main():
    """Run end-to-end extraction + ingestion workflow.

    @return None.
    @raises RuntimeError If MinIO connectivity or pipeline execution fails.
    """
    config = load_config()
    client = Minio(
        config.minio_endpoint,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.minio_secure,
    )

    try:
        client.bucket_exists(config.minio_bucket)
    except HTTPError as exc:
        raise RuntimeError("MinIO connection failed.") from exc

    tools = Toolset(config=config, minio_client=client).registry()
    agent = build_agent(config=config, tools=tools)

    result = agent.invoke(
        {
            "bucket": config.minio_bucket,
            "prefix": config.minio_prefix,
            "local_dir": config.local_dir,
        }
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    ingest_images_result = result.get("ingest_images_result", {})
    print(
        f"Ingested to ClickHouse warehouse: table={ingest_images_result.get('table')} "
        f"rows={ingest_images_result.get('inserted_rows')} run_id={ingest_images_result.get('run_id')}"
    )

    ingest_result = result.get("ingest_result", {})
    print(
        f"Ingested to ClickHouse warehouse: table={ingest_result.get('table')} "
        f"rows={ingest_result.get('inserted_rows')} run_id={ingest_result.get('run_id')}"
    )


if __name__ == "__main__":
    main()
