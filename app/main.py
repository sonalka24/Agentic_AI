import pandas as pd
from minio import Minio
from urllib3.exceptions import HTTPError

from agent_graph import build_agent
from config import load_config
from tools import Toolset


def main():
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

    print(result.get("message", "Run complete."))
    print(f"OpenAI configured: {result.get('openai_configured')} | model={config.openai_model}")
    print(f"Mode: {result.get('execution_mode')}")
    print(f"Schema sections: {len(result.get('common_schema', {}))}")

    product_rows = result.get("product_table_rows", [])
    print(f"Tabular rows: {len(product_rows)}")
    if product_rows:
        preview_df = pd.DataFrame(product_rows)
        print(preview_df[["product_id", "section", "key", "value"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
