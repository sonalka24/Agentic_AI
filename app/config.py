import os


class Config:
    def __init__(
        self,
        minio_endpoint,
        minio_secure,
        access_key,
        secret_key,
        openai_api_key,
        openai_model,
        minio_bucket,
        minio_prefix,
        local_dir,
    ):
        self.minio_endpoint = minio_endpoint
        self.minio_secure = minio_secure
        self.access_key = access_key
        self.secret_key = secret_key
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.minio_bucket = minio_bucket
        self.minio_prefix = minio_prefix
        self.local_dir = local_dir



def load_config():
    return Config(
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
        minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        minio_bucket=os.getenv("MINIO_BUCKET", "plm"),
        minio_prefix=os.getenv("MINIO_PREFIX", "synthetic_data/"),
        local_dir="./synthetic_data",
    )
