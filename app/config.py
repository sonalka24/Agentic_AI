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
        clickhouse_host,
        clickhouse_port,
        clickhouse_user,
        clickhouse_password,
        clickhouse_db,
    ):
        """Initialize configuration values.

        @param minio_endpoint MinIO host:port endpoint.
        @param minio_secure Whether MinIO uses TLS.
        @param access_key MinIO access key.
        @param secret_key MinIO secret key.
        @param openai_api_key OpenAI API key.
        @param openai_model OpenAI model name.
        @param minio_bucket Source bucket name.
        @param minio_prefix Source prefix in bucket.
        @param local_dir Local download directory.
        @param clickhouse_host ClickHouse host.
        @param clickhouse_port ClickHouse HTTP port.
        @param clickhouse_user ClickHouse username.
        @param clickhouse_password ClickHouse password.
        @param clickhouse_db ClickHouse database name.
        @return None.
        """
        self.minio_endpoint = minio_endpoint
        self.minio_secure = minio_secure
        self.access_key = access_key
        self.secret_key = secret_key
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.minio_bucket = minio_bucket
        self.minio_prefix = minio_prefix
        self.local_dir = local_dir
        self.clickhouse_host = clickhouse_host
        self.clickhouse_port = clickhouse_port
        self.clickhouse_user = clickhouse_user
        self.clickhouse_password = clickhouse_password
        self.clickhouse_db = clickhouse_db



def load_config():
    """Load configuration from environment variables.

    @return Config Fully initialized configuration object.
    """
    return Config(
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
        minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        minio_bucket=os.getenv("MINIO_BUCKET", "plm"),
        minio_prefix=os.getenv("MINIO_PREFIX", "synthetic_data/"),
        local_dir="/tmp/synthetic_data",
        clickhouse_host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
        clickhouse_port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        clickhouse_user=os.getenv("CLICKHOUSE_USER", "admin"),
        clickhouse_password=os.getenv("CLICKHOUSE_PASSWORD", "admin"),
        clickhouse_db=os.getenv("CLICKHOUSE_DB", "plm"),
    )
