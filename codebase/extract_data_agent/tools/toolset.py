from minio import Minio
from urllib3.exceptions import HTTPError

from config import config

from .file_tools import FileToolsMixin
from .helpers import SharedHelpersMixin
from .ingest_tools import IngestToolsMixin
from .llm_tools import LlmToolsMixin
from .prompts import load_prompts
from .registry import RegistryMixin


class Toolset(SharedHelpersMixin, FileToolsMixin, LlmToolsMixin, IngestToolsMixin, RegistryMixin):
    """Collection of data + LLM tools used by the LangGraph agent."""

    def __init__(self, minio_client):
        self.client = minio_client
        self.prompts = load_prompts()

    @classmethod
    def from_config(cls):
        client = Minio(config.minio_endpoint, access_key=config.access_key, secret_key=config.secret_key, secure=config.minio_secure)
        try:
            client.bucket_exists(config.minio_bucket)
        except HTTPError as exc:
            raise RuntimeError("MinIO connection failed.") from exc
        return cls(minio_client=client)

    def downloader_from_datalake(self, bucket, prefix, local_dir):
        """Download objects from MinIO into a local directory.

        Args:
            bucket: MinIO bucket name to read from.
            prefix: Object prefix inside the bucket.
            local_dir: Local directory where files should be downloaded.
        """
        return self.downloader_from_datalake_tool({"bucket": bucket, "prefix": prefix, "local_dir": local_dir})

    def read_excel_rows_text(self, excel_files):
        """Read Excel files into normalized text grouped by file and sheet.

        Args:
            excel_files: Downloaded Excel file paths to parse.
        """
        return self.read_excel_rows_text_tool({"excel_files": excel_files})

    def extract_excel_images_by_section(self, excel_files, target_sections, output_dir=None):
        """Extract Excel images and assign them to top-level sections using row proximity.

        Args:
            excel_files: Excel file paths containing embedded images.
            target_sections: Ordered section names used to map images.
            output_dir: Optional output directory for extracted images.
        """
        payload = {"excel_files": excel_files, "target_sections": target_sections}
        if output_dir is not None:
            payload["output_dir"] = output_dir
        return self.extract_excel_images_by_section_tool(payload)

    def translate_chinese_text_llm(self, excel_rows_text):
        """Translate Chinese spreadsheet text to English while preserving workbook structure.

        Args:
            excel_rows_text: Parsed spreadsheet text grouped by file and sheet.
        """
        return self.translate_chinese_text_llm_tool({"excel_rows_text": excel_rows_text})

    def clickhouse_table_count(self, table="product_facts"):
        """Return the row count for a ClickHouse table in the configured database.

        Args:
            table: Target ClickHouse table name in the configured database.
        """
        return self.clickhouse_table_count_tool({"table": table})

    def arrange_extract_excel_images_llm(self, excel_files, target_section_schema, products_dict=None, output_dir=None):
        """Assign extracted Excel images to section and subsection buckets.

        Args:
            excel_files: Excel file paths containing embedded images.
            target_section_schema: Section and subsection schema used for image assignment.
            products_dict: Product facts keyed by product id.
            output_dir: Optional output directory for arranged images.
        """
        payload = {"excel_files": excel_files, "target_section_schema": target_section_schema, "products_dict": products_dict or {}}
        if output_dir is not None:
            payload["output_dir"] = output_dir
        return self.arrange_extract_excel_images_llm_tool(payload)

    def fill_schema_values_llm(self, excel_rows_text, target_sections, target_section_schema):
        """Fill the target schema from spreadsheet text and return normalized product facts.

        Args:
            excel_rows_text: Parsed spreadsheet text grouped by file and sheet.
            target_sections: Target section names to extract.
            target_section_schema: Target section, subsection, and key schema.
        """
        return self.fill_schema_values_llm_tool({"excel_rows_text": excel_rows_text, "target_sections": target_sections, "target_section_schema": target_section_schema})

    def ingest_product_facts_clickhouse(self, product_table_rows):
        """Create or update the product facts table and ingest normalized fact rows into ClickHouse.

        Args:
            product_table_rows: Normalized fact rows ready for ClickHouse ingestion.
        """
        return self.ingest_product_facts_clickhouse_tool({"product_table_rows": product_table_rows})

    def ingest_product_images_clickhouse(self, product_images):
        """Create or update the product images table and ingest prepared image rows into ClickHouse.

        Args:
            product_images: Prepared image rows ready for ClickHouse ingestion.
        """
        return self.ingest_product_images_clickhouse_tool({"product_images": product_images})
