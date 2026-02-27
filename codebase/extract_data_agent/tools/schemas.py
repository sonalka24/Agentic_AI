from pydantic import BaseModel, Field


class DownloaderFromDatalakeInput(BaseModel):
    bucket: str = Field(description="MinIO bucket name to read from.")
    prefix: str = Field(description="Object prefix inside the bucket.")
    local_dir: str = Field(description="Local directory where files should be downloaded.")


class ReadExcelRowsTextInput(BaseModel):
    excel_files: list[str] = Field(description="Downloaded Excel file paths to parse.")


class ExtractExcelImagesBySectionInput(BaseModel):
    excel_files: list[str] = Field(description="Excel file paths containing embedded images.")
    target_sections: list[str] = Field(description="Ordered section names used to map images.")
    output_dir: str | None = Field(default=None, description="Optional output directory for extracted images.")


class TranslateChineseTextLlmInput(BaseModel):
    excel_rows_text: dict = Field(description="Parsed spreadsheet text grouped by file and sheet.")


class ClickhouseTableCountInput(BaseModel):
    table: str = Field(default="product_facts", description="Target ClickHouse table name in the configured database.")


class ArrangeExtractExcelImagesLlmInput(BaseModel):
    excel_files: list[str] = Field(description="Excel file paths containing embedded images.")
    target_section_schema: dict = Field(description="Section/subsection schema used for image assignment.")
    products_dict: dict | None = Field(default=None, description="Product facts keyed by product id.")
    output_dir: str | None = Field(default=None, description="Optional output directory for arranged images.")


class FillSchemaValuesLlmInput(BaseModel):
    excel_rows_text: dict = Field(description="Parsed spreadsheet text grouped by file and sheet.")
    target_sections: list[str] = Field(description="Target section names to extract.")
    target_section_schema: dict = Field(description="Target section/subsection/key schema.")


class IngestProductFactsClickhouseInput(BaseModel):
    product_table_rows: list[dict] = Field(description="Normalized fact rows ready for ClickHouse ingestion.")


class IngestProductImagesClickhouseInput(BaseModel):
    product_images: list[dict] = Field(description="Prepared image rows ready for ClickHouse ingestion.")
