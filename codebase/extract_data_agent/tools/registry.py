try:
    from langchain_core.tools import StructuredTool
except Exception:
    StructuredTool = None

from .schemas import (
    ArrangeExtractExcelImagesLlmInput,
    ClickhouseTableCountInput,
    DownloaderFromDatalakeInput,
    ExtractExcelImagesBySectionInput,
    FillSchemaValuesLlmInput,
    IngestProductFactsClickhouseInput,
    IngestProductImagesClickhouseInput,
    ReadExcelRowsTextInput,
    TranslateChineseTextLlmInput,
)


class RegistryMixin:
    def registry(self):
        if StructuredTool is None:
            return {
                "downloader_from_datalake": self.downloader_from_datalake,
                "read_excel_rows_text": self.read_excel_rows_text,
                "extract_excel_images_by_section": self.extract_excel_images_by_section,
                "arrange_extract_excel_images_llm": self.arrange_extract_excel_images_llm,
                "translate_chinese_text_llm": self.translate_chinese_text_llm,
                "fill_schema_values_llm": self.fill_schema_values_llm,
                "ingest_product_facts_clickhouse": self.ingest_product_facts_clickhouse,
                "ingest_product_images_clickhouse": self.ingest_product_images_clickhouse,
                "clickhouse_table_count": self.clickhouse_table_count,
            }

        def structured_from_docstring(fn, args_schema=None):
            kwargs = {"description": (fn.__doc__ or "").strip(), "parse_docstring": False, "error_on_invalid_docstring": False}
            if args_schema is not None:
                kwargs["args_schema"] = args_schema
            return StructuredTool.from_function(fn, **kwargs)

        return {
            "downloader_from_datalake": structured_from_docstring(self.downloader_from_datalake, DownloaderFromDatalakeInput),
            "read_excel_rows_text": structured_from_docstring(self.read_excel_rows_text, ReadExcelRowsTextInput),
            "extract_excel_images_by_section": structured_from_docstring(self.extract_excel_images_by_section, ExtractExcelImagesBySectionInput),
            "arrange_extract_excel_images_llm": structured_from_docstring(self.arrange_extract_excel_images_llm, ArrangeExtractExcelImagesLlmInput),
            "translate_chinese_text_llm": structured_from_docstring(self.translate_chinese_text_llm, TranslateChineseTextLlmInput),
            "fill_schema_values_llm": structured_from_docstring(self.fill_schema_values_llm, FillSchemaValuesLlmInput),
            "ingest_product_facts_clickhouse": structured_from_docstring(self.ingest_product_facts_clickhouse, IngestProductFactsClickhouseInput),
            "ingest_product_images_clickhouse": structured_from_docstring(self.ingest_product_images_clickhouse, IngestProductImagesClickhouseInput),
            "clickhouse_table_count": structured_from_docstring(self.clickhouse_table_count, ClickhouseTableCountInput),
        }
