import logging
import json
from pathlib import Path

from langgraph.graph import END, StateGraph
from rich.logging import RichHandler


def _get_logger():
    """Create or reuse the module logger.

    @return logging.Logger configured with rich handler.
    """
    logger = logging.getLogger("extract_data_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RichHandler(rich_tracebacks=True, markup=True)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _summarize_step_result(result):
    """Build a compact log-safe summary of a tool result.

    @param result Previous step output object.
    @return str Summary text for logging.
    """
    if not isinstance(result, dict):
        return "non-dict result"
    if result.get("error"):
        return f"error={result.get('error')}"
    message = str(result.get("message", "")).strip()
    if message:
        return message
    keys = sorted(result.keys())
    return f"ok keys={keys}"


def _load_target_section_schema():
    """Load target section schema from local JSON file.

    @return dict Non-empty section/subsection schema.
    @raises RuntimeError If schema file is unreadable or invalid.
    """
    path = Path(__file__).with_name("schema.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load tool schema from {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"Invalid schema in {path}: expected non-empty object.")
    return data


def build_agent(config, tools):
    """Construct the extraction and ingestion agent graph.

    @param config Runtime configuration object.
    @param tools Registry of executable tool callables.
    @return Compiled LangGraph application.
    """
    logger = _get_logger()
    target_section_schema = _load_target_section_schema()
    target_sections = list(target_section_schema.keys())

    def planner_node(state):
        plan = {
            "bucket": state.get("bucket", config.minio_bucket),
            "prefix": state.get("prefix", config.minio_prefix),
            "local_dir": state.get("local_dir", config.local_dir),
            "execution_mode": "single_thread",
        }
        logger.info(
            "Agent decision: set execution_mode=%s | Next step: tool_executor | Previous step result: none (planner start)",
            plan["execution_mode"],
        )
        return plan

    def tool_executor_node(state):
        try:
            previous_step = "none"
            previous_result = {"message": "initial state"}

            logger.info(
                "Agent decision: execute downloader_from_datalake | Next step: downloader_from_datalake | Previous step result: %s",
                _summarize_step_result(previous_result),
            )
            download_result = tools["downloader_from_datalake"](
                {"bucket": state["bucket"], "prefix": state["prefix"], "local_dir": state["local_dir"]}
            )
            previous_step, previous_result = "downloader_from_datalake", download_result

            logger.info(
                "Agent decision: derive excel_files from downloader output | Next step: read_excel_rows_text | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            downloaded_files = download_result.get("downloaded_files", [])
            excel_files = []
            for file_path in downloaded_files:
                lower = str(file_path).lower()
                if lower.endswith(".xlsx") or lower.endswith(".xls") or lower.endswith(".xlsm"):
                    excel_files.append(file_path)
            previous_step, previous_result = "derive_excel_files", {
                "excel_files": excel_files,
                "message": f"Derived {len(excel_files)} Excel file(s) from downloader output.",
            }

            logger.info(
                "Agent decision: execute read_excel_rows_text | Next step: read_excel_rows_text | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            full_text_result = tools["read_excel_rows_text"]({"excel_files": excel_files})
            previous_step, previous_result = "read_excel_rows_text", full_text_result
            logger.info(
                "Agent decision: execute translate_chinese_text_llm on sheet text | Next step: translate_chinese_text_llm | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            full_text_translated = tools["translate_chinese_text_llm"](
                {"excel_rows_text": full_text_result.get("excel_rows_text", {})}
            )
            previous_step, previous_result = "translate_chinese_text_llm", full_text_translated
            logger.info(
                "Agent decision: execute fill_schema_values_llm with explicit sections | Next step: fill_schema_values_llm | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            section_template = {
                section: {
                    subsection: {key: "not available" for key in keys}
                    for subsection, keys in subsection_map.items()
                }
                for section, subsection_map in target_section_schema.items()
            }
            fill_result = tools["fill_schema_values_llm"](
                {
                    "excel_rows_text": full_text_translated.get("excel_rows_text", {}),
                    "target_sections": target_sections,
                    "target_section_schema": target_section_schema,
                }
            )
            previous_step, previous_result = "fill_schema_values_llm", fill_result
            logger.info(
                "Agent decision: execute arrange_extract_excel_images_llm | Next step: arrange_extract_excel_images_llm | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            image_result = tools["arrange_extract_excel_images_llm"](
                {
                    "excel_files": excel_files,
                    "target_section_schema": target_section_schema,
                    "products_dict": fill_result.get("products_dict", {}),
                }
            )
            previous_step, previous_result = "arrange_extract_excel_images_llm", image_result
            logger.info(
                "Agent decision: execute ingest_product_images_clickhouse | Next step: ingest_product_images_clickhouse | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            ingest_images_result = tools["ingest_product_images_clickhouse"](
                {"product_images": image_result.get("product_images", [])}
            )
            previous_step, previous_result = "ingest_product_images_clickhouse", ingest_images_result
            logger.info(
                "Agent decision: execute ingest_product_facts_clickhouse | Next step: ingest_product_facts_clickhouse | Previous step (%s) result: %s",
                previous_step,
                _summarize_step_result(previous_result),
            )
            ingest_result = tools["ingest_product_facts_clickhouse"](
                {"product_table_rows": fill_result.get("product_table_rows", [])}
            )
            previous_step, previous_result = "ingest_product_facts_clickhouse", ingest_result

            final_message = (
                f"{download_result.get('message', '')} "
                f"{previous_result.get('message', '')} "
                f"{full_text_result.get('message', '')} "
                f"{image_result.get('message', '')} "
                f"{full_text_translated.get('message', '')} "
                f"{fill_result.get('message', '')}"
                f"{ingest_images_result.get('message', '')} "
                f"{ingest_result.get('message', '')} "
            ).strip()

            return {
                "excel_files": excel_files,
                "excel_rows_text": full_text_result.get("excel_rows_text", {}),
                "product_images": image_result.get("product_images", []),
                "full_text_translated": full_text_translated.get("excel_rows_text", {}),
                "identified_section_keys": {},
                "common_schema": section_template,
                "products_dict": fill_result.get("products_dict", {}),
                "product_table_rows": fill_result.get("product_table_rows", []),
                "ingest_images_result": ingest_images_result,
                "ingest_result": ingest_result,
                "message": final_message,
                "openai_configured": bool(config.openai_api_key),
                "execution_mode": "single_thread",
            }
        except Exception as exc:
            logger.exception("Tool execution failed: %s", exc)
            return {"error": f"Tool execution failed: {exc}"}

    graph_builder = StateGraph(dict)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("tool_executor", tool_executor_node)
    graph_builder.set_entry_point("planner")
    graph_builder.add_edge("planner", "tool_executor")
    graph_builder.add_edge("tool_executor", END)
    return graph_builder.compile()
