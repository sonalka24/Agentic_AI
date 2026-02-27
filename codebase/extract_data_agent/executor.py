from config import config


def summarize_step_result(result):
    """Build a compact log-safe summary of a tool result."""
    if not isinstance(result, dict):
        return "non-dict result"
    if result.get("error"):
        return f"error={result.get('error')}"
    message = str(result.get("message", "")).strip()
    if message:
        return message
    return f"ok keys={sorted(result.keys())}"


def derive_excel_files(downloaded_files):
    """Derive Excel files from downloaded object paths."""
    excel_files = []
    for file_path in downloaded_files:
        lower = str(file_path).lower()
        if lower.endswith((".xlsx", ".xls", ".xlsm")):
            excel_files.append(file_path)
    return excel_files


def resolve_value(value, context):
    """Resolve a value recursively; `$key` pulls from context."""
    if isinstance(value, str) and value.startswith("$"):
        return context.get(value[1:], None)
    if isinstance(value, list):
        return [resolve_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, context) for key, item in value.items()}
    return value


def resolve_tool_input(step, tool_meta, context):
    """Resolve a step input spec into a concrete tool input payload."""
    input_spec = step.get("input", {}) if isinstance(step, dict) else {}
    if not isinstance(input_spec, dict):
        input_spec = {}

    resolved = {}
    for key, value in input_spec.items():
        resolved_value = resolve_value(value, context)
        if resolved_value is not None:
            resolved[str(key)] = resolved_value

    for required_key in tool_meta.get("input_keys", []):
        if required_key not in resolved and required_key in context:
            resolved[required_key] = context[required_key]

    tool_name = str(step.get("tool", "")) if isinstance(step, dict) else ""
    if tool_name == "downloader_from_datalake":
        resolved["bucket"] = config.minio_bucket
        resolved["prefix"] = config.minio_prefix
        resolved["local_dir"] = config.local_dir
    elif tool_name == "extract_excel_images_by_section":
        output_dir = str(resolved.get("output_dir", "") or "").strip()
        if not output_dir or output_dir.startswith("/"):
            resolved["output_dir"] = "output/excel_images"
    elif tool_name == "arrange_extract_excel_images_llm":
        output_dir = str(resolved.get("output_dir", "") or "").strip()
        if not output_dir or output_dir.startswith("/"):
            resolved["output_dir"] = "output/excel_images_llm"
    return resolved


def build_executor_node(
    logger,
    tools,
    tool_meta_by_name,
    default_goal,
    target_sections,
    target_section_schema,
):
    """Create the single-step tool executor node."""

    def tool_executor_node(state):
        try:
            step = state.get("next_step")
            if not isinstance(step, dict):
                updates = dict(state)
                updates["workflow_status"] = "complete"
                return updates

            tool_name = str(step.get("tool", "")).strip()
            if not tool_name:
                updates = dict(state)
                updates["workflow_status"] = "complete"
                return updates

            tool_obj = tools.get(tool_name)
            tool_meta = tool_meta_by_name.get(tool_name, {"input_keys": []})
            if tool_obj is None:
                logger.info("Agent decision: skip unknown tool '%s'", tool_name)
                updates = dict(state)
                updates.update(
                    {
                        "current_step_index": int(state.get("current_step_index", 0) or 0) + 1,
                        "workflow_status": "planner",
                        "critic_phase": None,
                        "next_step": None,
                    }
                )
                return updates

            context = {
                "goal": state.get("goal", default_goal),
                "target_sections": state.get("target_sections", target_sections),
                "target_section_schema": state.get("target_section_schema", target_section_schema),
                "count_table": str(state.get("count_table", "product_facts")),
                "excel_files": list(state.get("excel_files", []) or []),
                "excel_rows_text": dict(state.get("excel_rows_text", {}) or {}),
                "products_dict": dict(state.get("products_dict", {}) or {}),
                "product_table_rows": list(state.get("product_table_rows", []) or []),
                "product_images": list(state.get("product_images", []) or []),
            }

            previous_result = state.get("last_tool_result", {"message": "initial state"})
            current_step_index = int(state.get("current_step_index", 0) or 0)
            planned_steps = state.get("planned_steps", [])

            logger.info(
                "Agent decision: execute step %d/%d tool=%s | Previous result: %s",
                current_step_index + 1,
                len(planned_steps),
                tool_name,
                summarize_step_result(previous_result),
            )

            tool_input = resolve_tool_input(step, tool_meta, context)
            result = tool_obj.invoke(tool_input) if hasattr(tool_obj, "invoke") else tool_obj(tool_input)
            if not isinstance(result, dict):
                result = {"message": f"Tool '{tool_name}' returned non-dict output."}
            if result.get("error"):
                raise RuntimeError(str(result["error"]))

            if "downloaded_files" in result and "excel_files" not in result:
                result = dict(result)
                result["excel_files"] = derive_excel_files(result.get("downloaded_files", []))

            executed_steps = list(state.get("executed_steps", []) or [])
            executed_steps.append({"tool": tool_name, "input_keys": sorted(tool_input.keys())})

            step_messages = list(state.get("step_messages", []) or [])
            message = str(result.get("message", "")).strip()
            if message:
                step_messages.append(message)

            updates = dict(state)
            updates.update(
                {
                    "current_step_index": current_step_index + 1,
                    "workflow_status": "critic",
                    "critic_phase": "post",
                    "last_tool_result": result,
                    "last_executed_tool": tool_name,
                    "executed_steps": executed_steps,
                    "step_messages": step_messages,
                    "next_step": None,
                    "message": " ".join(step_messages).strip(),
                    "openai_configured": bool(config.openai_api_key),
                }
            )
            if "excel_files" in result:
                updates["excel_files"] = result["excel_files"]
            if "excel_rows_text" in result:
                updates["excel_rows_text"] = result["excel_rows_text"]
                updates["full_text_translated"] = result["excel_rows_text"]
            if "product_images" in result:
                updates["product_images"] = result["product_images"]
            if "products_dict" in result:
                updates["products_dict"] = result["products_dict"]
            if "product_table_rows" in result:
                updates["product_table_rows"] = result["product_table_rows"]
            if tool_name == "ingest_product_images_clickhouse":
                updates["ingest_images_result"] = result
            if tool_name == "ingest_product_facts_clickhouse":
                updates["ingest_result"] = result
            return updates
        except Exception as exc:
            logger.exception("Tool execution failed: %s", exc)
            updates = dict(state)
            updates.update(
                {
                    "error": f"Tool execution failed: {exc}",
                    "workflow_status": "error",
                    "critic_phase": None,
                }
            )
            return updates

    return tool_executor_node
