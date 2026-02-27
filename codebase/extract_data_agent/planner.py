import json

from config import config


def safe_json_load(text):
    """Parse JSON with a simple object-recovery fallback."""
    try:
        return json.loads(text)
    except Exception:
        start = str(text).find("{")
        end = str(text).rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(str(text)[start : end + 1])
            except Exception:
                return None
        return None


def extract_input_keys(tool_obj):
    """Extract input field names from a LangChain tool schema when present."""
    schema = getattr(tool_obj, "args_schema", None)
    if schema is None:
        return []
    if hasattr(schema, "model_fields"):
        return list(schema.model_fields.keys())
    if hasattr(schema, "__fields__"):
        return list(schema.__fields__.keys())
    return []


def extract_tool_description(tool_obj):
    """Extract a planner-facing description from a LangChain tool."""
    description = str(getattr(tool_obj, "description", "")).strip()
    return description or "No description."


def build_tool_catalog(tools):
    """Build structured metadata for tools visible to planner and critic."""
    catalog = []
    for name, tool_obj in tools.items():
        catalog.append(
            {
                "name": str(getattr(tool_obj, "name", name)),
                "description": extract_tool_description(tool_obj),
                "input_keys": extract_input_keys(tool_obj),
            }
        )
    catalog.sort(key=lambda item: item["name"])
    return catalog


def normalize_steps(raw_steps, tool_catalog):
    """Normalize LLM plan into executable steps."""
    if not isinstance(raw_steps, list):
        return []
    available = {item["name"] for item in tool_catalog}
    normalized = []
    for item in raw_steps:
        if isinstance(item, str):
            tool_name = item.strip()
            input_spec = {}
        elif isinstance(item, dict):
            tool_name = str(item.get("tool", item.get("name", ""))).strip()
            input_spec = item.get("input", {})
        else:
            continue
        if not tool_name or tool_name not in available:
            continue
        if not isinstance(input_spec, dict):
            input_spec = {}
        normalized.append({"tool": tool_name, "input": input_spec})
        if len(normalized) >= 20:
            break
    return normalized


def summarize_result_for_llm(result):
    """Reduce large tool outputs to compact planner-safe summaries."""
    if not isinstance(result, dict):
        return {"type": type(result).__name__}

    summary = {}
    if "message" in result:
        summary["message"] = str(result.get("message", ""))[:400]
    if "error" in result:
        summary["error"] = str(result.get("error", ""))[:400]
    if "downloaded_files" in result:
        files = list(result.get("downloaded_files", []) or [])
        summary["downloaded_files_count"] = len(files)
        summary["downloaded_files_sample"] = files[:3]
    if "excel_files" in result:
        files = list(result.get("excel_files", []) or [])
        summary["excel_files_count"] = len(files)
        summary["excel_files_sample"] = files[:3]
    if "excel_rows_text" in result:
        excel_rows_text = dict(result.get("excel_rows_text", {}) or {})
        summary["excel_rows_text_files"] = len(excel_rows_text)
        sample = {}
        for file_name, payload in list(excel_rows_text.items())[:2]:
            sheets = dict(payload.get("sheets", {}) or {}) if isinstance(payload, dict) else {}
            sample[file_name] = {"sheet_count": len(sheets), "sheet_names": list(sheets.keys())[:3]}
        summary["excel_rows_text_sample"] = sample
    if "products_dict" in result:
        products = dict(result.get("products_dict", {}) or {})
        summary["products_count"] = len(products)
        summary["product_ids_sample"] = list(products.keys())[:5]
    if "product_table_rows" in result:
        rows = list(result.get("product_table_rows", []) or [])
        summary["product_table_rows_count"] = len(rows)
        summary["product_table_rows_sample"] = rows[:2]
    if "product_images" in result:
        images = list(result.get("product_images", []) or [])
        summary["product_images_count"] = len(images)
        summary["product_images_sample"] = [
            {
                "product_id": item.get("product_id"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "image_id": item.get("image_id"),
            }
            for item in images[:2]
            if isinstance(item, dict)
        ]
    if "inserted_rows" in result:
        summary["inserted_rows"] = result.get("inserted_rows")
    if "table" in result:
        summary["table"] = result.get("table")
    if "row_count" in result:
        summary["row_count"] = result.get("row_count")
    return summary


def summarize_steps_for_llm(steps):
    """Reduce executed or planned steps to compact summaries."""
    out = []
    for step in list(steps or [])[-8:]:
        if isinstance(step, dict):
            out.append(
                {
                    "tool": step.get("tool"),
                    "input_keys": list(step.get("input_keys", []))[:8] if isinstance(step.get("input_keys"), list) else [],
                }
            )
    return out


def plan_with_llm(goal, tool_catalog, context_keys, critic_feedback="", executed_steps=None, last_tool_result=None):
    """Use the shared LLM to generate an execution plan."""
    llm = config.get_llm()
    if llm is None:
        raise RuntimeError("Planner requires an available LLM client.")

    feedback_text = str(critic_feedback or "none").strip()
    prompt = (
        "Create an execution plan for the goal using available tools.\n"
        "Return strict JSON only with shape: "
        '{"steps":[{"tool":"tool_name","input":{"arg":"$context_key_or_literal"}}],"reason":"..."}.\n'
        "Rules:\n"
        "- Use only listed tool names.\n"
        "- steps must be ordered exactly as you want to execute.\n"
        "- input values can reference context using $key notation.\n"
        "- Keep plan minimal and goal-directed.\n"
        "- If critic feedback exists, repair the plan rather than repeating the same mistake.\n\n"
        f"Goal:\n{goal}\n\n"
        "Critic feedback:\n"
        f"{feedback_text[:1200]}\n\n"
        "Executed steps so far:\n"
        f"{json.dumps(summarize_steps_for_llm(executed_steps), ensure_ascii=False)}\n\n"
        "Last tool result summary:\n"
        f"{json.dumps(summarize_result_for_llm(last_tool_result), ensure_ascii=False)}\n\n"
        "Context keys currently available:\n"
        f"{json.dumps(context_keys, ensure_ascii=False)}\n\n"
        "Tool catalog:\n"
        f"{json.dumps(tool_catalog, ensure_ascii=False)}"
    )
    response = llm.bind(response_format={"type": "json_object"}).invoke(
        [("system", "Return strict JSON only."), ("user", prompt)]
    )
    content = response.content if hasattr(response, "content") else ""
    if isinstance(content, list):
        content = "".join(
            [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content]
        )
    parsed = safe_json_load(str(content))
    if not isinstance(parsed, dict):
        raise RuntimeError("Planner returned invalid JSON.")
    raw_steps = parsed.get("steps", parsed.get("plan", []))
    steps = normalize_steps(raw_steps, tool_catalog)
    if not steps:
        raise RuntimeError("Planner returned empty or invalid steps.")
    reason = str(parsed.get("reason", "LLM produced dynamic execution plan.")).strip()
    return steps, "llm", reason or "LLM produced dynamic execution plan."


def build_planner_node(logger, tool_catalog, default_goal, target_sections, target_section_schema):
    """Create the planner node used by the graph."""

    def planner_node(state):
        try:
            goal = str(state.get("goal", default_goal)).strip() or default_goal
            planned_steps = state.get("planned_steps", [])
            current_step_index = int(state.get("current_step_index", 0) or 0)
            should_replan = bool(state.get("workflow_status") == "replan" or state.get("needs_replan"))

            updates = dict(state)
            updates["goal"] = goal
            updates.setdefault("target_sections", target_sections)
            updates.setdefault("target_section_schema", target_section_schema)

            if should_replan:
                planned_steps = []
                current_step_index = 0
                updates["current_step_index"] = 0
                updates["planned_steps"] = []

            if not isinstance(planned_steps, list) or not planned_steps:
                context_keys = sorted(
                    {
                        "target_sections",
                        "target_section_schema",
                        "excel_files",
                        "excel_rows_text",
                        "products_dict",
                        "product_table_rows",
                        "product_images",
                        "count_table",
                    }
                )
                planned_steps, plan_source, plan_reason = plan_with_llm(
                    goal=goal,
                    tool_catalog=tool_catalog,
                    context_keys=context_keys,
                    critic_feedback=str(state.get("plan_feedback", "")),
                    executed_steps=list(state.get("executed_steps", []) or []),
                    last_tool_result=dict(state.get("last_tool_result", {}) or {}),
                )
                updates.update(
                    {
                        "planned_steps": planned_steps,
                        "plan_source": plan_source,
                        "plan_reason": plan_reason,
                        "plan_feedback": "",
                        "needs_replan": False,
                    }
                )
                logger.info(
                    "Agent decision: planned %d step(s) via %s | goal=%s",
                    len(planned_steps),
                    plan_source,
                    goal,
                )

            next_step = None
            workflow_status = "complete"
            critic_phase = None
            if current_step_index < len(planned_steps):
                next_step = planned_steps[current_step_index]
                workflow_status = "critic"
                critic_phase = "pre"
                logger.info(
                    "Agent decision: planner selected step %d/%d tool=%s",
                    current_step_index + 1,
                    len(planned_steps),
                    str(next_step.get("tool", "")),
                )
            else:
                logger.info(
                    "Agent decision: workflow complete | executed %d step(s)",
                    len(state.get("executed_steps", [])),
                )

            updates.update(
                {
                    "current_step_index": current_step_index,
                    "next_step": next_step,
                    "workflow_status": workflow_status,
                    "critic_phase": critic_phase,
                }
            )
            return updates
        except Exception as exc:
            logger.exception("Planner failed: %s", exc)
            updates = dict(state)
            updates.update({"error": f"Planner failed: {exc}", "workflow_status": "error", "critic_phase": None})
            return updates

    return planner_node
