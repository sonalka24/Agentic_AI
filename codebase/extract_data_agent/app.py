import json
import os
from pathlib import Path

from langgraph.graph import END, StateGraph

from config import config
from .critic import build_critic_node
from .executor import build_executor_node
from .planner import build_planner_node, build_tool_catalog
from .telemetry import logger
from .tools import Toolset

JSON_DIR = Path(__file__).resolve().parent / "json"
DEFAULT_GOAL = (
    "Download product spreadsheets, extract text and images, map to schema sections, "
    "and ingest facts/images into ClickHouse."
)


def _load_target_section_schema():
    """Load target section schema from local JSON file."""
    path = JSON_DIR / "schema.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load tool schema from {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"Invalid schema in {path}: expected non-empty object.")
    return data


def build_agent(tools):
    """Construct the extraction workflow graph."""
    target_section_schema = _load_target_section_schema()
    target_sections = list(target_section_schema.keys())
    tool_catalog = build_tool_catalog(tools)
    tool_meta_by_name = {item["name"]: item for item in tool_catalog}

    planner_node = build_planner_node(
        logger=logger,
        tool_catalog=tool_catalog,
        default_goal=DEFAULT_GOAL,
        target_sections=target_sections,
        target_section_schema=target_section_schema,
    )
    critic_node = build_critic_node(logger=logger, tool_meta_by_name=tool_meta_by_name)
    executor_node = build_executor_node(
        logger=logger,
        tools=tools,
        tool_meta_by_name=tool_meta_by_name,
        default_goal=DEFAULT_GOAL,
        target_sections=target_sections,
        target_section_schema=target_section_schema,
    )

    def route_after_planner(state):
        if state.get("workflow_status") == "critic":
            return "critic"
        return END

    def route_after_critic(state):
        status = state.get("workflow_status")
        if status == "execute":
            return "tool_executor"
        if status in {"planner", "replan"}:
            return "planner"
        return END

    graph_builder = StateGraph(dict)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("tool_executor", executor_node)
    graph_builder.set_entry_point("planner")
    graph_builder.add_conditional_edges("planner", route_after_planner)
    graph_builder.add_conditional_edges("critic", route_after_critic)
    graph_builder.add_edge("tool_executor", "critic")
    return graph_builder.compile()


def _resolve_goal():
    """Resolve goal from env or default."""
    env_goal = str(os.getenv("AGENT_GOAL", "")).strip()
    return env_goal or DEFAULT_GOAL


def main():
    """Run end-to-end extraction and ingestion workflow."""
    tools = Toolset.from_config().registry()
    agent = build_agent(tools=tools)
    result = agent.invoke({"goal": _resolve_goal()})

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
