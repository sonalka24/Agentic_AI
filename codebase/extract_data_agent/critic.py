import json

from config import config
from .planner import safe_json_load, summarize_result_for_llm, summarize_steps_for_llm


def llm_critic(phase, payload):
    """Review workflow state with the shared LLM and return approve/replan."""
    llm = config.get_llm()
    if llm is None:
        raise RuntimeError("Critic requires an available LLM client.")

    prompt = (
        "You are the critic for a LangGraph workflow. Review the payload and return strict JSON only.\n"
        'Use shape: {"decision":"approve"|"replan","feedback":"..."}.\n'
        "Choose replan only when the next step or latest result is clearly wrong, unsafe, unusable, or blocks downstream execution.\n"
        "It is acceptable for extracted data to contain placeholders such as '?', 'not available', empty-looking values, or partially missing fields.\n"
        "Do not request replan just because some fields are unknown or incomplete if the workflow can still continue to the next tool.\n"
        "Prefer approve when outputs are structurally valid and usable by downstream steps, even if some values are missing.\n\n"
        f"Phase: {phase}\n"
        "Payload:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
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
        raise RuntimeError("Critic returned invalid JSON.")

    decision = str(parsed.get("decision", "")).strip().lower()
    feedback = str(parsed.get("feedback", "")).strip()
    if decision not in {"approve", "replan"}:
        raise RuntimeError("Critic returned an invalid decision.")
    return decision, feedback or "Critic reviewed the workflow state."


def build_critic_node(logger, tool_meta_by_name):
    """Create an LLM-only critic node."""

    def critic_node(state):
        try:
            updates = dict(state)
            phase = str(state.get("critic_phase", "")).strip().lower()

            if phase == "pre":
                step = state.get("next_step")
                if not isinstance(step, dict):
                    updates.update({"workflow_status": "planner", "critic_phase": None})
                    return updates
                tool_name = str(step.get("tool", "")).strip()
                tool_meta = tool_meta_by_name.get(tool_name, {"input_keys": []})
                decision, feedback = llm_critic(
                    "pre",
                    {
                        "goal": str(state.get("goal", ""))[:800],
                        "next_step": step,
                        "tool_meta": tool_meta,
                        "executed_steps": summarize_steps_for_llm(state.get("executed_steps", [])),
                        "available_state_keys": sorted(state.keys()),
                        "last_tool_result": summarize_result_for_llm(state.get("last_tool_result", {})),
                    },
                )
                logger.info("Agent decision: critic %s before execution | %s", decision, feedback)
                if decision == "replan":
                    updates.update(
                        {
                            "workflow_status": "replan",
                            "critic_phase": None,
                            "next_step": None,
                            "planned_steps": [],
                            "current_step_index": 0,
                            "needs_replan": True,
                            "plan_feedback": feedback,
                        }
                    )
                else:
                    updates.update(
                        {
                            "workflow_status": "execute",
                            "critic_phase": None,
                            "plan_feedback": feedback,
                        }
                    )
                return updates

            if phase == "post":
                tool_name = str(state.get("last_executed_tool", "")).strip()
                result = dict(state.get("last_tool_result", {}) or {})
                decision, feedback = llm_critic(
                    "post",
                    {
                        "goal": str(state.get("goal", ""))[:800],
                        "tool": tool_name,
                        "result": summarize_result_for_llm(result),
                        "executed_steps": summarize_steps_for_llm(state.get("executed_steps", [])),
                        "current_step_index": state.get("current_step_index", 0),
                        "planned_steps": [
                            {"tool": step.get("tool")} for step in list(state.get("planned_steps", []) or [])[:8] if isinstance(step, dict)
                        ],
                    },
                )
                logger.info("Agent decision: critic %s after execution | %s", decision, feedback)
                if decision == "replan":
                    updates.update(
                        {
                            "workflow_status": "replan",
                            "critic_phase": None,
                            "planned_steps": [],
                            "current_step_index": 0,
                            "needs_replan": True,
                            "plan_feedback": feedback,
                        }
                    )
                else:
                    updates.update(
                        {
                            "workflow_status": "planner",
                            "critic_phase": None,
                            "needs_replan": False,
                            "plan_feedback": feedback,
                        }
                    )
                return updates

            updates.update({"workflow_status": "planner", "critic_phase": None})
            return updates
        except Exception as exc:
            logger.exception("Critic failed: %s", exc)
            updates = dict(state)
            updates.update({"error": f"Critic failed: {exc}", "workflow_status": "error", "critic_phase": None})
            return updates

    return critic_node
