import json

from langgraph.graph import END, StateGraph

from config import config
from .sql import clickhouse_http_query, safe_json_load, sanitize_sql


def build_conversation_graph(runtime_config):
    """Build a question-to-SQL conversation graph."""
    llm = runtime_config.get_llm()

    def planner_node(state):
        return {"question": str(state.get("question", "")).strip()}

    def generate_sql_node(state):
        question = state.get("question", "")
        if not question:
            return {"error": "Question is empty."}
        if llm is None:
            return {"error": "OPENAI_API_KEY/model is required for conversation SQL generation."}

        prompt = (
            "You write ClickHouse SQL for a single table.\n"
            f"Table: {runtime_config.clickhouse_db}.product_facts\n"
            "Columns: run_id, product_id, file, sheet, section, subsection, fact_key, fact_value, ingested_at\n"
            "Rules:\n"
            "- Return JSON only: {\"sql\":\"...\"}\n"
            "- SQL must be SELECT only\n"
            "- Must query only the table above\n"
            "- Use case-insensitive matching for text filters using lowerUTF8(...)\n"
            "- Prefer broad matching first (section/subsection/fact_key/fact_value) before strict equality\n"
            "- If question asks about a concept, search with OR across fact_key and fact_value\n"
            "- If question mentions product id/art no, filter by product_id and also by fact_value when key resembles Art No\n"
            "- Never require exact key text unless explicitly requested by user\n"
            "- Include LIMIT 50 or less\n\n"
            "Useful key synonyms:\n"
            "- color => color, colour\n"
            "- art number => art no, article no, product id\n"
            "- package/packaging => packaging, inner packaging, outer packaging, pallet\n"
            "- size => size, packed size\n"
            "- material => material, type of product\n\n"
            "Query strategy:\n"
            "1) Build WHERE using broad LIKE match on lowerUTF8(fact_key) and lowerUTF8(fact_value).\n"
            "2) If a product id token exists, include it in WHERE.\n"
            "3) Select useful columns: product_id, section, subsection, fact_key, fact_value, file, sheet.\n"
            "4) Add ORDER BY product_id, section, subsection, fact_key.\n\n"
            f"Question: {question}"
        )
        response = llm.bind(response_format={"type": "json_object"}).invoke(
            [("system", "Return strict JSON only."), ("user", prompt)]
        )
        content = response.content if hasattr(response, "content") else ""
        if isinstance(content, list):
            content = "".join([str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content])
        parsed = safe_json_load(str(content))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sql"), str):
            return {"error": "Failed to generate SQL from question."}
        try:
            sql = sanitize_sql(runtime_config.clickhouse_db, parsed["sql"])
        except Exception as exc:
            return {"error": str(exc)}
        return {"sql": sql}

    def execute_sql_node(state):
        if state.get("error"):
            return {}
        sql = state.get("sql", "")
        if not sql:
            return {"error": "Generated SQL is empty."}
        try:
            raw = clickhouse_http_query(runtime_config, sql)
            parsed = json.loads(raw)
            rows = parsed.get("data", []) if isinstance(parsed, dict) else []
            return {"rows": rows}
        except Exception as exc:
            return {"error": f"ClickHouse query failed: {exc}"}

    def answer_node(state):
        if state.get("error"):
            return {"answer": f"Error: {state['error']}"}
        rows = state.get("rows", [])
        question = state.get("question", "")
        if llm is None:
            return {"answer": f"Rows returned: {len(rows)}", "rows": rows}
        prompt = (
            "Answer user question using SQL rows. Keep answer concise and factual.\n"
            "If rows exist, do NOT say data not found. Summarize what was found.\n"
            "If rows are empty, say data not found.\n"
            "When possible, mention product_id and section/subsection.\n\n"
            f"Question: {question}\n\nRows:\n{json.dumps(rows, ensure_ascii=False)}"
        )
        response = llm.invoke([("system", "You are a data assistant."), ("user", prompt)])
        content = response.content if hasattr(response, "content") else ""
        if isinstance(content, list):
            content = "".join([str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content])
        return {"answer": str(content).strip(), "rows": rows}

    graph_builder = StateGraph(dict)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("generate_sql", generate_sql_node)
    graph_builder.add_node("execute_sql", execute_sql_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.set_entry_point("planner")
    graph_builder.add_edge("planner", "generate_sql")
    graph_builder.add_edge("generate_sql", "execute_sql")
    graph_builder.add_edge("execute_sql", "answer")
    graph_builder.add_edge("answer", END)
    return graph_builder.compile()


def main():
    """Run the interactive conversation CLI."""
    app = build_conversation_graph(config)
    print(
        f"Connected target: {config.clickhouse_db}.product_facts @ "
        f"{config.clickhouse_host}:{config.clickhouse_port}"
    )
    print("Type a question, or 'exit'.")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = app.invoke({"question": question})
        print(f"A> {result.get('answer', 'No answer.')}" )
