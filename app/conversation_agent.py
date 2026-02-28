import base64
import json
from urllib import parse, request

from langgraph.graph import END, StateGraph

from config import load_config

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


def _get_llm(config):
    """Create a chat model client for SQL/answer generation.

    @param config Runtime configuration.
    @return ChatOpenAI or None when unavailable.
    """
    if not config.openai_api_key or ChatOpenAI is None:
        return None
    return ChatOpenAI(
        api_key=config.openai_api_key,
        model=config.openai_model,
        temperature=0,
    )


def _clickhouse_http_query(config, query):
    """Execute a ClickHouse query over HTTP.

    @param config Runtime configuration with ClickHouse creds.
    @param query SQL text to execute.
    @return str Raw response body.
    """
    params = parse.urlencode({"database": config.clickhouse_db, "query": query})
    url = f"http://{config.clickhouse_host}:{config.clickhouse_port}/?{params}"
    req = request.Request(url=url, data=b"", method="POST")
    token = base64.b64encode(
        f"{config.clickhouse_user}:{config.clickhouse_password}".encode("utf-8")
    ).decode("utf-8")
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Content-Type", "text/plain; charset=utf-8")
    with request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _safe_json_load(text):
    """Parse JSON with simple brace-based recovery fallback.

    @param text Candidate JSON content.
    @return dict|list|None Parsed object or None on failure.
    """
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


def _sanitize_sql(db, sql):
    """Validate and normalize SQL to safe read-only ClickHouse query.

    @param db Allowed database name.
    @param sql Candidate SQL string from LLM.
    @return str Sanitized SQL with `FORMAT JSON`.
    @raises RuntimeError If query violates read-only/table constraints.
    """
    candidate = str(sql).strip().rstrip(";")
    if not candidate.lower().startswith("select"):
        raise RuntimeError("Only SELECT queries are allowed.")
    if any(token in candidate.lower() for token in ["insert ", "update ", "delete ", "drop ", "alter "]):
        raise RuntimeError("Write/DDL queries are not allowed.")
    if f"{db}.product_facts" not in candidate:
        # Force the intended table.
        raise RuntimeError(f"Query must target {db}.product_facts.")
    if " limit " not in candidate.lower():
        candidate = f"{candidate} LIMIT 50"
    if not candidate.lower().endswith("format json"):
        candidate = f"{candidate} FORMAT JSON"
    return candidate


def build_conversation_graph(config):
    """Build a question-to-SQL conversation graph.

    @param config Runtime configuration.
    @return Compiled LangGraph application.
    """
    llm = _get_llm(config)

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
            f"Table: {config.clickhouse_db}.product_facts\n"
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
            content = "".join(
                [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content]
            )
        parsed = _safe_json_load(str(content))
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sql"), str):
            return {"error": "Failed to generate SQL from question."}
        try:
            sql = _sanitize_sql(config.clickhouse_db, parsed["sql"])
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
            raw = _clickhouse_http_query(config, sql)
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
            content = "".join(
                [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content]
            )
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
    """Run the interactive conversation CLI.

    @return None.
    """
    config = load_config()
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
        print(f"A> {result.get('answer', 'No answer.')}")


if __name__ == "__main__":
    main()
   