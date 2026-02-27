import base64
import json
from urllib import parse, request


def clickhouse_http_query(config, query):
    """Execute a ClickHouse query over HTTP."""
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


def safe_json_load(text):
    """Parse JSON with simple brace-based recovery fallback."""
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


def sanitize_sql(db, sql):
    """Validate and normalize SQL to a safe read-only ClickHouse query."""
    candidate = str(sql).strip().rstrip(";")
    if not candidate.lower().startswith("select"):
        raise RuntimeError("Only SELECT queries are allowed.")
    if any(token in candidate.lower() for token in ["insert ", "update ", "delete ", "drop ", "alter "]):
        raise RuntimeError("Write/DDL queries are not allowed.")
    if f"{db}.product_facts" not in candidate:
        raise RuntimeError(f"Query must target {db}.product_facts.")
    if " limit " not in candidate.lower():
        candidate = f"{candidate} LIMIT 50"
    if not candidate.lower().endswith("format json"):
        candidate = f"{candidate} FORMAT JSON"
    return candidate
