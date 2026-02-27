import json
from datetime import datetime, timezone

from config import config


class IngestToolsMixin:
    def ingest_product_images_clickhouse_tool(self, tool_input):
        product_images = tool_input.get("product_images", [])
        db = config.clickhouse_db
        table = "product_images"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._clickhouse_http_query(f"CREATE DATABASE IF NOT EXISTS {db}")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {db}.{table} (
            run_id String,
            product_id String,
            section String,
            subsection String,
            image_id String,
            position UInt32,
            image_blob String,
            ingested_at DateTime
        ) ENGINE = MergeTree
        ORDER BY (product_id, section, subsection, position, image_id)
        """
        self._clickhouse_http_query(create_sql)
        if not product_images:
            return {"inserted_rows": 0, "table": f"{db}.{table}", "run_id": run_id, "message": "No image rows to ingest. Table ensured."}
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for row in product_images:
            lines.append(json.dumps({"run_id": run_id, "product_id": str(row.get("product_id", "")), "section": str(row.get("section", "Unassigned")), "subsection": str(row.get("subsection", "General")), "image_id": str(row.get("image_id", "")), "position": int(row.get("position", 0)), "image_blob": str(row.get("image_blob", "")), "ingested_at": now_str}, ensure_ascii=False))
        self._clickhouse_http_query(f"INSERT INTO {db}.{table} FORMAT JSONEachRow", body="\n".join(lines))
        return {"inserted_rows": len(lines), "table": f"{db}.{table}", "run_id": run_id, "message": f"Ingested {len(lines)} image row(s) into {db}.{table}."}

    def ingest_product_facts_clickhouse_tool(self, tool_input):
        product_table_rows = tool_input.get("product_table_rows", [])
        db = config.clickhouse_db
        table = "product_facts"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._clickhouse_http_query(f"CREATE DATABASE IF NOT EXISTS {db}")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {db}.{table} (
            run_id String,
            product_id String,
            file String,
            sheet String,
            section String,
            subsection String,
            fact_key String,
            fact_value String,
            ingested_at DateTime
        ) ENGINE = MergeTree
        ORDER BY (product_id, section, subsection, fact_key, sheet, file)
        """
        self._clickhouse_http_query(create_sql)
        if not product_table_rows:
            return {"inserted_rows": 0, "table": f"{db}.{table}", "run_id": run_id, "message": "No rows to ingest. Table ensured."}
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for row in product_table_rows:
            lines.append(json.dumps({"run_id": run_id, "product_id": str(row.get("product_id", "")), "file": str(row.get("file", "")), "sheet": str(row.get("sheet", "")), "section": str(row.get("section", "")), "subsection": str(row.get("subsection", "General")), "fact_key": str(row.get("key", "")), "fact_value": str(row.get("value", "")), "ingested_at": now_str}, ensure_ascii=False))
        self._clickhouse_http_query(f"INSERT INTO {db}.{table} FORMAT JSONEachRow", body="\n".join(lines))
        return {"inserted_rows": len(lines), "table": f"{db}.{table}", "run_id": run_id, "message": f"Ingested {len(lines)} row(s) into {db}.{table}."}

    def clickhouse_table_count_tool(self, tool_input):
        table = str(tool_input.get("table", "product_facts"))
        db = config.clickhouse_db
        raw = self._clickhouse_http_query(f"SELECT count() AS c FROM {db}.{table} FORMAT JSON")
        try:
            parsed = json.loads(raw)
            rows = parsed.get("data", [])
            count = int(rows[0].get("c", 0)) if rows else 0
        except Exception as exc:
            raise RuntimeError(f"Failed to parse ClickHouse count response: {exc}") from exc
        return {"table": f"{db}.{table}", "row_count": count, "message": f"Table {db}.{table} has {count} row(s)."}
