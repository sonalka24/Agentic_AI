import json
from pathlib import Path

import pandas as pd

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from config import Config


def load_prompts(prompts_path=None):
    """Load prompt templates from JSON and merge with defaults."""
    path = prompts_path or Path(__file__).with_name("prompts.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        raise ValueError('Could not load prompts.json')

    

class Toolset:
    """Collection of data + LLM tools used by the LangGraph agent."""

    def __init__(self, config, minio_client):
        self.config = config
        self.client = minio_client
        self._llm = None
        self.prompts = load_prompts()

    def _get_llm(self):
        """Create/reuse a ChatOpenAI client; return None when unavailable."""
        if self._llm is not None:
            return self._llm
        if not self.config.openai_api_key or ChatOpenAI is None:
            return None
        try:
            self._llm = ChatOpenAI(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                temperature=0,
            )
            return self._llm
        except Exception:
            return None

    @staticmethod
    def _normalize_cell_value(value):
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat") and not isinstance(value, str):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value).strip()

    @staticmethod
    def _safe_json_load(text):
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

    @staticmethod
    def _rows_to_text(rows, max_rows=300):
        lines = []
        for i, row in enumerate(rows[:max_rows]):
            lines.append(f"{i + 1}. " + " | ".join([str(x) for x in row]))
        return "\n".join(lines)

    def _call_openai_json(self, prompt):
        """Execute an LLM call expecting a JSON object and parse the result."""
        llm = self._get_llm()
        if llm is None:
            return None
        try:
            response = llm.bind(response_format={"type": "json_object"}).invoke(
                [
                    ("system", self.prompts["system_json_only"]),
                    ("user", prompt),
                ]
            )
            content = response.content if hasattr(response, "content") else ""
            if isinstance(content, list):
                content = "".join(
                    [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content]
                )
            return self._safe_json_load(str(content))
        except Exception:
            return None

    @staticmethod
    def _heuristic_rows_to_sections(rows):
        section = "General"
        out = {}
        for row in rows:
            if not row:
                continue
            key = str(row[0]).strip()
            value = row[1] if len(row) > 1 else None
            if value is None or str(value).strip() == "":
                section = key
                out.setdefault(section, [])
                continue
            out.setdefault(section, []).append(key)
        for sec in out:
            out[sec] = sorted(set(out[sec]))
        return out

    @staticmethod
    def _heuristic_fill_schema(rows, schema):
        section = "General"
        values = {s: {k: None for k in keys} for s, keys in schema.items()}
        for row in rows:
            if not row:
                continue
            key = str(row[0]).strip()
            value = row[1] if len(row) > 1 else None
            if value is None or str(value).strip() == "":
                section = key
                continue
            if section in values and key in values[section] and values[section][key] is None:
                values[section][key] = value
        return values

    @staticmethod
    def _extract_product_id(rows, default_id):
        for row in rows:
            if len(row) > 1 and str(row[0]).startswith("Art No."):
                return str(row[1])
        return default_id

    def downloader_from_datalake_tool(self, tool_input):
        """Download objects from MinIO path into local directory.

        Input: {bucket, prefix, local_dir}
        Return: {downloaded_files, message}
        """
        bucket_name = tool_input["bucket"]
        raw_prefix = tool_input.get("prefix", "")
        local_dir = Path(tool_input.get("local_dir", self.config.local_dir))
        normalized_prefix = raw_prefix.lstrip("/")
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []
        for obj in self.client.list_objects(bucket_name, prefix=normalized_prefix, recursive=True):
            object_name = obj.object_name
            if not object_name:
                continue
            relative_path = object_name
            if normalized_prefix and object_name.startswith(normalized_prefix):
                relative_path = object_name[len(normalized_prefix) :].lstrip("/")
            destination = local_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.client.fget_object(bucket_name, object_name, str(destination))
            downloaded_files.append(str(destination))

        return {
            "downloaded_files": downloaded_files,
            "message": f"Downloaded {len(downloaded_files)} file(s).",
        }

    def find_excel_files_tool(self, tool_input):
        """Discover Excel files recursively under a local directory.

        Input: {local_dir}
        Return: {excel_files, message}
        """
        local_dir = Path(tool_input.get("local_dir", self.config.local_dir))
        patterns = ("*.xlsx", "*.xls", "*.xlsm")
        excel_files = []
        for pattern in patterns:
            for file_path in local_dir.rglob(pattern):
                excel_files.append(str(file_path))
        excel_files = sorted(set(excel_files))
        return {"excel_files": excel_files, "message": f"Found {len(excel_files)} Excel file(s)."}

    def read_excel_files_single_thread_tool(self, tool_input):
        """Read workbook metadata one file at a time (single-thread only).

        Input: {excel_files}
        Return: {excel_summaries, message}
        """
        excel_files = tool_input.get("excel_files", [])
        summaries = []
        for file_path in excel_files:
            try:
                workbook = pd.ExcelFile(file_path)
                rows = 0
                for sheet in workbook.sheet_names:
                    rows += len(pd.read_excel(file_path, sheet_name=sheet))
                summaries.append({"file": file_path, "sheets": workbook.sheet_names, "rows": rows, "error": None})
            except Exception as exc:
                summaries.append({"file": file_path, "sheets": [], "rows": 0, "error": str(exc)})
        return {
            "excel_summaries": summaries,
            "message": f"Read {len(summaries)} Excel file(s) in single-thread mode.",
        }

    def read_excel_full_text_dict_tool(self, tool_input):
        """Extract full row-level sheet text into nested dict format.

        Input: {excel_files}
        Return: {excel_full_text_dict, message}
        """
        excel_files = tool_input.get("excel_files", [])
        excel_full_text_dict = {}
        for file_path in excel_files:
            file_name = Path(file_path).name
            excel_full_text_dict[file_name] = {"file_path": file_path, "sheets": {}, "error": None}
            try:
                workbook = pd.ExcelFile(file_path)
                for sheet_name in workbook.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                    sheet_rows = []
                    for row_values in df.itertuples(index=False, name=None):
                        normalized = []
                        for v in row_values:
                            n = self._normalize_cell_value(v)
                            if n is not None:
                                normalized.append(n)
                        if normalized:
                            sheet_rows.append(normalized)
                    excel_full_text_dict[file_name]["sheets"][sheet_name] = sheet_rows
            except Exception as exc:
                excel_full_text_dict[file_name]["error"] = str(exc)
        return {
            "excel_full_text_dict": excel_full_text_dict,
            "message": f"Loaded full text dict for {len(excel_full_text_dict)} file(s).",
        }

    def identify_keys_in_sections_llm_tool(self, tool_input):
        """LLM Tool #1: infer section->keys map for each sheet.

        Input: {excel_full_text_dict}
        Return: {identified_section_keys, message}
        Fallback: heuristic extraction when LLM is unavailable/unparseable.
        """
        excel_full_text_dict = tool_input.get("excel_full_text_dict", {})
        identified = {}
        for file_name, payload in excel_full_text_dict.items():
            identified[file_name] = {}
            for sheet_name, rows in payload.get("sheets", {}).items():
                prompt = self.prompts["identify_keys_in_sections"] + "\n\nRows:\n" + self._rows_to_text(rows)
                parsed = self._call_openai_json(prompt)
                if parsed and isinstance(parsed, dict) and isinstance(parsed.get("sections"), dict):
                    sections = {}
                    for sec, keys in parsed.get("sections", {}).items():
                        if isinstance(keys, list):
                            sections[str(sec)] = sorted(set([str(k) for k in keys if str(k).strip()]))
                    identified[file_name][sheet_name] = sections
                else:
                    identified[file_name][sheet_name] = self._heuristic_rows_to_sections(rows)
        return {"identified_section_keys": identified, "message": "Identified keys by section for all sheets."}

    @staticmethod
    def build_common_schema_tool(tool_input):
        """Tool #2: build a common nested schema of frequent section keys.

        Input: {identified_section_keys}
        Return: {common_schema, message}
        """
        identified = tool_input.get("identified_section_keys", {})
        section_key_counts = {}
        sheet_count = 0
        for _, sheets in identified.items():
            for _, sections in sheets.items():
                sheet_count += 1
                for sec, keys in sections.items():
                    section_key_counts.setdefault(sec, {})
                    for key in keys:
                        section_key_counts[sec][key] = section_key_counts[sec].get(key, 0) + 1

        min_hits = max(1, sheet_count // 2)
        common_schema = {}
        for sec, key_counts in section_key_counts.items():
            common_keys = [k for k, c in key_counts.items() if c >= min_hits]
            if common_keys:
                common_schema[sec] = sorted(common_keys)

        return {
            "common_schema": common_schema,
            "message": f"Built common schema with {len(common_schema)} section(s).",
        }

    def fill_schema_values_llm_tool(self, tool_input):
        """LLM Tool #3: fill schema key-values per product/sheet.

        Input: {excel_full_text_dict, common_schema}
        Return: {products_dict, product_table_rows, message}
        Fallback: heuristic fill when LLM is unavailable/unparseable.
        """
        excel_full_text_dict = tool_input.get("excel_full_text_dict", {})
        common_schema = tool_input.get("common_schema", {})
        products = {}
        table_rows = []

        for file_name, payload in excel_full_text_dict.items():
            default_id = Path(file_name).stem.split()[0]
            for sheet_name, rows in payload.get("sheets", {}).items():
                product_id = self._extract_product_id(rows, default_id)
                prompt = (
                    self.prompts["fill_schema_values"]
                    + "\n\nSchema:\n"
                    + json.dumps(common_schema, ensure_ascii=True)
                    + "\n\nRows:\n"
                    + self._rows_to_text(rows)
                )
                parsed = self._call_openai_json(prompt)
                if parsed and isinstance(parsed, dict) and isinstance(parsed.get("sections"), dict):
                    sections_filled = parsed.get("sections", {})
                    pid = str(parsed.get("product_id", product_id))
                else:
                    sections_filled = self._heuristic_fill_schema(rows, common_schema)
                    pid = product_id

                products[pid] = {
                    "product_id": pid,
                    "file": file_name,
                    "sheet": sheet_name,
                    "sections": sections_filled,
                }
                for sec, kv in sections_filled.items():
                    if not isinstance(kv, dict):
                        continue
                    for key, value in kv.items():
                        table_rows.append(
                            {
                                "product_id": pid,
                                "file": file_name,
                                "sheet": sheet_name,
                                "section": sec,
                                "key": key,
                                "value": value,
                            }
                        )

        return {
            "products_dict": products,
            "product_table_rows": table_rows,
            "message": f"Filled schema values for {len(products)} product(s).",
        }

    def registry(self):
        """Return tool registry consumed by the agent graph."""
        return {
            "downloader_from_datalake": self.downloader_from_datalake_tool,
            "find_excel_files": self.find_excel_files_tool,
            "read_excel_files_single_thread": self.read_excel_files_single_thread_tool,
            "read_excel_full_text_dict": self.read_excel_full_text_dict_tool,
            "identify_keys_in_sections_llm": self.identify_keys_in_sections_llm_tool,
            "build_common_schema": self.build_common_schema_tool,
            "fill_schema_values_llm": self.fill_schema_values_llm_tool,
        }
