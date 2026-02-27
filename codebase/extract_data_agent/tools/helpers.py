import base64
import json
import re
from urllib import parse, request

import pandas as pd

from config import config


class SharedHelpersMixin:
    def _get_llm(self):
        """Return the shared ChatOpenAI client from configuration."""
        return config.get_llm()

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
    def _rows_to_text(rows):
        return "\n".join([f"{i + 1}. " + " | ".join([str(x) for x in row]) for i, row in enumerate(rows)])

    @staticmethod
    def _text_to_rows(sheet_text):
        rows = []
        for line in str(sheet_text).splitlines():
            raw = line.strip()
            if not raw:
                continue
            if ". " in raw:
                prefix, rest = raw.split(". ", 1)
                if prefix.isdigit():
                    raw = rest
            cells = [c.strip() for c in raw.split("|")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        return rows

    @staticmethod
    def _contains_chinese(text):
        return bool(re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", str(text)))

    @staticmethod
    def _extract_chinese_tokens(text):
        tokens = re.findall(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+", str(text))
        unique_tokens = []
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        return unique_tokens

    @staticmethod
    def _replace_tokens(text, token_map):
        out = str(text)
        for token in sorted(token_map.keys(), key=len, reverse=True):
            replacement = str(token_map.get(token, ""))
            if replacement:
                out = out.replace(token, replacement)
        return out

    def _clickhouse_http_query(self, query, body=None):
        host = config.clickhouse_host
        port = config.clickhouse_port
        db = config.clickhouse_db
        params = parse.urlencode({"database": db, "query": query})
        url = f"http://{host}:{port}/?{params}"
        req = request.Request(url=url, data=body.encode("utf-8") if isinstance(body, str) else body, method="POST")
        user = str(config.clickhouse_user or "")
        password = str(config.clickhouse_password or "")
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("utf-8")
        req.add_header("Authorization", f"Basic {token}")
        req.add_header("Content-Type", "text/plain; charset=utf-8")
        try:
            with request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            raise RuntimeError(f"ClickHouse query failed: {exc}") from exc

    def _call_openai_json(self, prompt, raise_on_error=False):
        llm = self._get_llm()
        if llm is None:
            if raise_on_error:
                raise RuntimeError("OpenAI client unavailable. Check OPENAI_API_KEY/model configuration.")
            return None
        try:
            response = llm.bind(response_format={"type": "json_object"}).invoke([("system", self.prompts["system_json_only"]), ("user", prompt)])
            content = response.content if hasattr(response, "content") else ""
            if isinstance(content, list):
                content = "".join([str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content])
            return self._safe_json_load(str(content))
        except Exception as exc:
            if raise_on_error:
                raise RuntimeError(f"OpenAI JSON call failed: {exc}") from exc
            return None

    def _translate_chinese_to_english(self, text):
        source = str(text)
        if not source.strip() or not self._contains_chinese(source):
            return source
        tokens = self._extract_chinese_tokens(source)
        if not tokens:
            return source
        prompt = self.prompts["translate_chinese_to_english"] + "\n\nTranslate only these Chinese tokens and return a mapping.\nTokens:\n" + json.dumps(tokens, ensure_ascii=False) + '\n\nReturn STRICT JSON with shape: {"translations":{"<chinese_token>":"<english_translation>"}}'
        parsed = self._call_openai_json(prompt, raise_on_error=True)
        translations = parsed.get("translations") if isinstance(parsed, dict) else None
        if not isinstance(translations, dict):
            retry_prompt = prompt + '\n\nYour previous output was invalid. Return ONLY: {"translations":{"...":"..."}}'
            parsed = self._call_openai_json(retry_prompt, raise_on_error=True)
            translations = parsed.get("translations") if isinstance(parsed, dict) else None
        if not isinstance(translations, dict):
            raise RuntimeError("LLM translation failed: invalid JSON translations map.")
        token_map = {}
        for token in tokens:
            value = translations.get(token)
            if isinstance(value, str) and value.strip():
                token_map[token] = value.strip()
        if not token_map:
            raise RuntimeError("LLM translation failed: no token translations returned.")
        return self._replace_tokens(source, token_map)

    def _translate_sections_to_english(self, sections):
        translated_sections = {}
        for section_name, subsection_map in sections.items():
            out_section = {}
            if not isinstance(subsection_map, dict):
                translated_sections[section_name] = out_section
                continue
            for subsection_name, kv in subsection_map.items():
                out_kv = {}
                if not isinstance(kv, dict):
                    out_section[subsection_name] = out_kv
                    continue
                for key, value in kv.items():
                    key_text = str(key)
                    value_text = str(value) if value is not None else ""
                    key_en = self._translate_chinese_to_english(key_text) if self._contains_chinese(key_text) else key_text
                    value_en = self._translate_chinese_to_english(value_text) if self._contains_chinese(value_text) else value_text
                    out_kv[key_en] = value_en
                out_section[subsection_name] = out_kv
            translated_sections[section_name] = out_section
        return translated_sections

    @staticmethod
    def _normalize_sections_from_llm(parsed, target_section_schema):
        if not isinstance(parsed, dict):
            return None, None
        candidate = parsed
        if not isinstance(candidate.get("sections"), dict):
            for wrapper_key in ("data", "result", "output", "payload"):
                wrapped = candidate.get(wrapper_key)
                if isinstance(wrapped, dict) and isinstance(wrapped.get("sections"), dict):
                    candidate = wrapped
                    break
        raw_sections = candidate.get("sections")
        if not isinstance(raw_sections, dict):
            return None, None
        sections_filled = {}
        for section, subsection_schema in target_section_schema.items():
            raw_section = raw_sections.get(section)
            if raw_section is None:
                for sec_name, sec_payload in raw_sections.items():
                    if str(sec_name).strip().lower() == str(section).strip().lower():
                        raw_section = sec_payload
                        break
            if not isinstance(raw_section, dict):
                raw_section = {}
            has_nested = any(isinstance(v, dict) for v in raw_section.values())
            section_out = {}
            subsection_names = list(subsection_schema.keys())
            first_sub = subsection_names[0] if subsection_names else "General"
            if not has_nested:
                section_out[first_sub] = raw_section
                for sub in subsection_names:
                    section_out.setdefault(sub, {})
                sections_filled[section] = section_out
                continue
            for subsection in subsection_names:
                raw_kv = raw_section.get(subsection)
                if raw_kv is None:
                    for sub_name, sub_payload in raw_section.items():
                        if str(sub_name).strip().lower() == str(subsection).strip().lower():
                            raw_kv = sub_payload
                            break
                section_out[subsection] = raw_kv if isinstance(raw_kv, dict) else {}
            sections_filled[section] = section_out
        product_id = candidate.get("product_id", parsed.get("product_id", ""))
        return sections_filled, str(product_id) if product_id is not None else ""

    @staticmethod
    def _extract_product_id(rows, default_id):
        for row in rows:
            if len(row) > 1 and str(row[0]).startswith("Art No."):
                return str(row[1])
        return default_id

    @staticmethod
    def _norm_label(text):
        return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())

    @staticmethod
    def _safe_path_name(text):
        cleaned = re.sub(r"[^\w.\-]+", "_", str(text).strip())
        return cleaned.strip("_") or "unknown"

    @staticmethod
    def _anchor_to_row_col(anchor):
        marker = getattr(anchor, "_from", None)
        if marker is not None:
            return int(marker.row) + 1, int(marker.col) + 1
        if isinstance(anchor, str) and anchor:
            m = re.match(r"^([A-Za-z]+)(\d+)$", anchor)
            if m:
                col_letters, row_str = m.groups()
                col = 0
                for ch in col_letters.upper():
                    col = col * 26 + (ord(ch) - ord("A") + 1)
                return int(row_str), int(col)
        return None, None

    def _row_section_ranges(self, ws, target_sections):
        target_lookup = {}
        for section in target_sections:
            norm = self._norm_label(section)
            if norm:
                target_lookup[norm] = str(section)
        starts = []
        max_row = int(ws.max_row or 0)
        for row_idx, row_values in enumerate(ws.iter_rows(values_only=True), start=1):
            row_tokens = []
            for value in row_values:
                normalized = self._normalize_cell_value(value)
                if normalized:
                    row_tokens.append(normalized)
            if not row_tokens:
                continue
            matched = None
            for token in row_tokens:
                token_norm = self._norm_label(token)
                if not token_norm:
                    continue
                if token_norm in target_lookup:
                    matched = target_lookup[token_norm]
                    break
                for section_norm, section_name in target_lookup.items():
                    if len(section_norm) >= 4 and (section_norm in token_norm or token_norm in section_norm):
                        matched = section_name
                        break
                if matched:
                    break
            if matched and (not starts or starts[-1]["section"] != matched):
                starts.append({"row": row_idx, "section": matched})
        if not starts:
            return []
        ranges = []
        for i, start in enumerate(starts):
            start_row = start["row"]
            end_row = starts[i + 1]["row"] - 1 if i + 1 < len(starts) else max_row
            ranges.append({"section": start["section"], "start_row": start_row, "end_row": end_row})
        return ranges

    @staticmethod
    def _resolve_section_for_row(row_num, section_ranges):
        if row_num is None:
            return "Unassigned"
        for item in section_ranges:
            if item["start_row"] <= row_num <= item["end_row"]:
                return item["section"]
        if section_ranges and row_num < section_ranges[0]["start_row"]:
            return section_ranges[0]["section"]
        if section_ranges:
            return section_ranges[-1]["section"]
        return "Unassigned"

    def _assign_images_with_llm_and_fallback(self, images, section_rows, subsection_rows, target_section_schema):
        section_names = list(target_section_schema.keys())
        valid_pairs = set()
        for section, subsection_map in target_section_schema.items():
            if isinstance(subsection_map, dict):
                for subsection in subsection_map.keys():
                    valid_pairs.add((str(section), str(subsection)))
        section_rows_sorted = sorted([r for r in section_rows if isinstance(r.get("row"), int)], key=lambda x: int(x["row"]))
        subsection_rows_sorted = sorted([r for r in subsection_rows if isinstance(r.get("row"), int)], key=lambda x: int(x["row"]))

        def _section_from_row(image_row):
            if not section_rows_sorted:
                return "Unassigned"
            section = None
            for rec in section_rows_sorted:
                if int(rec["row"]) <= int(image_row):
                    section = str(rec.get("section", "Unassigned"))
                else:
                    break
            if section is not None:
                return section
            return str(section_rows_sorted[0].get("section", "Unassigned"))

        def _subsection_from_row(section, image_row):
            in_section = [r for r in subsection_rows_sorted if str(r.get("section", "")) == str(section)]
            if not in_section:
                subsection_map = target_section_schema.get(section, {})
                return str(next(iter(subsection_map.keys()), "General")) if isinstance(subsection_map, dict) else "General"
            subsection = None
            for rec in in_section:
                if int(rec["row"]) <= int(image_row):
                    subsection = str(rec.get("subsection", "General"))
                else:
                    break
            if subsection is not None:
                return subsection
            return str(in_section[0].get("subsection", "General"))

        def deterministic_assign(image_row):
            if not isinstance(image_row, int) or image_row <= 0:
                return {"section": "Unassigned", "subsection": "General"}
            section = _section_from_row(image_row)
            subsection = _subsection_from_row(section, image_row)
            return {"section": section, "subsection": subsection}

        fallback = {}
        for img in images:
            fallback[str(img.get("image_id"))] = deterministic_assign(img.get("row"))

        llm_images = [{"image_id": str(img.get("image_id")), "row": int(img.get("row", 0)) if isinstance(img.get("row"), int) else 0, "col": int(img.get("col", 0)) if isinstance(img.get("col"), int) else 0, "anchor_cell": str(img.get("anchor_cell", ""))} for img in images]
        prompt = (
            "Assign each image to the closest section/subsection using row proximity.\n"
            "Rules:\n"
            "- Prefer closest subsection row when available.\n"
            "- If subsection unclear, pick closest section and a valid subsection under that section.\n"
            "- Use only given section/subsection names.\n"
            "- Keep one assignment per image_id.\n"
            "Return JSON only: {\"assignments\":[{\"image_id\":\"...\",\"section\":\"...\",\"subsection\":\"...\",\"position\":123}]}\n\n"
            + "Sections:\n" + json.dumps(section_names, ensure_ascii=False)
            + "\n\nSection rows:\n" + json.dumps(section_rows, ensure_ascii=False)
            + "\n\nSubsection rows:\n" + json.dumps(subsection_rows, ensure_ascii=False)
            + "\n\nImages:\n" + json.dumps(llm_images, ensure_ascii=False)
        )
        parsed = self._call_openai_json(prompt)
        assignments = parsed.get("assignments") if isinstance(parsed, dict) else None
        if not isinstance(assignments, list):
            assignments = []
        out = {}
        for rec in assignments:
            if not isinstance(rec, dict):
                continue
            image_id = str(rec.get("image_id", "")).strip()
            if not image_id:
                continue
            section = str(rec.get("section", "")).strip()
            subsection = str(rec.get("subsection", "")).strip()
            position = rec.get("position")
            if (section, subsection) not in valid_pairs:
                fb = fallback.get(image_id, {"section": "Unassigned", "subsection": "General"})
                section, subsection = fb["section"], fb["subsection"]
            try:
                position_int = int(position)
            except Exception:
                position_int = 0
            out[image_id] = {"section": section, "subsection": subsection, "position": position_int}
        for img in images:
            image_id = str(img.get("image_id"))
            if image_id not in out:
                fb = fallback.get(image_id, {"section": "Unassigned", "subsection": "General"})
                row_num = img.get("row")
                out[image_id] = {"section": fb["section"], "subsection": fb["subsection"], "position": int(row_num) if isinstance(row_num, int) else 0}
        return out
