#!/usr/bin/env python3
"""Build PDF collages from images stored in ClickHouse `product_images`."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib import parse, request

from PIL import Image, ImageDraw, ImageFont


def _load_font(size, bold=False):
    """Load a readable TrueType font with fallback.

    @param size Font size in points.
    @return PIL font object.
    """
    # Prefer bundled/common fonts; fallback keeps script functional.
    bold_fonts = ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf")
    regular_fonts = ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf")
    for name in (bold_fonts if bold else regular_fonts):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _clickhouse_http_query(host, port, db, user, password, query):
    """Execute ClickHouse HTTP query and return response text.

    @param host ClickHouse host.
    @param port ClickHouse HTTP port.
    @param db Database name.
    @param user Username.
    @param password Password.
    @param query SQL query.
    @return str Response payload text.
    """
    params = parse.urlencode({"database": db, "query": query})
    url = f"http://{host}:{port}/?{params}"
    req = request.Request(url=url, data=b"", method="POST")
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("utf-8")
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Content-Type", "text/plain; charset=utf-8")
    with request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _latest_run_id(host, port, db, user, password, table):
    """Fetch latest run_id from a ClickHouse table.

    @param host ClickHouse host.
    @param port ClickHouse HTTP port.
    @param db Database name.
    @param user Username.
    @param password Password.
    @param table Table name.
    @return str Latest run_id or empty string.
    """
    sql = f"SELECT max(run_id) AS run_id FROM {db}.{table} FORMAT JSON"
    payload = _clickhouse_http_query(host, port, db, user, password, sql)
    parsed = json.loads(payload)
    rows = parsed.get("data", [])
    if not rows:
        return ""
    return str(rows[0].get("run_id", "") or "")


def _load_rows(host, port, db, user, password, table, run_id):
    """Load image rows from ClickHouse.

    @param host ClickHouse host.
    @param port ClickHouse HTTP port.
    @param db Database name.
    @param user Username.
    @param password Password.
    @param table Table name.
    @param run_id Optional run_id filter.
    @return list[dict] Image rows.
    """
    where = f"WHERE run_id = '{run_id}'" if run_id else ""
    sql = (
        f"SELECT run_id, product_id, section, subsection, image_id, position, image_blob "
        f"FROM {db}.{table} {where} "
        "ORDER BY product_id, section, subsection, position, image_id "
        "FORMAT JSON"
    )
    payload = _clickhouse_http_query(host, port, db, user, password, sql)
    parsed = json.loads(payload)
    return list(parsed.get("data", []))


def _decode_image(blob_text):
    """Decode base64 image blob into PIL image.

    @param blob_text Base64-encoded image bytes.
    @return PIL.Image.Image or None.
    """
    try:
        raw = base64.b64decode(str(blob_text), validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _draw_label(draw, x, y, text, font):
    """Draw a line of label text.

    @param draw ImageDraw context.
    @param x Left coordinate.
    @param y Top coordinate.
    @param text Label text.
    @param font PIL font object.
    @return int Height consumed.
    """
    draw.text((x, y), text, fill=(20, 20, 20), font=font)
    bbox = draw.textbbox((x, y), text, font=font)
    return max(24, bbox[3] - bbox[1] + 10)


def _tile_size(page_w, margin, gap, columns):
    """Compute tile width/height.

    @param page_w Page width in px.
    @param margin Horizontal page margin in px.
    @param gap Gap between tiles in px.
    @param columns Number of columns.
    @return tuple[int, int] Tile width, tile height.
    """
    usable = page_w - (2 * margin) - ((columns - 1) * gap)
    tw = max(160, usable // columns)
    th = int(tw * 0.75)
    return tw, th


def _build_pdf(rows, out_path):
    """Build collage PDF grouped by product/section/subsection.

    @param rows ClickHouse rows containing image blobs.
    @param out_path Output PDF path.
    @return int Number of pages written.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        pid = str(row.get("product_id", "") or "")
        sec = str(row.get("section", "Unassigned") or "Unassigned")
        sub = str(row.get("subsection", "General") or "General")
        grouped[pid][sec][sub].append(row)

    heading_font = _load_font(58, bold=True)
    pair_font = _load_font(44, bold=True)
    caption_font = _load_font(30)
    pages = []

    page_w, page_h = 2480, 3508  # A4 @300dpi
    margin = 100
    gap = 48
    columns = 3
    tile_w, tile_h = _tile_size(page_w, margin, gap, columns)
    pair_spacing = 84

    for product_id in sorted(grouped.keys()):
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = margin
        y += _draw_label(draw, margin, y, f"Product: {product_id}", heading_font)
        y += 18

        for section in sorted(grouped[product_id].keys()):
            for subsection in sorted(grouped[product_id][section].keys()):
                pair_title = f"Section: {section}  |  Subsection: {subsection}"
                if y > page_h - margin - 320:
                    pages.append(page)
                    page = Image.new("RGB", (page_w, page_h), "white")
                    draw = ImageDraw.Draw(page)
                    y = margin
                    y += _draw_label(draw, margin, y, f"Product: {product_id} (cont.)", heading_font)
                    y += 18
                y += _draw_label(draw, margin + 20, y, pair_title, pair_font)
                y += 20

                x = margin + 20
                row_h = 0
                for item in grouped[product_id][section][subsection]:
                    img = _decode_image(item.get("image_blob", ""))
                    if img is None:
                        continue

                    thumb = img.copy()
                    thumb.thumbnail((tile_w, tile_h))
                    if x + tile_w > page_w - margin:
                        x = margin + 20
                        y += row_h + gap
                        row_h = 0
                    if y + tile_h + 80 > page_h - margin:
                        pages.append(page)
                        page = Image.new("RGB", (page_w, page_h), "white")
                        draw = ImageDraw.Draw(page)
                        y = margin
                        y += _draw_label(draw, margin, y, f"Product: {product_id} (cont.)", heading_font)
                        y += _draw_label(draw, margin + 20, y, pair_title, pair_font)
                        y += 20
                        x = margin + 20
                        row_h = 0

                    frame_h = tile_h + 70
                    draw.rectangle(
                        [x - 3, y - 3, x + tile_w + 3, y + frame_h + 3],
                        outline=(210, 210, 210),
                        width=2,
                    )
                    ix = x + (tile_w - thumb.width) // 2
                    iy = y + (tile_h - thumb.height) // 2
                    page.paste(thumb, (ix, iy))
                    caption = f"{item.get('image_id', '')}  pos={item.get('position', '')}"
                    draw.text((x + 4, y + tile_h + 16), caption[:40], fill=(55, 55, 55), font=caption_font)

                    x += tile_w + gap
                    row_h = max(row_h, frame_h)

                y += row_h + pair_spacing

        pages.append(page)

    if not pages:
        blank = Image.new("RGB", (page_w, page_h), "white")
        d = ImageDraw.Draw(blank)
        d.text((margin, margin), "No image rows found in ClickHouse.", fill=(20, 20, 20), font=heading_font)
        pages = [blank]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_path, "PDF", save_all=True, append_images=pages[1:])
    return len(pages)


def main():
    """CLI entrypoint.

    @return None.
    """
    parser = argparse.ArgumentParser(description="Create PDF collage from ClickHouse product_images table.")
    parser.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "clickhouse"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "8123")))
    parser.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin"))
    parser.add_argument("--db", default=os.getenv("CLICKHOUSE_DB", "plm"))
    parser.add_argument("--table", default="product_images")
    parser.add_argument("--run-id", default="", help="Optional run_id filter; default uses latest run_id.")
    parser.add_argument("--output", default="/app/output/image_collage.pdf")
    args = parser.parse_args()

    run_id = args.run_id.strip()
    if not run_id:
        run_id = _latest_run_id(args.host, args.port, args.db, args.user, args.password, args.table)
    rows = _load_rows(args.host, args.port, args.db, args.user, args.password, args.table, run_id)
    pages = _build_pdf(rows, Path(args.output))
    print(
        f"Wrote collage PDF: {args.output} | pages={pages} | rows={len(rows)} | "
        f"table={args.db}.{args.table} | run_id={run_id or 'all'}"
    )


if __name__ == "__main__":
    main()
