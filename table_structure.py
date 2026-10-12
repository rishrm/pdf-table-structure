#!/usr/bin/env python3
"""Structure-only PDF table baseline, with optional bounded Ollama proposals."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import pdfplumber


def cluster_boundaries(values, tolerance=1.0):
    """Cluster by total range, so long chains cannot collapse a narrow cell."""
    groups = []
    for value in sorted(set(float(v) for v in values)):
        if groups and value - groups[-1][0] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group), 3) for group in groups]


def check_grid(table):
    """Check logical occupancy; validity is not a claim of detection accuracy."""
    occupied = set()
    warnings = []
    for cell in table["cells"]:
        for row in range(cell["row"], cell["row"] + cell["row_span"]):
            for column in range(cell["column"], cell["column"] + cell["column_span"]):
                if (row, column) in occupied:
                    warnings.append("overlapping_grid_cells")
                occupied.add((row, column))
    total = table["row_count"] * table["column_count"]
    if len(occupied) < total:
        warnings.append("uncovered_grid_slots")
    return sorted(set(warnings))


def structure_from_cells(raw_cells, method, tolerance=1.0):
    boxes = sorted(set(tuple(float(v) for v in box) for box in raw_cells),
                   key=lambda b: (b[1], b[0], b[3], b[2]))
    if not boxes or any(not all(math.isfinite(v) for v in b) or
                        b[0] >= b[2] or b[1] >= b[3] for b in boxes):
        raise ValueError("Invalid cell geometry")
    xs = cluster_boundaries([b[i] for b in boxes for i in (0, 2)], tolerance)
    ys = cluster_boundaries([b[i] for b in boxes for i in (1, 3)], tolerance)
    cells = []
    for index, box in enumerate(boxes):
        left, right = [min(range(len(xs)), key=lambda i: abs(xs[i] - box[j]))
                       for j in (0, 2)]
        top, bottom = [min(range(len(ys)), key=lambda i: abs(ys[i] - box[j]))
                       for j in (1, 3)]
        if left >= right or top >= bottom:
            raise ValueError("Cell too small for boundary tolerance")
        cells.append({"id": f"c{index:04d}", "row": top, "column": left,
                      "row_span": bottom - top, "column_span": right - left,
                      "bbox": [xs[left], ys[top], xs[right], ys[bottom]]})
    table = {"bbox": [xs[0], ys[0], xs[-1], ys[-1]],
             "method": method, "row_count": len(ys) - 1,
             "column_count": len(xs) - 1, "x_boundaries": xs,
             "y_boundaries": ys, "cells": cells}
    table["warnings"] = check_grid(table)
    if method == "text":
        table["warnings"].append("experimental_borderless_detection")
    return table


def word_boxes_only(page):
    # pdfplumber uses text locally for word segmentation. Only positions survive.
    return [[round(word[k], 3) for k in ("x0", "top", "x1", "bottom")]
            for word in page.extract_words()]


def boxes_in_table(boxes, bbox):
    return [box for box in boxes if
            bbox[0] <= (box[0] + box[2]) / 2 <= bbox[2] and
            bbox[1] <= (box[1] + box[3]) / 2 <= bbox[3]]


def collapse_text_gaps(table, boxes):
    """Drop whitespace-only interstitial bands in text-strategy tables.

    Deliberately restricted to tables without vertical spans. This does not
    solve wrapped-text vs. logical-row ambiguity, which remains a warning.
    """
    if any(c["row_span"] != 1 for c in table["cells"]):
        return
    ys = table["y_boundaries"]
    active = [r for r in range(table["row_count"]) if any(
        ys[r] <= (b[1] + b[3]) / 2 < ys[r + 1] for b in boxes)]
    if not active or len(active) == table["row_count"]:
        return
    boundaries = [ys[active[0]]]
    boundaries += [(ys[a + 1] + ys[b]) / 2 for a, b in zip(active, active[1:])]
    boundaries += [ys[active[-1] + 1]]
    boundaries = [round(v, 3) for v in boundaries]
    row_map = {old: new for new, old in enumerate(active)}
    cells = []
    for original in table["cells"]:
        if original["row"] not in row_map:
            continue
        cell = copy.deepcopy(original)
        cell["row"] = row_map[cell["row"]]
        cell["bbox"][1] = boundaries[cell["row"]]
        cell["bbox"][3] = boundaries[cell["row"] + 1]
        cells.append(cell)
    for index, cell in enumerate(cells):
        cell["id"] = f"c{index:04d}"
    table.update(row_count=len(active), y_boundaries=boundaries, cells=cells)
    table["bbox"][1], table["bbox"][3] = boundaries[0], boundaries[-1]
    table["warnings"] = check_grid(table) + [
        "experimental_borderless_detection", "whitespace_row_bands_collapsed",
        "wrapped_text_may_be_split_into_rows"]


def space_text_boundaries(table, boxes, bounds):
    """Place inferred dividers within observed whitespace, rather than on text."""
    xs = table["x_boundaries"]
    columns = [[] for _ in range(table["column_count"])]
    for box in boxes:
        center = (box[0] + box[2]) / 2
        for index in range(len(columns)):
            if xs[index] <= center <= xs[index + 1]:
                columns[index].append(box)
                break
    updated = list(xs)
    for index in range(1, len(xs) - 1):
        if columns[index - 1] and columns[index]:
            left = max(b[2] for b in columns[index - 1])
            right = min(b[0] for b in columns[index])
            if left < right:
                updated[index] = round((left + right) / 2, 3)
    updated[0] = max(bounds[0], updated[0] - 4)
    updated[-1] = min(bounds[2], updated[-1] + 4)
    ys = list(table["y_boundaries"])
    ys[0] = max(bounds[1], ys[0] - 4)
    ys[-1] = min(bounds[3], ys[-1] + 4)
    table["x_boundaries"], table["y_boundaries"] = updated, ys
    table["bbox"] = [updated[0], ys[0], updated[-1], ys[-1]]
    for cell in table["cells"]:
        r, c = cell["row"], cell["column"]
        cell["bbox"] = [updated[c], ys[r], updated[c + cell["column_span"]],
                        ys[r + cell["row_span"]]]
    table["warnings"].append("boundaries_inferred_from_word_gaps")


def apply_merges(table, groups):
    """Apply already-validated merges to a separate review candidate."""
    result = copy.deepcopy(table)
    by_id = {c["id"]: c for c in table["cells"]}
    consumed = {cid for group in groups for cid in group}
    result["cells"] = [c for c in result["cells"] if c["id"] not in consumed]
    for index, group in enumerate(groups):
        cells = [by_id[cid] for cid in group]
        row, column = min(c["row"] for c in cells), min(c["column"] for c in cells)
        bottom = max(c["row"] + c["row_span"] for c in cells)
        right = max(c["column"] + c["column_span"] for c in cells)
        result["cells"].append({"id": f"m{index:04d}", "row": row,
            "column": column, "row_span": bottom - row,
            "column_span": right - column,
            "bbox": [table["x_boundaries"][column], table["y_boundaries"][row],
                     table["x_boundaries"][right], table["y_boundaries"][bottom]],
            "source_cell_ids": group})
    result["cells"].sort(key=lambda c: (c["row"], c["column"]))
    result["warnings"] = sorted(set(result["warnings"] + check_grid(result) +
                                     ["model_proposal_requires_visual_review"]))
    return result


def detect_page(page, strategy="auto", bbox=None, tolerance=2.0, scan_dpi=144):
    region = page.crop(tuple(bbox)) if bbox else page
    warnings = []
    # Conservative auto mode avoids applying text detection to a whole page
    # that already has ruled tables. Explicit --strategy text is available.
    method = "lines" if strategy == "auto" else strategy
    settings = {"vertical_strategy": method, "horizontal_strategy": method,
                "snap_tolerance": tolerance, "join_tolerance": tolerance,
                "intersection_tolerance": tolerance}
    raw_tables = [] if strategy == "raster" else region.find_tables(settings)
    raster_attempted = strategy == "raster" or (
        strategy == "auto" and not raw_tables and bool(region.images))
    raster_diagnostics = None
    transform = None
    if raster_attempted:
        from raster_structure import page_grids
        raw_tables, transform, raster_diagnostics = page_grids(
            region, dpi=scan_dpi, tolerance=tolerance)
        method = "raster"
        warnings.extend(raster_diagnostics["warnings"])
        warnings.append("raster_borderless_or_broken_tables_may_be_missed")
    boxes = None
    if strategy == "auto" and not raw_tables and region.chars and not raster_attempted:
        method = "text"
        settings.update(vertical_strategy="text", horizontal_strategy="text")
        raw_tables = region.find_tables(settings)
    tables = []
    for raw_table in raw_tables:
        try:
            raw_cells = raw_table if method == "raster" else raw_table.cells
            table = structure_from_cells(raw_cells, method,
                                         tolerance=min(tolerance, 1.0))
        except ValueError:
            warnings.append("invalid_candidate_geometry_skipped")
            continue
        if method == "text":
            if boxes is None:
                boxes = word_boxes_only(region)
            table_boxes = boxes_in_table(boxes, table["bbox"])
            collapse_text_gaps(table, table_boxes)
            space_text_boundaries(table, table_boxes, region.bbox)
        if table["row_count"] < 2 or table["column_count"] < 2:
            warnings.append("single_row_or_column_candidate_skipped")
            continue
        if method == "raster":
            from raster_structure import map_to_original
            table["warnings"].append("experimental_raster_detection_requires_review")
            map_to_original(table, transform)
        table["id"] = f"p{page.page_number:03d}_t{len(tables) + 1:02d}"
        tables.append(table)
    if strategy == "auto" and tables and method == "lines":
        warnings.append("auto_mode_does_not_search_for_additional_borderless_tables")
    if not region.chars and region.images and not raster_attempted:
        warnings.append("use_auto_or_raster_strategy_for_image_content")
    elif region.images and not raster_attempted:
        warnings.append("raster_tables_may_be_missed")
    if not tables:
        warnings.append("no_table_candidates_detected")
    result = {"page": page.page_number, "width": page.width, "height": page.height,
            "page_bbox": list(page.bbox), "rotation": page.rotation,
            "status": "candidates_detected" if tables else
                ("needs_fallback" if raster_attempted else
                 ("needs_vision" if not region.chars and region.images else "no_candidates")),
            "tables": tables, "warnings": sorted(set(warnings))}
    if raster_diagnostics is not None:
        result["raster_detection"] = raster_diagnostics
    return result


def render_overlay(page, tables, path, dpi=120, candidate=False):
    image = page.to_image(resolution=dpi, antialias=True)

    def outline(item, color, width):
        if "polygon" in item:
            points = item["polygon"]
            for start, end in zip(points, points[1:] + points[:1]):
                image.draw_line((tuple(start), tuple(end)), stroke=color, stroke_width=width)
        else:
            image.draw_rect(tuple(item["bbox"]), fill=None, stroke=color, stroke_width=width)

    for entry in tables:
        table = entry.get("model_candidate", entry) if candidate else entry
        outline(table, "#172554", 3)
        for cell in table["cells"]:
            merged = cell["row_span"] > 1 or cell["column_span"] > 1
            color = "#d97706" if merged else "#0891b2"
            outline(cell, color, 2)
    image.save(str(path), quantize=False)


def parse_pages(value, page_count):
    if not value:
        return list(range(1, page_count + 1))
    result = set()
    try:
        for part in value.split(","):
            if "-" in part:
                left, right = [int(v) for v in part.split("-", 1)]
                if not 1 <= left <= right <= page_count:
                    raise ValueError
                result.update(range(left, right + 1))
            else:
                number = int(part)
                if not 1 <= number <= page_count:
                    raise ValueError
                result.add(number)
    except ValueError:
        raise ValueError(f"Pages must lie between 1 and {page_count}, e.g. 1,3-5") from None
    return sorted(result)


def run(args):
    started = time.perf_counter()
    output_dir = Path(args.output).resolve()
    # Fresh output folders prevent stale overlays from earlier runs looking current.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output folder must be empty; choose a new folder for each run")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.1", "coordinate_system":
              "PDF points; x right, y down; bbox=[x0, top, x1, bottom] in pdfplumber page coordinates",
              "indexing": "page is 1-based; row and column are 0-based",
              "cell_text_in_json": False, "overlays_include_original_page": not args.no_overlays,
              "model_input": "geometry_only", "model_requested": args.ollama_model,
              "strategy": args.strategy, "pages": []}
    with pdfplumber.open(args.pdf) as document:
        selected = parse_pages(args.pages, len(document.pages))
        for number in selected:
            page_started = time.perf_counter()
            page = document.pages[number - 1]
            result = detect_page(page, args.strategy, args.bbox, args.tolerance,
                                 getattr(args, "scan_dpi", 144))
            result["detection_seconds"] = round(time.perf_counter() - page_started, 4)
            if args.ollama_model:
                from ollama_refine import propose_merges
                boxes = word_boxes_only(page)
                for table in result["tables"]:
                    # Model proposals are useful only for inferred borderless cells.
                    # Actual drawn borders are already stronger evidence.
                    if table["method"] in ("lines", "raster"):
                        table["model_review"] = {"status": "skipped",
                            "reason": "detected_borders_used_as_structure", "merges": []}
                        continue
                    if check_grid(table):
                        table["model_review"] = {"status": "skipped",
                            "reason": "incomplete_or_overlapping_grid", "merges": []}
                        continue
                    review = propose_merges(table, boxes_in_table(boxes, table["bbox"]),
                        model=args.ollama_model,
                        url=os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat"),
                        api_key=os.environ.get("OLLAMA_API_KEY"), timeout=args.timeout,
                        json_schema=not args.json_mode)
                    table["model_review"] = review
                    if review["status"] == "proposed":
                        table["model_candidate"] = apply_merges(table, review["merges"])
            if not args.no_overlays:
                overlay_path = output_dir / f"page-{number:03d}-baseline.png"
                try:
                    render_overlay(page, result["tables"], overlay_path, args.dpi)
                    result["baseline_overlay"] = overlay_path.name
                    if any("model_candidate" in t for t in result["tables"]):
                        proposed = output_dir / f"page-{number:03d}-proposed.png"
                        render_overlay(page, result["tables"], proposed, args.dpi, candidate=True)
                        result["proposed_overlay"] = proposed.name
                except Exception as exc:
                    # No exception text: PDF libraries can include document content.
                    result["warnings"].append(f"overlay_render_failed:{type(exc).__name__}")
            report["pages"].append(result)
            page.close()
    report["metrics"] = {"wall_seconds": round(time.perf_counter() - started, 4),
                         "page_count": len(report["pages"]),
                         "table_candidates": sum(len(p["tables"]) for p in report["pages"])}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["metrics"]["process_peak_rss_mb"] = round(
            peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
    except ImportError:
        pass
    path = output_dir / "structure.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Input PDF (never modified)")
    parser.add_argument("--output", default="runs/baseline", help="New, empty output folder")
    parser.add_argument("--strategy", choices=["auto", "lines", "text", "raster"], default="auto")
    parser.add_argument("--pages", help="1-based selection, e.g. 1,3-5")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "TOP", "X1", "BOTTOM"),
                        help="Restrict detection to a known region, in PDF points")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Line snapping tolerance in points")
    parser.add_argument("--no-overlays", action="store_true", help="Export JSON only; avoid rendering cost")
    parser.add_argument("--dpi", type=int, default=120, help="Local overlay rendering resolution")
    parser.add_argument("--scan-dpi", type=int, default=144,
                        help="Pixel detection resolution (72-200); dimensions capped at 2200 pixels")
    parser.add_argument("--ollama-model", help="Opt-in model alias, e.g. llama3.1:latest, oss, gemma3.1")
    parser.add_argument("--timeout", type=float, default=45, help="HTTP timeout seconds")
    parser.add_argument("--json-mode", action="store_true", help="Use format=json instead of JSON Schema")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or not 0 < args.tolerance <= 20:
        parser.error("tolerance must be greater than 0 and no more than 20")
    if not 36 <= args.dpi <= 300:
        parser.error("dpi must be between 36 and 300")
    if not 72 <= args.scan_dpi <= 200:
        parser.error("scan-dpi must be between 72 and 200")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
        parser.error("timeout must be greater than 0 and no more than 120")
    try:
        path, report = run(args)
    except (ValueError, OSError) as exc:
        # CLI argument/path errors are safe; PDF parse errors are handled below.
        print(f"Cannot run: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"PDF processing failed ({type(exc).__name__}). Check the input PDF.", file=sys.stderr)
        return 2
    print(f"Wrote {path}")
    print(json.dumps(report["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
