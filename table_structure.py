#!/usr/bin/env python3
"""Structure-only PDF table baseline, with optional bounded Ollama proposals."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import statistics
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
    if (len(xs) - 1) * (len(ys) - 1) > 65536:
        raise ValueError("Grid area exceeds the local processing budget")
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
    # Scale the join tolerance to font size: a fixed three-point gap can merge
    # a publisher's entire line into one word when spaces are positioning gaps.
    return [[round(word[k], 3) for k in ("x0", "top", "x1", "bottom")]
            for word in page.extract_words(x_tolerance_ratio=0.15)]


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


def text_evidence(table, boxes, region_bbox):
    """Conservative geometry checks, without inspecting word meanings."""
    words = boxes_in_table(boxes, table["bbox"])
    counts = {cell["id"]: 0 for cell in table["cells"]}
    cuts = 0
    for word in words:
        cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
        cell = next((c for c in table["cells"] if
                     c["bbox"][0] <= cx <= c["bbox"][2] and
                     c["bbox"][1] <= cy <= c["bbox"][3]), None)
        if cell is None:
            cuts += 1
            continue
        counts[cell["id"]] += 1
        b = cell["bbox"]
        if word[0] < b[0] - 0.75 or word[2] > b[2] + 0.75 or \
                word[1] < b[1] - 0.75 or word[3] > b[3] + 0.75:
            cuts += 1
    nonempty = [count for count in counts.values() if count]
    evidence = {"word_boxes": len(words), "word_cut_fraction":
                round(cuts / len(words), 4) if words else 0,
                "median_words_per_nonempty_cell": statistics.median(nonempty) if nonempty else 0}
    column_medians = []
    for column in range(table["column_count"]):
        values = [counts[c["id"]] for c in table["cells"] if
                  c["column"] == column and c["column_span"] == 1 and counts[c["id"]]]
        column_medians.append(statistics.median(values) if values else 0)
    table_width = table["bbox"][2] - table["bbox"][0]
    heavy_columns = [i for i, value in enumerate(column_medians) if value >= 8 and
        table["x_boundaries"][i + 1] - table["x_boundaries"][i] >= 0.3 * table_width]
    evidence["column_word_medians"] = column_medians
    evidence["long_prose_column_beside_short_cells"] = bool(
        table["row_count"] >= 4 and table["column_count"] <= 5 and heavy_columns and
        any(0 < value <= 3 for i, value in enumerate(column_medians) if i not in heavy_columns))
    reasons = []
    if words and evidence["word_cut_fraction"] > 0.18:
        reasons.append("inferred_boundaries_cut_words")
    region_height = region_bbox[3] - region_bbox[1]
    if table["column_count"] <= 4 and table["row_count"] >= 8 and \
            table["bbox"][3] - table["bbox"][1] > 0.25 * region_height and \
            evidence["median_words_per_nonempty_cell"] >= 5:
        reasons.append("prose_like_columns")
    return reasons, evidence


def mixed_image_evidence(table, boxes, images):
    """Do not join a picture and its separate neighboring prose into one grid."""
    b = table["bbox"]
    area = (b[2] - b[0]) * (b[3] - b[1])
    words = boxes_in_table(boxes, b)
    if not words:
        return []
    for image in images:
        bounds = [image.get(key) for key in ("x0", "top", "x1", "bottom")]
        if any(value is None for value in bounds):
            continue
        x0, top, x1, bottom = bounds
        overlap = max(0, min(b[2], x1) - max(b[0], x0)) * \
                  max(0, min(b[3], bottom) - max(b[1], top))
        if not 0.15 < overlap / area < 0.85:
            continue
        outside = sum(not (x0 <= (w[0] + w[2]) / 2 <= x1 and
                           top <= (w[1] + w[3]) / 2 <= bottom) for w in words)
        extends_sideways = b[0] < x0 - 20 or b[2] > x1 + 20
        if extends_sideways and outside / len(words) > 0.3:
            return ["layout_crosses_image_and_neighboring_prose"]
    return []


def contained_grid(inner, outer):
    """A crop rediscovery is redundant only when its actual cell boxes agree."""
    a, b = inner["bbox"], outer["bbox"]
    if a[0] < b[0] - 1 or a[1] < b[1] - 1 or a[2] > b[2] + 1 or a[3] > b[3] + 1:
        return False
    return all(any(all(abs(x - y) <= 1 for x, y in zip(cell["bbox"], other["bbox"]))
                   for other in outer["cells"]) for cell in inner["cells"])


def same_table(first, second):
    """Deduplicate substantially overlapping results, retaining earlier routes."""
    a, b = first["bbox"], second["bbox"]
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
                   max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return (union > 0 and intersection / union >= 0.65) or contained_grid(second, first)


def image_evidence(table, metadata, horizontal_rules, figure_context=None):
    """Reject graphic-dominated or weak tiny layout candidates; no OCR used."""
    from figure_quality import context_evidence, figure_evidence_reasons
    pixel = dict(metadata.get("pixel_evidence", {}))
    if figure_context is not None:
        pixel["diagram_context"] = context_evidence(table["bbox"], figure_context)
    b = table["bbox"]
    width = b[2] - b[0]
    rule_support = any(
        min(b[2], rule[2]) - max(b[0], rule[0]) >= 0.75 * width and
        b[1] - 16 <= (rule[1] + rule[3]) / 2 <= b[3] + 3
        for rule in horizontal_rules)
    evidence = {"long_horizontal_rule_support": rule_support, "pixel": pixel}
    reasons = figure_evidence_reasons(pixel)
    if table.get("evidence", {}).get("long_prose_column_beside_short_cells") and not rule_support:
        reasons.append("unruled_layout_includes_long_prose_column")
    if pixel.get("status") == "ok":
        if pixel.get("large_component_ink_fraction", 0) > 0.55 and \
                pixel.get("character_component_ink_fraction", 0) < 0.35:
            reasons.append("graphic_dominated_ink")
        # A few aligned marks inside a figure are not enough to establish a
        # borderless table. Real small ruled tables retain independent support.
        if table["row_count"] < 4 and not rule_support:
            reasons.append("insufficient_repeated_layout_or_rules")
    return reasons, evidence


def detect_page(page, strategy="auto", bbox=None, tolerance=2.0, scan_dpi=144,
                layout_dpi=200):
    region = page.crop(tuple(bbox)) if bbox else page
    warnings, tables, attempts = [], [], []
    boxes = None
    image_attempted = False
    diagnostics = {}

    def ingest(groups, method, transform=None, backend_details=None):
        nonlocal boxes
        attempt = {"method": method, "raw_candidates": len(groups),
                   "accepted": 0, "rejected": []}
        for raw_index, raw in enumerate(groups):
            try:
                cells = raw.cells if method in ("lines", "text") else raw
                if len(cells) > 4096:
                    raise ValueError("cell_limit")
                candidate = structure_from_cells(cells, method, min(tolerance, 1.0))
            except (ValueError, TypeError):
                attempt["rejected"].append({"reasons": ["invalid_or_oversize_cell_geometry"]})
                continue
            reasons = []
            if candidate["row_count"] < 2 or candidate["column_count"] < 2:
                reasons.append("single_row_or_column_candidate")
            if candidate["row_count"] * candidate["column_count"] > 65536:
                reasons.append("grid_area_limit")
            b, bounds = candidate["bbox"], region.bbox
            if b[0] < bounds[0] - 1 or b[1] < bounds[1] - 1 or \
                    b[2] > bounds[2] + 1 or b[3] > bounds[3] + 1:
                reasons.append("candidate_outside_selected_region")
            if not reasons:
                reasons.extend(check_grid(candidate))
            if method == "native" and not backend_details:
                reasons.append("native_envelope_unverified")
            if method == "text" and not reasons:
                if boxes is None:
                    boxes = word_boxes_only(region)
                table_words = boxes_in_table(boxes, candidate["bbox"])
                collapse_text_gaps(candidate, table_words)
                space_text_boundaries(candidate, table_words, region.bbox)
                reasons.extend(check_grid(candidate))
                if candidate["row_count"] < 2 or candidate["column_count"] < 2:
                    reasons.append("single_row_or_column_after_text_normalization")
            if method in ("text", "img2table", "native") and region.chars and not reasons:
                if boxes is None:
                    boxes = word_boxes_only(region)
                evidence_reasons, evidence = text_evidence(candidate, boxes, region.bbox)
                candidate["evidence"] = evidence
                reasons.extend(evidence_reasons)
                if method == "img2table":
                    reasons.extend(mixed_image_evidence(candidate, boxes, region.images))
            if method in ("img2table", "native", "scan_recovery") and backend_details and not reasons:
                group_metadata = backend_details.get("returned_group_metadata", [])
                metadata = group_metadata[raw_index] if raw_index < len(group_metadata) else {}
                if method == "native" and not metadata.get("complete_envelope"):
                    reasons.append("native_envelope_unverified")
                if metadata.get("layout_repair", {}).get("status") == "repaired":
                    candidate["layout_repair"] = metadata["layout_repair"]
                    candidate["warnings"].append("rule_anchored_layout_repair_requires_review")
                if method == "native":
                    candidate["native_evidence"] = metadata
                else:
                    image_reasons, evidence = image_evidence(candidate, metadata,
                        backend_details.get("horizontal_rules", []), backend_details.get("figure_context"))
                    candidate.setdefault("evidence", {}).update(evidence)
                    reasons.extend(image_reasons)
            if reasons:
                attempt["rejected"].append({"bbox": candidate["bbox"],
                                            "reasons": sorted(set(reasons))})
                continue
            if method in ("raster", "img2table", "native", "scan_recovery"):
                from raster_structure import map_to_original
                if transform:
                    map_to_original(candidate, transform)
                candidate["warnings"].append(f"experimental_{method}_detection_requires_review")
            expansions = [old for old in tables if contained_grid(old, candidate) and
                          not contained_grid(candidate, old)]
            if method == "native" and candidate.get("native_evidence", {}).get("complete_envelope"):
                # A verified native envelope can include unruled body rows that
                # were absent from a small closed vector fragment. Its new cell
                # partition need not preserve the fragment's accidental grid.
                for old in tables:
                    a, b = old["bbox"], candidate["bbox"]
                    old_area = (a[2] - a[0]) * (a[3] - a[1])
                    new_area = (b[2] - b[0]) * (b[3] - b[1])
                    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
                              max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    if old["method"] == "lines" and overlap >= .95 * old_area and \
                            new_area >= 1.2 * old_area and old not in expansions:
                        expansions.append(old)
            if any(same_table(accepted, candidate) for accepted in tables if accepted not in expansions):
                attempt["rejected"].append({"bbox": candidate["bbox"],
                                            "reasons": ["duplicate_of_preferred_candidate"]})
                continue
            # Prefer a fuller result over an earlier fragment only when the
            # original cell rectangles are preserved exactly in the new grid.
            tables[:] = [old for old in tables if old not in expansions]
            candidate["review_status"] = "unreviewed"
            tables.append(candidate)
            attempt["accepted"] += 1
        attempts.append(attempt)

    def vector(method):
        settings = {"vertical_strategy": method, "horizontal_strategy": method,
                    "snap_tolerance": tolerance, "join_tolerance": tolerance,
                    "intersection_tolerance": tolerance}
        try:
            ingest(region.find_tables(settings), method)
        except MemoryError:
            raise
        except Exception as exc:
            details = {"method": method, "status": "failed", "raw_candidates": 0,
                       "accepted": 0, "rejected": [], "error_type": type(exc).__name__}
            attempts.append(details)
            warnings.append(f"{method}_backend_failed:{type(exc).__name__}")

    def native():
        try:
            from native_structure import page_grids
            groups, details = page_grids(region)
            diagnostics["native_detection"] = details
            warnings.extend(details.get("warnings", []))
            ingest(groups, "native", backend_details=details)
        except MemoryError:
            raise
        except Exception as exc:
            warnings.append(f"native_backend_failed:{type(exc).__name__}")
            diagnostics["native_detection"] = {"status": "failed", "error_type": type(exc).__name__}
            attempts.append({"method": "native", "status": "failed", "raw_candidates": 0,
                             "accepted": 0, "rejected": [], "error_type": type(exc).__name__})

    def image_backend(method, dpi):
        nonlocal image_attempted, boxes
        image_attempted = True
        try:
            if method == "raster":
                from raster_structure import page_grids
            else:
                from img2table_structure import page_grids
            groups, transform, details = page_grids(region, dpi=dpi, tolerance=tolerance)
            if method == "img2table" and region.chars and groups:
                try:
                    from layout_repair import repair_word_groups
                    if boxes is None:
                        boxes = word_boxes_only(region)
                    repaired, notes = repair_word_groups(groups, details.get("horizontal_rules", []), boxes)
                    old_metadata = details.get("returned_group_metadata", [])
                    new_metadata = []
                    for cells, note in zip(repaired, notes):
                        source_index = note["source_group_indices"][0]
                        item = dict(old_metadata[source_index]) if source_index < len(old_metadata) else {}
                        if note["status"] == "repaired":
                            item.update(bbox=note["bbox"], unique_cells=len(cells), layout_repair=note,
                                pixel_evidence={"status": "unavailable", "reason": "native_layout_rebuilt_region"})
                        new_metadata.append(item)
                    groups = repaired
                    details["returned_group_metadata"] = new_metadata
                    details["returned_tables"] = len(groups)
                    details["unique_cells"] = sum(len(group) for group in groups)
                    details["native_layout_repaired_tables"] = sum(note["status"] == "repaired" for note in notes)
                except MemoryError:
                    raise
                except Exception as exc:
                    warnings.append(f"native_layout_repair_failed:{type(exc).__name__}")
            diagnostics[f"{method}_detection"] = details
            warnings.extend(details.get("warnings", []))
            ingest(groups, method, transform, details)
            if method == "img2table" and details.get("figure_context"):
                # Reuse the page's image context to review earlier raster grids.
                # Diagram boxes can form closed cells even though they are not
                # a table. No extra render is needed for this second check.
                from figure_quality import context_evidence, figure_evidence_reasons
                removed = []
                for table in list(tables):
                    if table["method"] != "raster":
                        continue
                    evidence = context_evidence(table["bbox"], details["figure_context"])
                    reasons = figure_evidence_reasons({"diagram_context": evidence})
                    if reasons:
                        tables.remove(table)
                        rejected = {"bbox": table["bbox"], "reasons": reasons,
                                    "stage": "page_figure_context"}
                        removed.append(rejected)
                        for attempt in attempts:
                            if attempt["method"] == "raster":
                                attempt["accepted"] = max(0, attempt["accepted"] - 1)
                                attempt["rejected"].append(rejected)
                diagnostics["raster_context_review"] = {"rejected": removed,
                    "retained": sum(t["method"] == "raster" for t in tables)}
        except MemoryError:
            raise
        except Exception as exc:
            code = f"{method}_backend_failed:{type(exc).__name__}"
            diagnostics[f"{method}_detection"] = {"status": "failed", "warnings": [code]}
            attempts.append({"method": method, "status": "failed", "raw_candidates": 0,
                             "accepted": 0, "rejected": [], "error_type": type(exc).__name__})
            warnings.append(code)

    if strategy in ("auto", "lines", "text"):
        vector("text" if strategy == "text" else "lines")
    if strategy == "native" or (strategy == "auto" and region.chars):
        native()
    if strategy == "raster" or (strategy == "auto" and region.images):
        image_backend("raster", scan_dpi)
    # Accepted candidates, not raw fragments, determine fallback. Image-containing
    # mixed pages also get layout detection so a valid vector table cannot hide
    # another scanned/partially ruled table elsewhere on the page.
    if strategy == "img2table" or (strategy == "auto" and (not tables or region.images)):
        image_backend("img2table", layout_dpi)
    if strategy == "auto" and not tables and region.images and not region.chars:
        rules = diagnostics.get("img2table_detection", {}).get("horizontal_rules", [])
        if rules:
            try:
                from scan_recovery import recover_ruled_groups
                groups, details = recover_ruled_groups(region, rules, dpi=layout_dpi)
                context = diagnostics.get("img2table_detection", {}).get("figure_context")
                if context is not None:
                    details["figure_context"] = context
                diagnostics["scan_recovery"] = details
                warnings.extend(details.get("warnings", []))
                ingest(groups, "scan_recovery", backend_details=details)
            except MemoryError:
                raise
            except Exception as exc:
                warnings.append(f"scan_recovery_failed:{type(exc).__name__}")
    if strategy == "auto" and not tables and region.chars:
        vector("text")
    if strategy == "auto" and tables and not region.images and \
            all(t["method"] in ("lines", "native") for t in tables):
        warnings.append("additional_borderless_regions_not_searched")
    if region.images and not image_attempted:
        warnings.append("use_auto_or_img2table_for_image_content")
    if not tables:
        warnings.append("no_table_candidates_accepted")
    tables.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for index, table in enumerate(tables, 1):
        table["id"] = f"p{page.page_number:03d}_t{index:02d}"
    for attempt in attempts:
        attempt["retained"] = sum(t["method"] == attempt["method"] for t in tables)
    return {"page": page.page_number, "width": page.width, "height": page.height,
            "page_bbox": list(page.bbox), "rotation": page.rotation,
            "status": "candidates_detected" if tables else
                ("needs_fallback" if image_attempted and region.images else "no_candidates"),
            "tables": tables, "warnings": sorted(set(warnings)),
            "detection_attempts": attempts, **diagnostics}


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
    report = {"schema_version": "1.3", "coordinate_system":
              "PDF points; x right, y down; bbox=[x0, top, x1, bottom] in pdfplumber page coordinates",
              "indexing": "page is 1-based; row and column are 0-based",
              "cell_text_in_json": False, "overlays_include_original_page": not args.no_overlays,
              "model_input": "geometry_only", "model_requested": args.ollama_model,
              "strategy": args.strategy, "pages": []}
    source = Path(args.pdf)
    input_digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            input_digest.update(chunk)
    report["input_sha256"] = input_digest.hexdigest()
    code = hashlib.sha256()
    for name in ("table_structure.py", "raster_structure.py", "img2table_structure.py", "layout_repair.py", "figure_quality.py", "native_structure.py", "scan_recovery.py", "scan_headers.py", "ollama_refine.py"):
        code.update((Path(__file__).parent / name).read_bytes())
    report["code_sha256"] = code.hexdigest()
    report["effective_settings"] = {"strategy": args.strategy, "pages": args.pages,
        "bbox": args.bbox, "tolerance_pt": args.tolerance,
        "scan_dpi": getattr(args, "scan_dpi", 144),
        "layout_dpi": getattr(args, "layout_dpi", 200), "overlay_dpi": args.dpi,
        "overlays": not args.no_overlays, "model": args.ollama_model,
        "model_timeout_seconds": args.timeout, "json_schema": not args.json_mode,
        "minimum_rows": 2, "minimum_columns": 2, "maximum_word_cut_fraction": 0.18,
        "maximum_candidate_cells": 4096, "maximum_grid_slots": 65536,
        "minimum_unruled_layout_rows": 4, "duplicate_iou": 0.65,
        "graphic_large_ink_limit": 0.55, "graphic_character_ink_limit": 0.35,
        "rule_anchored_layout_repair": True, "word_x_tolerance_ratio": 0.15,
        "native_envelope_detection": True, "scan_rule_recovery": True}
    report["dependency_versions"] = {}
    for name in ("pdfplumber", "pdfminer.six", "Pillow", "pypdfium2", "img2table",
                 "opencv-contrib-python", "numpy"):
        try:
            report["dependency_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependency_versions"][name] = None
    with pdfplumber.open(args.pdf) as document:
        selected = parse_pages(args.pages, len(document.pages))
        for number in selected:
            page_started = time.perf_counter()
            page = document.pages[number - 1]
            result = detect_page(page, args.strategy, args.bbox, args.tolerance,
                                 getattr(args, "scan_dpi", 144), getattr(args, "layout_dpi", 200))
            result["detection_seconds"] = round(time.perf_counter() - page_started, 4)
            if args.ollama_model:
                from ollama_refine import propose_merges
                boxes = word_boxes_only(page) if any(t["method"] == "text"
                    for t in result["tables"]) else []
                for table in result["tables"]:
                    # Model proposals are useful only for inferred borderless cells.
                    # Actual drawn borders are already stronger evidence.
                    if table["method"] != "text":
                        table["model_review"] = {"status": "skipped",
                            "reason": "model_only_supports_text_merge_candidates", "merges": []}
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
    parser.add_argument("--strategy", choices=["auto", "lines", "native", "text", "raster", "img2table"], default="auto")
    parser.add_argument("--pages", help="1-based selection, e.g. 1,3-5")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X0", "TOP", "X1", "BOTTOM"),
                        help="Restrict detection to a known region, in PDF points")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Line snapping tolerance in points")
    parser.add_argument("--no-overlays", action="store_true", help="Export JSON only; avoid rendering cost")
    parser.add_argument("--dpi", type=int, default=120, help="Local overlay rendering resolution")
    parser.add_argument("--scan-dpi", type=int, default=144,
                        help="Pixel detection resolution (72-200); dimensions capped at 2200 pixels")
    parser.add_argument("--layout-dpi", type=int, default=200,
                        help="img2table layout resolution (72-300); dimensions capped at 2200 pixels")
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
    if not 72 <= args.layout_dpi <= 300:
        parser.error("layout-dpi must be between 72 and 300")
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
