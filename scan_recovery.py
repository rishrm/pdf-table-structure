"""Recover a ruled scan region when the first image extractor found no table.

Only compatible printed top/header/bottom rules can propose a region. Pixel
components must then support repeated body records and persistent columns.
This module does not run OCR or a second TableExtractor pass. Callers still
apply their normal topology, figure-evidence and duplicate checks.
"""
from __future__ import annotations

from itertools import islice
import math
import time
from types import SimpleNamespace

from layout_repair import _lines, _repair, _runs

MAX_RULES = 256
MAX_REGIONS = 16
MAX_LAYOUT_PIXELS = 4_000_000
MAX_EVIDENCE_PIXELS = 4_000_000


def _rule_envelopes(horizontal_rules, region_bbox):
    """Return bounded adjacent triples sharing their physical left/right edges."""
    left, top, right, bottom = region_bbox
    minimum_width = max(75, .3 * (right - left))
    rules = []
    for index, raw in enumerate(islice(horizontal_rules, MAX_RULES)):
        try:
            box = tuple(float(v) for v in raw)
        except (TypeError, ValueError):
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        if box[2] - box[0] < minimum_width or abs(box[3] - box[1]) > 2:
            continue
        y = (box[1] + box[3]) / 2
        if box[0] < left - 2 or box[2] > right + 2 or not top <= y <= bottom:
            continue
        rules.append((index, (max(left, box[0]), y, min(right, box[2]), y)))
    rules.sort(key=lambda item: (item[1][1], item[1][0]))
    envelopes = []
    for first_position, (first_index, first) in enumerate(rules):
        tolerance = max(6, .03 * (first[2] - first[0]))
        compatible = [(first_index, first)]
        for index, rule in rules[first_position + 1:]:
            if abs(rule[0] - first[0]) > tolerance or abs(rule[2] - first[2]) > tolerance:
                continue
            if rule[1] - compatible[-1][1][1] < 3:
                continue
            compatible.append((index, rule))
            if len(compatible) == 3:
                break
        if len(compatible) != 3:
            continue
        header_height = compatible[1][1][1] - first[1]
        body_height = compatible[2][1][1] - compatible[1][1][1]
        if not 6 <= header_height <= 65 or not 30 <= body_height <= 400:
            continue
        triplet = [item[1] for item in compatible]
        bbox = [max(r[0] for r in triplet), first[1],
                min(r[2] for r in triplet), triplet[-1][1]]
        if bbox[2] - bbox[0] < minimum_width:
            continue
        # Nearby double rules should not propose the same region repeatedly.
        if any(all(abs(a - b) <= 3 for a, b in zip(bbox, old["bbox"])) for old in envelopes):
            continue
        envelopes.append({"bbox": bbox, "rules": triplet,
                          "rule_indices": [item[0] for item in compatible]})
        if len(envelopes) >= MAX_REGIONS:
            break
    return envelopes


def _normalize_wrapped_header(cells, note, components, rules):
    """Fit one wrapped header to one body-column footprint when unambiguous.

    Short numeric body values can leave a broad gutter whose midpoint cuts a
    longer column label. Intersect header and body gutters before choosing the
    divider. Grouped headings do not have a one-to-one horizontal footprint and
    therefore retain the existing repair's span/row interpretation.
    """
    if note["header_rows"] <= 1:
        return cells, note
    top, separator = rules[0][1], rules[1][1]
    body = [b for b in components if separator + 1 < (b[1] + b[3]) / 2 < rules[-1][1] - 1]
    header = [b for b in components if top + 1 < (b[1] + b[3]) / 2 < separator - 1]
    body_lines, char_width, char_height = _lines(body)
    header_lines, _, _ = _lines(header)
    if len(header_lines) < 2 or not body_lines:
        return cells, note
    if any(min(b[1] for b in second) - max(b[3] for b in first) > .75 * char_height
           for first, second in zip(header_lines, header_lines[1:])):
        return cells, note
    header_runs = _runs(header, 2 * char_width)
    body_runs = _runs([b for line in body_lines for b in line], 2 * char_width)
    columns = note["columns"]
    if len(header_runs) != columns or len(body_runs) != columns:
        return cells, note
    xs = [note["bbox"][0]]
    for index in range(columns - 1):
        lo = max(header_runs[index][1], body_runs[index][1])
        hi = min(header_runs[index + 1][0], body_runs[index + 1][0])
        if hi - lo < 2:
            return cells, note
        xs.append((lo + hi) / 2)
    xs.append(note["bbox"][2])
    body_cells = [b for b in cells if b[1] >= separator - .01]
    ys = sorted({b[i] for b in body_cells for i in (1, 3)})
    if len(ys) != note["body_rows"] + 1:
        return cells, note
    result = [(a, top, b, separator) for a, b in zip(xs, xs[1:])]
    result += [(a, y0, b, y1) for y0, y1 in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]
    return result, {**note, "header_rows": 1,
        "column_source": "matching_pixel_header_and_body_whitespace",
        "header_normalization": "wrapped_header_with_one_to_one_column_footprint"}


def recover_ruled_groups(region, horizontal_rules, *, dpi=200):
    """Return ``(point_cell_groups, diagnostics)``; no render without a triad."""
    started = time.perf_counter()
    details = {"engine": "scan_rule_recovery", "status": "no_candidates", "warnings": [],
        "horizontal_rules": [], "returned_group_metadata": [], "returned_tables": 0,
        "policy": {"maximum_input_rules": MAX_RULES, "maximum_regions": MAX_REGIONS,
                   "maximum_layout_crop_pixels": MAX_LAYOUT_PIXELS,
                   "maximum_evidence_crop_pixels": MAX_EVIDENCE_PIXELS,
                   "minimum_rule_width_points": 75, "minimum_relative_rule_width": .3,
                   "maximum_header_height_points": 65, "maximum_body_height_points": 400},
        "runtime_flags": {"ocr": False, "content_extraction": False,
                          "cpu_only": True, "second_table_extractor_pass": False}}

    def finish(groups=()):
        details["seconds"] = round(time.perf_counter() - started, 4)
        details["returned_tables"] = len(groups)
        details["status"] = "ok" if groups else details["status"]
        return list(groups), details

    try:
        input_rules = list(islice(horizontal_rules, MAX_RULES + 1))
        details["input_rules_truncated"] = len(input_rules) > MAX_RULES
        envelopes = _rule_envelopes(input_rules[:MAX_RULES], region.bbox)
    except (AttributeError, TypeError, ValueError):
        details["status"] = "failed"
        details["warnings"].append("scan_recovery_invalid_rule_geometry")
        return finish()
    details["proposed_regions"] = len(envelopes)
    details["region_cap_reached"] = len(envelopes) >= MAX_REGIONS
    if not envelopes:
        return finish()
    if not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
        details["status"] = "failed"
        details["warnings"].append("scan_recovery_invalid_dpi")
        return finish()
    try:
        # Lazy imports make the no-triad path cheap and tolerate absent optional
        # native dependencies. The image is rendered once for all proposals.
        from img2table_structure import (_load_backend, _render, _prepare_pixel_evidence,
                                        _layout_components, _pixel_evidence)
        np, _unused_extractor, cv2 = _load_backend()
        image, render_bbox, effective_dpi, _ = _render(region, dpi)
        sx = image.width / (render_bbox[2] - render_bbox[0])
        sy = image.height / (render_bbox[3] - render_bbox[1])
        rules = sorted({tuple(rule) for e in envelopes for rule in e["rules"]}, key=lambda b: (b[1], b[0]))
        details.update(horizontal_rules=[list(rule) for rule in rules],
                       effective_dpi=round(effective_dpi, 3), image_size=list(image.size))
        lines = [SimpleNamespace(x1=(b[0] - render_bbox[0]) * sx, y1=(b[1] - render_bbox[1]) * sy,
                                 x2=(b[2] - render_bbox[0]) * sx, y2=(b[3] - render_bbox[1]) * sy,
                                 thickness=1) for b in rules]
        extractor = SimpleNamespace(horizontal_lines=lines, vertical_lines=[])
        masks = _prepare_pixel_evidence(np.asarray(image, dtype=np.uint8), extractor, sx, sy, np, cv2)
        masks["layout_budget"] = MAX_LAYOUT_PIXELS
        masks["remaining_budget"] = MAX_EVIDENCE_PIXELS
        groups = []
        for envelope in envelopes:
            bbox, local_rules = envelope["bbox"], envelope["rules"]
            components = _layout_components(bbox, masks, render_bbox, sx, sy, cv2)
            if not components:
                continue
            repaired = _repair({"bbox": bbox, "indices": [0]}, [[bbox]], local_rules, [],
                               lambda _area: components)
            if not repaired:
                continue
            cells, note = _normalize_wrapped_header(*repaired, components, local_rules)
            note = {**note, "basis": "three_printed_rules_and_pixel_geometry",
                    "source_rule_indices": envelope["rule_indices"]}
            note.pop("source_group_indices", None)
            pixel_cells = [((b[0] - render_bbox[0]) * sx, (b[1] - render_bbox[1]) * sy,
                            (b[2] - render_bbox[0]) * sx, (b[3] - render_bbox[1]) * sy) for b in cells]
            evidence = _pixel_evidence(pixel_cells, masks, sx, sy, np, cv2)
            groups.append(cells)
            details["returned_group_metadata"].append({"bbox": note["bbox"], "unique_cells": len(cells),
                "layout_repair": note, "pixel_evidence": evidence, "source_candidate_index": None})
        details["layout_crop_pixels_used"] = MAX_LAYOUT_PIXELS - masks["layout_budget"]
        details["evidence_crop_pixels_used"] = MAX_EVIDENCE_PIXELS - masks["remaining_budget"]
        return finish(groups)
    except MemoryError:
        raise
    except Exception as exc:
        details["status"] = "failed"
        details["warnings"].append(f"scan_recovery_failed:{type(exc).__name__}")
        return finish()
