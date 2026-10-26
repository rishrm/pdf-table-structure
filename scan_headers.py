"""Restore one missing centered summary row inside a printed scan-table frame.

Only geometry is inspected. The body grid is retained, including its existing
merged cells. A repair needs matching printed top/bottom rules, a substantial
unused band above the body, and a single centered text-shaped line isolated
from repeated body records. Captions above the top rule are never included.
"""
from __future__ import annotations

from itertools import islice
import math
import statistics

from layout_repair import _lines, _runs

MAX_GROUPS = 64
MAX_RULES = 256
MAX_CELLS = 4096
MAX_COMPONENTS = 20_000


def _box(value):
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 and all(math.isfinite(v) for v in box) else None


def _boundaries(values):
    groups = []
    for value in sorted(set(values)):
        if groups and value - groups[-1][0] <= .75:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [statistics.mean(group) for group in groups]


def _rules(horizontal):
    result = []
    for raw in islice(horizontal, MAX_RULES):
        box = _box(raw)
        if box and box[2] - box[0] >= 60 and abs(box[3] - box[1]) <= 2:
            result.append((box[0], (box[1] + box[3]) / 2, box[2], (box[1] + box[3]) / 2))
    return sorted(set(result), key=lambda b: (b[1], b[0], b[2]))


def _repair(group, rules, read_components):
    if not 6 <= len(group) <= MAX_CELLS:
        return None, "unsupported_body_grid"
    cells = [_box(raw) for raw in group]
    if any(b is None or b[0] >= b[2] or b[1] >= b[3] for b in cells):
        return None, "invalid_body_geometry"
    xs = _boundaries([b[i] for b in cells for i in (0, 2)])
    ys = _boundaries([b[i] for b in cells for i in (1, 3)])
    if not 3 <= len(xs) <= 17 or not 4 <= len(ys) <= 129:
        return None, "unsupported_body_grid"
    left, top, right, bottom = xs[0], ys[0], xs[-1], ys[-1]
    width = right - left
    if width < 60 or any(b - a < 2 for a, b in zip(ys, ys[1:])):
        return None, "unsupported_body_grid"

    def aligned(rule):
        overlap = min(right, rule[2]) - max(left, rule[0])
        return (.9 * width <= rule[2] - rule[0] <= 1.25 * width and
                overlap >= .95 * width and abs(rule[0] - left) <= max(3, .025 * width) and
                abs(rule[2] - right) <= max(15, .1 * width))

    tops = [rule for rule in rules if aligned(rule) and 8 <= top - rule[1] <= 40]
    bottoms = [rule for rule in rules if aligned(rule) and bottom - 1 <= rule[1] <= bottom + 12]
    if not tops or not bottoms:
        return None, "no_compatible_rule_bracket"
    upper = max(tops, key=lambda b: b[1])
    lower = min(bottoms, key=lambda b: abs(b[1] - bottom))
    if abs(upper[0] - lower[0]) > 3 or abs(upper[2] - lower[2]) > 3:
        return None, "inconsistent_rule_edges"
    if any(upper[1] + 2 < r[1] < top - 1 and
           min(upper[2], r[2]) - max(upper[0], r[0]) >= .75 * width for r in rules):
        return None, "intervening_header_rule"
    # Preserve existing ink at a slightly inset/rounded rule endpoint, while
    # extending the outer value column to the independently printed right edge.
    new_left = min(left, max(upper[0], lower[0]))
    new_right = max(right, min(upper[2], lower[2]))
    new_top, new_bottom = upper[1], lower[1]
    envelope = [new_left, new_top, new_right, new_bottom]
    components = [_box(b) for b in islice(read_components(envelope), MAX_COMPONENTS + 1)]
    if len(components) > MAX_COMPONENTS:
        return None, "component_limit"
    components = [b for b in components if b and b[0] < b[2] and b[1] < b[3] and
                  new_left <= (b[0] + b[2]) / 2 <= new_right and
                  new_top < (b[1] + b[3]) / 2 < new_bottom]
    # Extending the old grid to printed boundaries must add padding, not a new
    # record or field. Header ink above the body is intentional; separate body
    # ink outside its former bounds would change topology and is ambiguous.
    rounding = .75
    body_region = [b for b in components if (b[1] + b[3]) / 2 >= top - rounding]
    if any(b[3] > bottom + rounding for b in body_region):
        return None, "additional_body_ink_below_grid"
    if any(b[0] < left - rounding or b[2] > right + rounding for b in body_region):
        return None, "additional_body_ink_beside_grid"
    header = [b for b in components if b[1] >= new_top + 1 and b[3] <= top - .5]
    body = [b for b in components if top <= (b[1] + b[3]) / 2 <= bottom]
    if len(header) < 6 or len(body) < 12:
        return None, "insufficient_header_or_body_ink"
    header_lines, char_width, header_height = _lines(header)
    body_lines, _, body_height = _lines(body)
    if len(header_lines) != 1 or len(body_lines) < 3:
        return None, "header_not_single_isolated_line"
    if not .65 <= header_height / max(.01, body_height) <= 1.8:
        return None, "header_scale_differs_from_body"
    runs = _runs(header_lines[0], max(2, 2 * char_width))
    if len(runs) != 1:
        return None, "separate_header_labels"
    hx0, hx1 = runs[0]
    full_width = new_right - new_left
    center_offset = abs((hx0 + hx1 - new_left - new_right) / 2)
    if not .12 <= (hx1 - hx0) / full_width <= .75 or center_offset > max(2, .06 * full_width):
        return None, "header_not_centered_across_envelope"
    header_top = min(b[1] for b in header_lines[0])
    header_bottom = max(b[3] for b in header_lines[0])
    first_body_top = min(b[1] for b in body_lines[0])
    if header_top - new_top < max(1, .25 * body_height) or \
            first_body_top - header_bottom < max(1.5, .3 * body_height):
        return None, "header_not_isolated_from_rules_or_body"
    # Every existing column needs independent body ink in multiple row bands.
    for a, b in zip(xs, xs[1:]):
        occupied = sum(any(a <= (c[0] + c[2]) / 2 < b and y0 <= (c[1] + c[3]) / 2 < y1
                           for c in body) for y0, y1 in zip(ys, ys[1:]))
        if occupied < 2:
            return None, "unsupported_body_column"
    if any(b[3] > new_bottom + .5 for b in body):
        return None, "bottom_rule_cuts_body"
    separator = min(top, (header_bottom + first_body_top) / 2)
    result = [(new_left, new_top, new_right, separator)]
    for a, y0, b, y1 in cells:
        result.append((new_left if abs(a - left) <= .75 else a,
                       separator if abs(y0 - top) <= .75 else y0,
                       new_right if abs(b - right) <= .75 else b,
                       new_bottom if abs(y1 - bottom) <= .75 else y1))
    return (result, {"status": "repaired", "basis": "printed_rule_bracket_and_centered_pixel_header",
        "bbox": envelope, "original_bbox": [left, top, right, bottom],
        "header_bbox": [new_left, new_top, new_right, separator],
        "header_rows": 1, "header_rows_added": 1, "body_rows": len(ys) - 1,
        "columns": len(xs) - 1, "header_component_count": len(header),
        "header_center_offset_points": round(center_offset, 3),
        "assumptions": ["single_centered_header_band", "printed_rule_bracket",
                        "outer_expansion_contains_no_new_body_ink", "body_topology_preserved"],
        "semantic_structure_verified": False}), None


def repair_headers(groups, horizontal_rules, read_components):
    """Return groups and aligned notes; preserve ordering and unrepaired inputs.

    ``read_components(bbox)`` supplies bounded glyph-shaped rectangles in the
    same PDF-point coordinate system. A caller can reuse its existing pixel
    masks and crop budget; this function never renders or recognizes text.
    """
    rules = _rules(horizontal_rules)
    result, notes = [], []
    for index, group in enumerate(groups):
        if index >= MAX_GROUPS:
            repaired, reason = None, "header_group_budget"
        else:
            try:
                repaired, reason = _repair(group, rules, read_components)
            except MemoryError:
                raise
            except Exception as exc:
                repaired, reason = None, f"header_components_failed:{type(exc).__name__}"
        if repaired:
            cells, note = repaired
            result.append(cells)
            notes.append(note)
        else:
            result.append(list(group))
            notes.append({"status": "unchanged", "reason": reason})
    return result, notes
