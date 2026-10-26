"""Complete native-text tables whose shaded cells encode column partitions.

This bounded geometry path uses filled PDF rectangles, a compatible bottom
rule (or terminal shaded band), and word positions. It never interprets or
exports document text. Plain prose and single-column callout boxes do not
provide the required native column evidence.
"""
from __future__ import annotations

import math
import statistics

MAX_RECTS = 3000
MAX_LINES = 6000
MAX_CHARS = 100_000
MAX_WORDS = 20000
MAX_CANDIDATES = 32
MAX_ROWS = 256
TOLERANCE = .8


def _cluster(values, tolerance=TOLERANCE):
    groups = []
    for value in sorted(set(values)):
        if groups and value - groups[-1][0] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(g) / len(g) for g in groups]


def _color(value):
    if isinstance(value, (list, tuple)):
        return tuple(round(float(v), 3) for v in value)
    if isinstance(value, (float, int)):
        return (round(float(value), 3),)
    return None


def _nonwhite(value):
    if value is None:
        return False
    return any(v > .01 for v in value) if len(value) == 4 else any(v < .99 for v in value)


def _box(obj):
    result = tuple(float(obj[k]) for k in ("x0", "top", "x1", "bottom"))
    return result if all(math.isfinite(v) for v in result) else None


def _bands(rects):
    """Group adjacent filled rectangles with common top/bottom into row bands."""
    buckets = []
    for rect in rects:
        box = _box(rect)
        color = _color(rect.get("non_stroking_color"))
        if not box or not rect.get("fill", True) or not _nonwhite(color):
            continue
        if box[2] - box[0] < 8 or not 3 <= box[3] - box[1] <= 700:
            continue
        bucket = next((b for b in buckets if abs(b[0][0][1] - box[1]) <= TOLERANCE
                       and abs(b[0][0][3] - box[3]) <= TOLERANCE), None)
        if bucket is None:
            buckets.append([(box, color)])
        else:
            bucket.append((box, color))
    result = []
    for bucket in buckets:
        groups = []
        for box, color in sorted(bucket):
            if groups and abs(box[0] - groups[-1][-1][0][2]) <= TOLERANCE:
                groups[-1].append((box, color))
            else:
                groups.append([(box, color)])
        for group in groups:
            boxes, colors = zip(*group)
            result.append({"left": boxes[0][0], "right": boxes[-1][2],
                "top": statistics.mean(b[1] for b in boxes),
                "bottom": statistics.mean(b[3] for b in boxes),
                "segments": list(boxes), "color": colors[0] if len(set(colors)) == 1 else None})
    return sorted(result, key=lambda b: (b["top"], b["left"]))


def _same_extent(first, second):
    return abs(first["left"] - second["left"]) <= 2 and abs(first["right"] - second["right"]) <= 2


def _horizontal_rules(region):
    pieces = []
    for line in getattr(region, "lines", []):
        box = _box(line)
        if box and abs(box[3] - box[1]) <= 1 and box[2] - box[0] >= 8:
            pieces.append((box[0], (box[1] + box[3]) / 2, box[2]))
    for rect in getattr(region, "rects", []):
        box = _box(rect)
        if box and 0 < box[3] - box[1] <= 1 and box[2] - box[0] >= 8:
            pieces.append((box[0], (box[1] + box[3]) / 2, box[2]))
    rules = []
    for y in _cluster([v[1] for v in pieces], 1):
        runs = []
        for left, _, right in sorted(v for v in pieces if abs(v[1] - y) <= 1):
            if runs and left <= runs[-1][1] + 2:
                runs[-1][1] = max(runs[-1][1], right)
            else:
                runs.append([left, right])
        rules.extend((left, y, right) for left, right in runs)
    return rules


def _inside(words, left, top, right, bottom):
    return [b for b in words if left - .5 <= (b[0] + b[2]) / 2 <= right + .5
            and top <= (b[1] + b[3]) / 2 < bottom]


def _text_lines(words):
    if not words:
        return []
    height = statistics.median(b[3] - b[1] for b in words)
    lines = []
    for box in sorted(words, key=lambda b: (b[1] + b[3]) / 2):
        center = (box[1] + box[3]) / 2
        if lines and abs(center - statistics.median((b[1] + b[3]) / 2 for b in lines[-1])) < .6 * height:
            lines[-1].append(box)
        else:
            lines.append([box])
    return lines


def _left_blocks(words, left, right, top, bottom):
    lines = _text_lines(_inside(words, left, top, right, bottom))
    blocks = []
    for line in lines:
        line_top, line_bottom = min(b[1] for b in line), max(b[3] for b in line)
        height = statistics.median(b[3] - b[1] for b in line)
        if blocks and line_top - blocks[-1]["bottom"] <= .5 * height:
            blocks[-1]["bottom"] = max(blocks[-1]["bottom"], line_bottom)
        else:
            blocks.append({"top": line_top, "bottom": line_bottom})
    return blocks


def _spans_columns(words, xs, height):
    """A continuous heading crosses a body gutter; independent cells do not."""
    if _inside(words, xs[-2], -math.inf, xs[-1], math.inf):
        return False
    for line in _text_lines(words):
        runs = []
        for box in sorted(line):
            if runs and box[0] - runs[-1][1] <= max(2, height * .5):
                runs[-1][1] = max(runs[-1][1], box[2])
            else:
                runs.append([box[0], box[2]])
        if any(a < boundary - 1 and b > boundary + 1 for a, b in runs for boundary in xs[1:-1]):
            return True
    return False


def _complete(header, bands, rules, words, page_bottom):
    left, right, top = header["left"], header["right"], header["top"]
    header_bands = [header]
    for band in bands:
        if band is header or not _same_extent(band, header):
            continue
        if abs(band["top"] - header_bands[-1]["bottom"]) <= TOLERANCE and \
                band["color"] == header["color"] and band["bottom"] - top <= 100:
            header_bands.append(band)
    header_bottom = header_bands[-1]["bottom"]
    xs = _cluster([left, right] + [b[i] for band in header_bands for b in band["segments"] for i in (0, 2)])
    if not 3 <= len(xs) <= 33:
        return None
    later_headers = [b["top"] for b in bands if _same_extent(b, header) and len(b["segments"]) >= 2
        and b["color"] == header["color"] and b["top"] > header_bottom + 2]
    limit = min(later_headers, default=page_bottom)
    body_bands = [b for b in bands if _same_extent(b, header) and
                  b["top"] >= header_bottom - TOLERANCE and b["bottom"] < limit + TOLERANCE and
                  b not in header_bands]
    ends = [y for x0, y, x1 in rules if abs(x0 - left) <= 2 and abs(x1 - right) <= 2
            and header_bottom + 10 < y < limit + TOLERANCE]
    if len(ends) > 1:
        # Fully ruled layouts belong to the existing vector reconstruction.
        return None
    bottom = ends[0] if ends else max((b["bottom"] for b in body_bands), default=header_bottom)
    if bottom - header_bottom < 15 or bottom - top > 750:
        return None
    if not ends and _inside(words, left, bottom + 2, right, page_bottom - 40):
        # A final colored row alone does not prove the table ends there if more
        # native text continues in the same column below it.
        return None
    body_bands = [b for b in body_bands if b["bottom"] <= bottom + TOLERANCE]
    local_words = _inside(words, left, top, right, bottom)
    header_words = _inside(local_words, left, top, right, header_bottom)
    body_words = _inside(local_words, left, header_bottom, right, bottom)
    if len(body_words) < 4 or sum(bool(_inside(header_words, a, top, b, header_bottom))
                                 for a, b in zip(xs, xs[1:])) < 2:
        return None
    height = statistics.median(b[3] - b[1] for b in local_words)
    padding = statistics.median(max(0, min((b[1] for b in _inside(words, left, band["top"], right, band["bottom"])),
        default=band["top"]) - band["top"]) for band in header_bands)
    padding = min(height, max(1, padding))
    cells = []
    for band in header_bands:
        for box in band["segments"]:
            a = min(xs, key=lambda x: abs(x - box[0]))
            b = min(xs, key=lambda x: abs(x - box[2]))
            cells.append((a, band["top"], b, band["bottom"]))
    stops = _cluster([header_bottom, bottom] + [b[k] for b in body_bands for k in ("top", "bottom")])
    rows = []
    for start, end in zip(stops, stops[1:]):
        if end - start < 2:
            continue
        covering = next((b for b in body_bands if abs(b["top"] - start) <= TOLERANCE
                         and abs(b["bottom"] - end) <= TOLERANCE), None)
        if covering:
            ink = _inside(local_words, left, start, right, end)
            occupied = sum(bool(_inside(ink, a, start, b, end)) for a, b in zip(xs, xs[1:]))
            # A publisher can paint an ordinary data row with one background
            # rectangle. Its independent column text still defines separate
            # cells; only a sparse or gutter-crossing heading supports a span.
            merged = len(covering["segments"]) == 1 and \
                (occupied <= 1 or _spans_columns(ink, xs, height))
            rows.append({"top": start, "bottom": end, "merge": merged})
            continue
        blocks = _left_blocks(local_words, xs[0], xs[1], start, end)
        if not blocks:
            if rows:
                rows[-1]["bottom"] = end
                continue
            return None
        first_start = start
        preceding_ink = [w for w in _inside(local_words, left, start, right, blocks[0]["top"])
                         if (w[0] + w[2]) / 2 >= xs[1]]
        if rows and preceding_ink and blocks[0]["top"] - start > 2 * height:
            first_start = max(start, blocks[0]["top"] - padding)
            rows[-1]["bottom"] = first_start
        boundaries = [first_start] + [b["top"] - padding for b in blocks[1:]] + [end]
        for row_top, row_bottom in zip(boundaries, boundaries[1:]):
            ink = _inside(local_words, left, row_top, right, row_bottom)
            rows.append({"top": row_top, "bottom": row_bottom,
                         "merge": _spans_columns(ink, xs, height)})
    if not 2 <= len(rows) <= MAX_ROWS:
        return None
    populated_rows = 0
    cuts, total = 0, 0
    for row in rows:
        if row["bottom"] - row["top"] < 2:
            return None
        ink = _inside(local_words, left, row["top"], right, row["bottom"])
        total += len(ink)
        if row["merge"]:
            cells.append((left, row["top"], right, row["bottom"]))
        else:
            occupancy = [bool(_inside(ink, a, row["top"], b, row["bottom"])) for a, b in zip(xs, xs[1:])]
            populated_rows += all(occupancy)
            cuts += sum(any(w[0] < x - .5 and w[2] > x + .5 for x in xs[1:-1]) for w in ink)
            cells.extend((a, row["top"], b, row["bottom"]) for a, b in zip(xs, xs[1:]))
    if populated_rows < 2 or cuts / max(1, total) > .02 or len(cells) > 4096:
        return None
    return cells, {"status": "completed", "complete_envelope": True,
        "basis": "native_shaded_cell_partitions_and_word_blocks",
        "bbox": [left, top, right, bottom], "header_bands": len(header_bands),
        "shaded_body_bands": len(body_bands),
        "source_rectangle_count": sum(len(b["segments"]) for b in header_bands + body_bands),
        "column_partitions": xs,
        "body_rows": len(rows), "columns": len(xs) - 1,
        "printed_bottom_rule": bool(ends), "word_cut_fraction": round(cuts / max(1, total), 6),
        "semantic_structure_verified": False}


def page_grids(region):
    """Return cell groups and aligned diagnostics, using native geometry only."""
    diagnostics = {"status": "no_candidates", "returned_group_metadata": [], "warnings": []}
    chars = getattr(region, "chars", [])
    if not chars:
        diagnostics["status"] = "no_native_text"
        return [], diagnostics
    rects = getattr(region, "rects", [])
    if len(rects) > MAX_RECTS or len(getattr(region, "lines", [])) > MAX_LINES or len(chars) > MAX_CHARS:
        diagnostics.update(status="budget_exceeded", warnings=["native_geometry_limit"])
        return [], diagnostics
    bands = _bands(rects)
    headers = [b for b in bands if len(b["segments"]) >= 2 and b["right"] - b["left"] >= 60
               and 5 <= b["bottom"] - b["top"] <= 75 and b["color"] is not None]
    if not headers:
        return [], diagnostics
    raw_words = region.extract_words(x_tolerance_ratio=.15)
    if len(raw_words) > MAX_WORDS:
        diagnostics.update(status="budget_exceeded", warnings=["native_word_limit"])
        return [], diagnostics
    words = [_box(w) for w in raw_words]
    words = [b for b in words if b and b[0] < b[2] and b[1] < b[3]]
    rules = _horizontal_rules(region)
    groups, notes = [], []
    for header in headers[:MAX_CANDIDATES]:
        if any(n["bbox"][0] - 2 <= header["left"] <= header["right"] <= n["bbox"][2] + 2
               and n["bbox"][1] <= header["top"] < n["bbox"][3] for n in notes):
            continue
        result = _complete(header, bands, rules, words, region.bbox[3])
        if result:
            cells, note = result
            groups.append(cells)
            notes.append(note)
    diagnostics.update(status="ok" if groups else "no_candidates", native_bands=len(bands),
        candidate_headers=len(headers), returned_tables=len(groups), returned_group_metadata=notes)
    return groups, diagnostics
