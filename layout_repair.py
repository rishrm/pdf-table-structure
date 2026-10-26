"""Conservative rule-anchored repair of partially ruled table geometry.

Inputs are rectangles and printed rules in PDF points. No text recognition or
lexical content is used. A repair needs two compatible header rules, at least
three repeated body records, and supported column separators. Ambiguous cases
keep the original candidates.
"""
from __future__ import annotations

import statistics

MAX_REPAIR_CLUSTERS = 64
MAX_COMPONENTS = 20_000


def bounds(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def _clusters(groups):
    clusters = []
    for index, group in enumerate(groups):
        box = bounds(group)
        matches = [c for c in clusters if _overlap(box, c["bbox"]) >=
                   .7 * min(box[2] - box[0], c["bbox"][2] - c["bbox"][0]) and
                   max(box[1], c["bbox"][1]) - min(box[3], c["bbox"][3]) <= 12]
        if matches:
            members = [index] + [i for c in matches for i in c["indices"]]
            clusters = [c for c in clusters if c not in matches]
            box = bounds([box] + [c["bbox"] for c in matches])
        else:
            members = [index]
        clusters.append({"indices": sorted(members), "bbox": box})
    return clusters


def _lines(boxes, minimum_boxes=3):
    if not boxes:
        return [], 0, 0
    height = statistics.median(b[3] - b[1] for b in boxes)
    width = statistics.median(b[2] - b[0] for b in boxes)
    lines = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        center = (box[1] + box[3]) / 2
        if lines and abs(center - statistics.median((b[1] + b[3]) / 2
                                                    for b in lines[-1])) < height * .6:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [line for line in lines if len(line) >= minimum_boxes], width, height


def _runs(boxes, gap):
    runs = []
    for box in sorted(boxes):
        if runs and box[0] - runs[-1][1] < gap:
            runs[-1][1] = max(runs[-1][1], box[2])
        else:
            runs.append([box[0], box[2]])
    return runs


def _repair(cluster, groups, horizontal, vertical, read_components):
    seed = cluster["bbox"]
    width = seed[2] - seed[0]
    rules = [r for r in horizontal if r[2] - r[0] >= max(75, width * .8) and
             _overlap(r, seed) >= .8 * width and
             seed[1] - 50 <= r[1] <= seed[3] + 12]
    rules.sort(key=lambda r: r[1])
    unique = []
    for rule in rules:
        if unique and rule[1] - unique[-1][1] < 3:
            continue
        unique.append(rule)
    if len(unique) < 2:
        return None
    top, separator = unique[:2]
    if not (6 <= separator[1] - top[1] <= 65 and
            top[1] <= seed[1] + min(25, .4 * (seed[3] - seed[1])) and
            separator[1] < seed[3] - 15):
        return None
    left, right = max(top[0], separator[0]), min(top[2], separator[2])
    if right - left > width * 1.7 or right <= left:
        return None
    bottom_rules = [r for r in unique[2:] if r[1] >= seed[3] - 20 and
                    _overlap(r, [left, 0, right, 0]) >= .95 * (right - left)]
    bottom = bottom_rules[0][1] if bottom_rules else seed[3] + 3
    # A fully ruled body already supplies stronger evidence than this repair.
    if any(left - 2 <= v[0] <= right + 2 and
           min(v[3], bottom) - max(v[1], separator[1]) > 4
           for v in vertical):
        return None
    area = [left, top[1], right, bottom]
    components = read_components(area)
    if not components or len(components) > MAX_COMPONENTS:
        return None
    header = [b for b in components if top[1] + 1 < (b[1] + b[3]) / 2 < separator[1] - 1]
    body = [b for b in components if separator[1] + 1 < (b[1] + b[3]) / 2 < bottom - 1]
    body_lines, char_width, char_height = _lines(body)
    header_lines, _, _ = _lines(header)
    if not 3 <= len(body_lines) <= 200 or not 1 <= len(header_lines) <= 6 or char_width <= 0:
        return None
    vertical_x = sorted({v[0] for v in vertical if left + 3 < v[0] < right - 3 and
                         v[1] <= top[1] + 3 and v[3] >= separator[1] - 3})
    vertical_x = [x for i, x in enumerate(vertical_x) if i == 0 or x - vertical_x[i - 1] > 2]
    if vertical_x:
        xs = [left] + vertical_x + [right]
        column_source = "printed_header_dividers"
    else:
        # Union whole text-shaped runs over all body lines. A separator must be
        # persistent whitespace; spaces inside one numeric value are insufficient.
        intervals = [r for line in body_lines for r in _runs(line, 2 * char_width)]
        occupied = _runs([[a, 0, b, 1] for a, b in intervals], 0)
        gaps = [(a[1], b[0]) for a, b in zip(occupied, occupied[1:])
                if b[0] - a[1] >= max(4, 2 * char_width)]
        xs = [left] + [(a + b) / 2 for a, b in gaps] + [right]
        # The lowest header line provides separate column labels. Move a
        # divider within its body gutter if that avoids cutting a header label.
        labels = _runs(header_lines[-1], 2 * char_width)
        for index, (a, b) in enumerate(gaps, 1):
            nearby = [r for r in labels if r[0] < b and r[1] > a]
            for label in nearby:
                if label[0] < xs[index] < label[1]:
                    alternatives = [r for r in labels if r[1] <= label[0]]
                    if alternatives and a <= (alternatives[-1][1] + label[0]) / 2 <= b:
                        xs[index] = (alternatives[-1][1] + label[0]) / 2
        column_source = "persistent_body_whitespace"
    if not 3 <= len(xs) <= 33 or any(b - a < 4 for a, b in zip(xs, xs[1:])):
        return None
    old_columns = max(len({round(b[i], 1) for b in groups[index] for i in (0, 2)}) - 1
                      for index in cluster["indices"])
    if not vertical_x and len(xs) - 1 < old_columns:
        return None
    anchor = [i for i, line in enumerate(body_lines) if
              any(left <= (b[0] + b[2]) / 2 < xs[1] for b in line)]
    if len(anchor) < 3 or anchor[0] != 0 or len(anchor) < .4 * len(body_lines):
        return None
    records = [sum(body_lines[start:end], []) for start, end in
               zip(anchor, anchor[1:] + [len(body_lines)])]
    # Every inferred column must have body ink in multiple independent records.
    if any(sum(any(xs[c] <= (b[0] + b[2]) / 2 < xs[c + 1] for b in record)
               for record in records) < 2 for c in range(len(xs) - 1)):
        return None
    if not bottom_rules:
        bottom = max(b[3] for b in records[-1]) + min(3, char_height / 2)
    body_ys = [separator[1]] + [(max(b[3] for b in first) + min(b[1] for b in second)) / 2
        for first, second in zip(records, records[1:])] + [bottom]
    if any(b - a < 3 for a, b in zip(body_ys, body_ys[1:])):
        return None
    # Printed full-height header dividers indicate wrapped labels in a single
    # header row. Without them, a heading crossing body dividers supports spans.
    grouped = not vertical_x and len(header_lines) > 1 and any(
        any(a + 1 < x < b - 1 for x in xs[1:-1])
        for line in header_lines[:-1] for a, b in _runs(line, 2 * char_width))
    if grouped:
        header_ys = [top[1]] + [(max(b[3] for b in first) + min(b[1] for b in second)) / 2
            for first, second in zip(header_lines, header_lines[1:])] + [separator[1]]
        header_bands = header_lines
    else:
        header_ys, header_bands = [top[1], separator[1]], [header]
    cells = []
    for row, band in enumerate(header_bands):
        joins = set()
        if grouped and row < len(header_bands) - 1:
            for a, b in _runs(band, 2 * char_width):
                joins.update(i for i, x in enumerate(xs[1:-1], 1) if a + 1 < x < b - 1)
        edges = [0] + [i for i in range(1, len(xs) - 1) if i not in joins] + [len(xs) - 1]
        cells.extend((xs[a], header_ys[row], xs[b], header_ys[row + 1])
                     for a, b in zip(edges, edges[1:]))
    cells.extend((a, top_y, b, bottom_y) for top_y, bottom_y in zip(body_ys, body_ys[1:])
                 for a, b in zip(xs, xs[1:]))
    return cells, {"status": "repaired", "basis": "printed_rules_and_pixel_geometry",
        "column_source": column_source, "header_rows": len(header_bands),
        "body_rows": len(records), "body_text_lines": len(body_lines),
        "columns": len(xs) - 1, "bottom_rule": bool(bottom_rules),
        "source_group_indices": cluster["indices"], "bbox": [left, top[1], right, bottom],
        "semantic_structure_verified": False}


def repair_groups(groups, horizontal, vertical, read_components):
    """Return groups and aligned repair notes; preserve inputs on no evidence."""
    result, notes = [], []
    for index, cluster in enumerate(_clusters(groups)):
        repaired = _repair(cluster, groups, horizontal, vertical, read_components) \
            if index < MAX_REPAIR_CLUSTERS else None
        if repaired:
            cells, note = repaired
            result.append(cells)
            notes.append(note)
        else:
            for original in cluster["indices"]:
                result.append(groups[original])
                notes.append({"status": "unchanged", "source_group_indices": [original]})
    return result, notes


def repair_word_groups(groups, horizontal, words):
    """Recover a ruled table envelope using selectable-word positions only.

    Repeated occupancy in the non-label columns identifies records, so wrapped
    row labels can remain in the same cell. A matching header column pattern
    and printed bottom rule are mandatory; words are never interpreted.
    """
    result, notes = [], []
    for cluster_index, cluster in enumerate(_clusters(groups)):
        if cluster_index >= MAX_REPAIR_CLUSTERS:
            for original in cluster["indices"]:
                result.append(groups[original])
                notes.append({"status": "unchanged", "source_group_indices": [original]})
            continue
        seed = cluster["bbox"]
        width = seed[2] - seed[0]
        rules = sorted([r for r in horizontal if r[2] - r[0] >= .85 * width and
            _overlap(r, seed) >= .9 * width and seed[1] - 200 <= r[1] <= seed[3] + 12],
            key=lambda r: r[1])
        bottoms = [r for r in rules if abs(r[1] - seed[3]) <= 12]
        repaired = None
        if bottoms:
            bottom = min(bottoms, key=lambda r: abs(r[1] - seed[3]))
            for top, separator in zip(rules, rules[1:]):
                if not (8 <= separator[1] - top[1] <= 60 and
                        separator[1] <= seed[1] + min(60, .5 * (seed[3] - seed[1])) and
                        separator[1] < bottom[1] - 30):
                    continue
                left = max(top[0], separator[0], bottom[0])
                right = min(top[2], separator[2], bottom[2])
                if right - left < .85 * width:
                    continue
                inside = [b for b in words if left - 2 <= (b[0] + b[2]) / 2 <= right + 2]
                header = [b for b in inside if top[1] < (b[1] + b[3]) / 2 < separator[1]]
                body = [b for b in inside if separator[1] < (b[1] + b[3]) / 2 < bottom[1]]
                if not header or not body or len(body) > MAX_COMPONENTS:
                    continue
                lines, _, height = _lines(body, minimum_boxes=1)
                if not 3 <= len(lines) <= 200:
                    continue
                gap = max(4, height * .5)
                header_columns, body_columns = _runs(header, gap), _runs(body, gap)
                if not 3 <= len(body_columns) <= 32 or len(header_columns) != len(body_columns):
                    continue
                xs = [left] + [(a[1] + b[0]) / 2 for a, b in zip(body_columns, body_columns[1:])] + [right]
                # The complete header must agree with the body-derived columns.
                if any(a < x < b for a, b in header_columns for x in xs[1:-1]):
                    continue
                occupied = [set(c for c in range(len(xs) - 1) if any(
                    xs[c] <= (b[0] + b[2]) / 2 < xs[c + 1] for b in line)) for line in lines]
                # A majority of columns can still contain continuation text.
                # Repeated starts spanning every column provide stronger anchors;
                # sparse/ambiguous layouts retain their original candidate.
                anchors = [i for i, cols in enumerate(occupied) if len(cols) == len(xs) - 1]
                if len(anchors) < 3 or anchors[0] != 0:
                    continue
                # A row with one blank value can resemble wrapped prose. Do not
                # absorb a dense, leading-label line when its spacing suggests
                # a separate record. Both checks are geometric; no labels or
                # cell values are interpreted.
                starts = [min(b[1] for b in line) for line in lines]
                record_pitch = min(starts[b] - starts[a] for a, b in zip(anchors, anchors[1:]))
                ambiguous = False
                previous_anchor = 0
                for index, cols in enumerate(occupied):
                    if index in anchors:
                        previous_anchor = index
                    elif 0 in cols and len(cols) >= max(2, (len(xs) - 1) * .5):
                        gap_before = starts[index] - max(b[3] for b in lines[index - 1])
                        if gap_before > height * .5 + .5 or \
                                starts[index] - starts[previous_anchor] >= .8 * record_pitch:
                            ambiguous = True
                            break
                if ambiguous:
                    continue
                records = [sum(lines[a:b], []) for a, b in zip(anchors, anchors[1:] + [len(lines)])]
                if any(not any(xs[0] <= (b[0] + b[2]) / 2 < xs[1] for b in record) for record in records):
                    continue
                ys = [top[1], separator[1]] + [
                    (max(b[3] for b in a) + min(b[1] for b in b_line)) / 2
                    for a, b_line in zip(records, records[1:])] + [bottom[1]]
                if any(b - a < 3 for a, b in zip(ys, ys[1:])):
                    continue
                cells = [(a, c, b, d) for c, d in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]
                repaired = cells, {"status": "repaired", "basis": "printed_rules_and_word_positions",
                    "column_source": "matching_header_and_body_whitespace", "header_rows": 1,
                    "body_rows": len(records), "body_text_lines": len(lines), "columns": len(xs) - 1,
                    "bottom_rule": True, "source_group_indices": cluster["indices"],
                    "bbox": [left, top[1], right, bottom[1]], "semantic_structure_verified": False}
                break
        if repaired:
            cells, note = repaired
            result.append(cells)
            notes.append(note)
        else:
            for original in cluster["indices"]:
                result.append(groups[original])
                notes.append({"status": "unchanged", "source_group_indices": [original]})
    return result, notes
