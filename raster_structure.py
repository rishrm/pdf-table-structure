"""Bounded, CPU-only ruled-table detection from page pixels.

Uses Pillow and pdfplumber, already installed with the project. No OCR, text
recognition, model downloads, NumPy, or OpenCV are required. This deliberately
detects closed ruled grids; absence of a result does not establish no table.
"""
from __future__ import annotations

import math
import re

from PIL import Image
from pdfplumber.table import (cells_to_tables, edges_to_intersections,
                             intersections_to_cells, merge_edges)

MAX_RAW_EDGES = 16000
MAX_MERGED_EDGES = 400
MAX_INTERSECTIONS = 2000


def _long_runs(mask, minimum):
    """Score uninterrupted dark runs, favoring long straight lines over pieces."""
    width, height = mask.size
    pixels = mask.tobytes()
    pattern = re.compile(b"\xff{" + str(minimum).encode("ascii") + b",}")
    return sum((match.end() - match.start()) ** 2 for row in range(height)
               for match in pattern.finditer(pixels[row * width:(row + 1) * width]))


def _deskew(mask, scale):
    """Search a bounded +/-2 degree range using long horizontal pixel runs."""
    # Deskew search runs on a smaller image; full-size pixels are used for cells.
    factor = min(1.0, 1000 / max(mask.size))
    small = mask.resize((max(1, round(mask.width * factor)),
                         max(1, round(mask.height * factor))), Image.Resampling.NEAREST)
    minimum = max(24, round(45 * scale * factor))
    original_score = _long_runs(small, minimum)
    best_score, best_angle = original_score, 0.0
    cache = {0.0: original_score}

    def score(angle):
        if angle not in cache:
            rotated = small.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            cache[angle] = _long_runs(rotated, minimum)
        return cache[angle]

    for angle in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        value = score(angle)
        if value > best_score:
            best_score, best_angle = value, angle
    center = best_angle
    for step in range(-3, 4):
        angle = round(center + step * 0.1, 1)
        if not -2 <= angle <= 2:
            continue
        value = score(angle)
        if value > best_score:
            best_score, best_angle = value, angle
    # Avoid choosing a tilt from sparse noise or tiny scoring differences.
    if best_score < max(150, original_score * 1.15):
        best_angle = 0.0
    if best_angle:
        mask = mask.rotate(best_angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    return mask, best_angle


def _segments(mask, minimum, gap, orientation):
    """Find near-solid runs along rows; transpose for vertical line segments."""
    working = mask if orientation == "h" else mask.transpose(Image.Transpose.TRANSPOSE)
    width, height = working.size
    pixels = working.tobytes()
    pattern = re.compile(b"\xff+")
    segments = []
    for position in range(height):
        row = pixels[position * width:(position + 1) * width]
        start = end = ink = None

        def append():
            if start is None or end - start < minimum or ink / (end - start) < 0.94:
                return
            if orientation == "h":
                x0, top, x1, bottom = start, position, end - 1, position
            else:
                x0, top, x1, bottom = position, start, position, end - 1
            segments.append({"orientation": orientation, "x0": float(x0),
                "x1": float(x1), "top": float(top), "bottom": float(bottom),
                "width": float(x1 - x0), "height": float(bottom - top),
                "y0": float(mask.height - bottom), "y1": float(mask.height - top)})

        for match in pattern.finditer(row):
            left, right = match.span()
            if start is not None and left - end <= gap:
                ink += right - left
                end = right
            else:
                append()
                start, end, ink = left, right, right - left
        append()
        if len(segments) > MAX_RAW_EDGES:
            raise ValueError("raster_line_limit")
    return segments


def detect_image_grids(image, *, scale=2.0, threshold=180, tolerance=2.0,
                       deskew=True):
    """Return pixel-coordinate cell groups, applied rotation, and diagnostics."""
    if image.width * image.height > 8_000_000:
        return [], 0.0, {"warnings": ["raster_pixel_limit"]}
    mask = image.convert("L").point(lambda value: 255 if value < threshold else 0)
    mask, angle = _deskew(mask, scale) if deskew else (mask, 0.0)
    diagnostics = {"threshold": threshold, "deskew_degrees": angle,
                   "image_size": list(image.size), "warnings": []}
    try:
        horizontal = _segments(mask, max(16, round(24 * scale)),
                               max(1, round(tolerance * scale)), "h")
        vertical = _segments(mask, max(12, round(12 * scale)),
                             max(1, round(tolerance * scale)), "v")
    except ValueError:
        diagnostics["warnings"].append("raster_line_limit")
        return [], angle, diagnostics
    diagnostics["raw_line_segments"] = len(horizontal) + len(vertical)
    snap = max(1, tolerance * scale)
    edges = merge_edges(horizontal + vertical, snap, snap, snap, snap)
    diagnostics["merged_line_segments"] = len(edges)
    if len(edges) > MAX_MERGED_EDGES:
        diagnostics["warnings"].append("raster_merged_line_limit")
        return [], angle, diagnostics
    intersections = edges_to_intersections(edges, snap, snap)
    diagnostics["intersections"] = len(intersections)
    if len(intersections) > MAX_INTERSECTIONS:
        diagnostics["warnings"].append("raster_intersection_limit")
        return [], angle, diagnostics
    cells = intersections_to_cells(intersections)
    # Reject microscopic rectangles created by stroke artifacts. Do not fill
    # missing cells: incomplete grids remain visible to the existing validator.
    cells = [box for box in cells if box[2] - box[0] >= 6 * scale and
             box[3] - box[1] >= 3 * scale]
    return cells_to_tables(cells), angle, diagnostics


def page_grids(region, *, dpi=144, tolerance=2.0):
    """Rasterize a page/region with bounded dimensions, returning PDF grids.

    bbox and polygons map back to the original page. x/y boundaries describe
    the straightened grid; a transform is explicitly attached for tilted scans.
    """
    longest = max(region.width, region.height)
    effective_dpi = min(dpi, 2200 * 72 / longest)
    rendered = region.to_image(resolution=effective_dpi, antialias=True)
    image = rendered.original
    bounds = rendered.bbox
    scale_x = image.width / (bounds[2] - bounds[0])
    scale_y = image.height / (bounds[3] - bounds[1])
    groups, angle, diagnostics = detect_image_grids(
        image, scale=min(scale_x, scale_y), tolerance=tolerance)
    diagnostics["effective_dpi"] = round(effective_dpi, 2)
    diagnostics["page_region"] = list(bounds)
    grids = [[(box[0] / scale_x + bounds[0], box[1] / scale_y + bounds[1],
               box[2] / scale_x + bounds[0], box[3] / scale_y + bounds[1])
              for box in group] for group in groups]
    transform = {"deskew_degrees": angle, "render_bbox": list(bounds),
                 "image_size": list(image.size), "scale_x": scale_x, "scale_y": scale_y}
    return grids, transform, diagnostics


def map_to_original(table, transform):
    """Map deskewed rectangles back to original PDF points, preserving corners."""
    angle = transform["deskew_degrees"]
    if not angle:
        return
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    bounds = transform["render_bbox"]
    sx, sy = transform["scale_x"], transform["scale_y"]
    cx, cy = [value / 2 for value in transform["image_size"]]

    def polygon(box):
        points = []
        for x, y in [(box[0], box[1]), (box[2], box[1]),
                     (box[2], box[3]), (box[0], box[3])]:
            dx, dy = (x - bounds[0]) * sx - cx, (y - bounds[1]) * sy - cy
            original_x, original_y = cosine * dx - sine * dy + cx, sine * dx + cosine * dy + cy
            points.append([round(original_x / sx + bounds[0], 3),
                           round(original_y / sy + bounds[1], 3)])
        return points

    for item in [table] + table["cells"]:
        points = polygon(item["bbox"])
        item["polygon"] = points
        item["bbox"] = [min(p[0] for p in points), min(p[1] for p in points),
                        max(p[0] for p in points), max(p[1] for p in points)]
    table["grid_transform"] = transform
    table["grid_coordinate_system"] = "deskewed PDF points; bbox/polygon are original page points"
