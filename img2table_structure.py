"""Local, OCR-free img2table 2.0.0 cell geometry adapter.

Only ``TableExtractor`` is used: the document/OCR/content-extraction API is
never entered. RGB pixels remain in this process. Returned cells are raw
rectangles for the caller to validate, including repeated merged-cell regions
deduplicated by coordinates and empty cells retained without content checks.
"""
from __future__ import annotations

from importlib import metadata
import math
import time

from PIL import Image
from figure_quality import FIGURE_EVIDENCE_POLICY, context_evidence

SUPPORTED_VERSION = "2.0.0"
MAX_RENDER_DIMENSION = 2200
MAX_CELL_POSITIONS = 100_000
MAX_TABLE_CANDIDATES = 256
MAX_RULE_HINTS = 20_000
MAX_EVIDENCE_CROP_PIXELS = 1_000_000
MAX_EVIDENCE_TOTAL_PIXELS = 4_000_000
MAX_EVIDENCE_COMPONENTS = 50_000
MAX_CONTEXT_FRAMES = 128
MAX_CONTEXT_SHAPES = 128
MAX_CONNECTOR_SAMPLE_PIXELS = 2_000_000
INSTALL_ACTION = 'Install optional backend with: python -m pip install "img2table==2.0.0"'
EXTRACT_FLAGS = {"implicit_rows": True, "implicit_columns": True,
                 "borderless_tables": True}


def _load_backend():
    """Import lazily so an unavailable optional backend cannot break fallback."""
    import cv2
    # Keep image processing on one CPU thread for the low-power route. The
    # pinned release uses Cython metrics, with no Numba/parallel worker pool.
    # Zero disables parallel regions even on macOS GCD builds, where setting
    # one can still report/use the system CPU count.
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if not hasattr(getattr(cv2, "ximgproc", None), "niBlackThreshold"):
        raise ImportError("img2table requires opencv-contrib-python")
    import numpy as np
    from img2table.tables.extractor import TableExtractor
    return np, TableExtractor, cv2


def _render(region, dpi):
    """Bound the full render, since pdfplumber crops *after* rendering a page."""
    root = getattr(region, "root_page", region)
    root_bounds = getattr(root, "cropbox", getattr(root, "bbox", None))
    dimensions = [float(region.width), float(region.height)]
    if root_bounds is not None:
        dimensions += [float(root_bounds[2] - root_bounds[0]),
                       float(root_bounds[3] - root_bounds[1])]
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("invalid_render_dimensions")
    effective_dpi = min(float(dpi), MAX_RENDER_DIMENSION * 72 / max(dimensions))
    rendered = region.to_image(resolution=effective_dpi, antialias=True)
    bounds = tuple(float(value) for value in rendered.bbox)
    if (len(bounds) != 4 or not all(math.isfinite(value) for value in bounds)
            or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
        raise ValueError("invalid_render_bounds")
    image = rendered.original
    original_size = list(image.size)
    # PDFium rounds render dimensions to pixels. Also defend against a renderer
    # returning a larger image than requested before handing it to img2table.
    if max(image.size) > MAX_RENDER_DIMENSION:
        ratio = MAX_RENDER_DIMENSION / max(image.size)
        image = image.resize((max(1, round(image.width * ratio)),
                              max(1, round(image.height * ratio))),
                             Image.Resampling.LANCZOS)
    if min(image.size) <= 0:
        raise ValueError("empty_render")
    return image.convert("RGB"), bounds, effective_dpi, original_size


def _raw_cells(tables):
    """Keep the library's merged rectangles rather than synthesizing slot cells."""
    if len(tables) > MAX_TABLE_CANDIDATES:
        raise ValueError("candidate_limit")
    groups, positions, invalid, source_indices = [], 0, 0, []
    for source_index, table in enumerate(tables):
        unique = set()
        malformed = False
        for row in table.rows:
            for cell in row.cells:
                positions += 1
                if positions > MAX_CELL_POSITIONS:
                    raise ValueError("cell_position_limit")
                # Do not inspect cell.content, get_content(), or extracted_table().
                box = tuple(float(getattr(cell, name)) for name in ("x1", "y1", "x2", "y2"))
                if (not all(math.isfinite(value) for value in box)
                        or box[0] >= box[2] or box[1] >= box[3]):
                    malformed = True
                unique.add(box)
        if malformed or not unique:
            invalid += 1
            continue
        groups.append(sorted(unique, key=lambda box: (box[1], box[0], box[3], box[2])))
        source_indices.append(source_index)
    return groups, positions, invalid, source_indices


def _horizontal_rules(extractor, bounds, sx, sy):
    """Map detected pixel rule endpoints without consulting table content."""
    lines = getattr(extractor, "horizontal_lines", [])
    unique = set()
    for line in lines[:MAX_RULE_HINTS]:
        box = tuple(float(getattr(line, name)) for name in ("x1", "y1", "x2", "y2"))
        if not all(math.isfinite(value) for value in box) or box[0] >= box[2]:
            continue
        unique.add((box[0] / sx + bounds[0], box[1] / sy + bounds[1],
                    box[2] / sx + bounds[0], box[3] / sy + bounds[1]))
    return [list(box) for box in sorted(unique, key=lambda box: (box[1], box[0], box[2]))], \
        len(lines) > MAX_RULE_HINTS


def _diagram_context(foreground, sx, sy, np, cv2):
    """Find connected box networks and distributed panel shapes once per page.

    Closed rectangular holes distinguish box interiors from glyphs. A flow
    network needs thin ink bridges through whitespace between box interiors,
    rather than ordinary adjacent/shared table borders. Coordinates are points
    relative to the rendered image; no OCR or document semantics are consulted.
    """
    contours, hierarchy = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {"status": "ok", "connected_box_groups": [], "large_curved_shapes": []}
    if len(contours) > MAX_EVIDENCE_COMPONENTS:
        return {"status": "partial", "reason": "context_contour_limit"}
    records, frames, heights = [], [], []
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = abs(float(cv2.contourArea(contour)))
        fill = contour_area / max(1, w * h)
        depth, ancestor = 0, index
        while hierarchy[0, ancestor, 3] >= 0:
            depth += 1
            ancestor = int(hierarchy[0, ancestor, 3])
            if depth > 64:
                return {"status": "partial", "reason": "context_nesting_limit"}
        box = [x / sx, y / sy, (x + w) / sx, (y + h) / sy]
        records.append((index, ancestor, depth, box, fill))
        if depth % 2 == 0 and .25 <= w / sx <= 18 and .5 <= h / sy <= 20 and \
                .15 <= contour_area / (sx * sy) <= 150:
            heights.append(h / sy)
        if depth % 2 == 1 and w / sx >= 25 and h / sy >= 10 and \
                contour_area / (sx * sy) >= 250 and fill > .88:
            frames.append({"bbox": box, "root": ancestor})
    if len(frames) > MAX_CONTEXT_FRAMES:
        return {"status": "partial", "reason": "context_frame_limit"}
    reference_height = float(np.median(heights)) if heights else 0
    root_counts = {}
    for frame in frames:
        root_counts[frame["root"]] = root_counts.get(frame["root"], 0) + 1
    curved = []
    for index, root, depth, box, fill in records:
        w, h = box[2] - box[0], box[3] - box[1]
        if depth % 2 == 0 and .3 < fill < .88 and .45 < w / h < 2.2 and \
                w > max(18, 3 * reference_height) and h > max(18, 5 * reference_height) and \
                not (index == root and root_counts.get(root, 0) >= 2):
            curved.append(box)
    if len(curved) > MAX_CONTEXT_SHAPES:
        return {"status": "partial", "reason": "context_shape_limit"}
    # Nested/double contours of the same shape must not multiply the evidence.
    unique_shapes = []
    for box in sorted(curved, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        duplicate = False
        for old in unique_shapes:
            intersection = max(0, min(box[2], old[2]) - max(box[0], old[0])) * \
                           max(0, min(box[3], old[3]) - max(box[1], old[1]))
            union = (box[2] - box[0]) * (box[3] - box[1]) + \
                    (old[2] - old[0]) * (old[3] - old[1]) - intersection
            if intersection / max(1e-9, union) > .8:
                duplicate = True
                break
        if not duplicate:
            unique_shapes.append(box)
    parents = list(range(len(frames)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(a, b):
        parents[find(a)] = find(b)

    edges, adjacent, remaining = [], [], MAX_CONNECTOR_SAMPLE_PIXELS
    for i, first in enumerate(frames):
        a = first["bbox"]
        for j in range(i + 1, len(frames)):
            b = frames[j]["bbox"]
            if first["root"] == frames[j]["root"]:
                join(i, j)
            xlo, xhi = max(a[0], b[0]), min(a[2], b[2])
            ylo, yhi = max(a[1], b[1]), min(a[3], b[3])
            gap_y = max(a[1], b[1]) - min(a[3], b[3])
            gap_x = max(a[0], b[0]) - min(a[2], b[2])
            if (abs(gap_y) < 3 and xhi - xlo > .5 * min(a[2] - a[0], b[2] - b[0])) or \
                    (abs(gap_x) < 3 and yhi - ylo > .5 * min(a[3] - a[1], b[3] - b[1])):
                adjacent.append((i, j))
            bridge = False
            if 4 <= gap_y <= 80 and xhi > xlo:
                margin = max(2, .1 * min(a[2] - a[0], b[2] - b[0]))
                left = int(max(a[0] + margin, b[0] + margin) * sx)
                right = int(min(a[2] - margin, b[2] - margin) * sx)
                top = int((min(a[3], b[3]) + 2) * sy)
                bottom = int((max(a[1], b[1]) - 2) * sy)
                if right > left and bottom > top:
                    remaining -= (right - left) * (bottom - top)
                    if remaining < 0:
                        return {"status": "partial", "reason": "connector_sample_budget"}
                    strip = foreground[top:bottom, left:right]
                    bridge = bool(np.max(np.count_nonzero(strip, axis=0)) / (bottom - top) > .8)
            if 4 <= gap_x <= 80 and yhi > ylo:
                margin = max(2, .1 * min(a[3] - a[1], b[3] - b[1]))
                top = int(max(a[1] + margin, b[1] + margin) * sy)
                bottom = int(min(a[3] - margin, b[3] - margin) * sy)
                left = int((min(a[2], b[2]) + 2) * sx)
                right = int((max(a[0], b[0]) - 2) * sx)
                if right > left and bottom > top:
                    remaining -= (right - left) * (bottom - top)
                    if remaining < 0:
                        return {"status": "partial", "reason": "connector_sample_budget"}
                    strip = foreground[top:bottom, left:right]
                    bridge |= bool(np.max(np.count_nonzero(strip, axis=1)) / (right - left) > .8)
            if bridge:
                edges.append((i, j))
                join(i, j)
    grouped = {}
    for index in range(len(frames)):
        grouped.setdefault(find(index), []).append(index)
    networks = []
    for members in grouped.values():
        connectors = sum(a in members and b in members for a, b in edges)
        sharing = {i for pair in adjacent if all(j in members for j in pair) for i in pair}
        if len(members) >= 3 and connectors >= 2 and len(sharing) / len(members) <= .25:
            networks.append({"boxes": [frames[i]["bbox"] for i in members], "connector_count": connectors})
    return {"status": "ok", "coordinate_system": "render_relative_points",
            "frame_count": len(frames), "connected_box_groups": networks,
            "large_curved_shapes": unique_shapes, "reference_height_points": round(reference_height, 3),
            "connector_sample_pixels": MAX_CONNECTOR_SAMPLE_PIXELS - remaining}


def _page_context(context, bounds):
    """Translate reusable image-relative evidence to exported PDF points."""
    def moved(box):
        return [box[0] + bounds[0], box[1] + bounds[1], box[2] + bounds[0], box[3] + bounds[1]]
    return {**context, "coordinate_system": "pdf_page_points",
        "connected_box_groups": [{**group, "boxes": [moved(box) for box in group["boxes"]]}
                                 for group in context.get("connected_box_groups", [])],
        "large_curved_shapes": [moved(box) for box in context.get("large_curved_shapes", [])]}


def _prepare_pixel_evidence(pixels, extractor, sx, sy, np, cv2):
    """Threshold once and erase detected long rulings; never recognize text."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    foreground = (gray < 180).astype(np.uint8) * 255
    nonrule = foreground.copy()
    processed = 0
    # Erase only thin detected axis-aligned strokes of at least 12 PDF points.
    # The thickness cap avoids treating a dark graphic band as a wide ruling.
    for orientation, lines in (("h", getattr(extractor, "horizontal_lines", [])),
                               ("v", getattr(extractor, "vertical_lines", []))):
        for line in lines[:MAX_RULE_HINTS - processed]:
            processed += 1
            coords = [float(getattr(line, name)) for name in ("x1", "y1", "x2", "y2")]
            thickness = float(getattr(line, "thickness", 1))
            if not all(math.isfinite(value) for value in coords + [thickness]):
                continue
            x1, y1, x2, y2 = coords
            length = abs(x2 - x1) / sx if orientation == "h" else abs(y2 - y1) / sy
            if length < 12:
                continue
            normal_scale = sy if orientation == "h" else sx
            stroke = max(1, round(min(max(1, thickness) + 2, 4 * normal_scale)))
            cv2.line(nonrule, (round(x1), round(y1)), (round(x2), round(y2)),
                     color=0, thickness=stroke)
    try:
        context = _diagram_context(foreground, sx, sy, np, cv2)
    except Exception as exc:
        context = {"status": "unavailable", "reason": type(exc).__name__}
    return {"foreground": foreground, "nonrule": nonrule, "gray": gray, "figure_context": context,
            "remaining_budget": MAX_EVIDENCE_TOTAL_PIXELS}


def _pixel_evidence(group, masks, sx, sy, np, cv2):
    """Advisory component ratios with physical cutoffs and bounded work.

    Connected components can be glyphs, touching words, or graphics; these
    measurements are independent evidence, not a table or text classification.
    """
    foreground, nonrule = masks["foreground"], masks["nonrule"]
    height, width = foreground.shape
    raw = (math.floor(min(box[0] for box in group)), math.floor(min(box[1] for box in group)),
           math.ceil(max(box[2] for box in group)) + 1, math.ceil(max(box[3] for box in group)) + 1)
    x0, y0, x1, y1 = (max(0, raw[0]), max(0, raw[1]), min(width, raw[2]), min(height, raw[3]))
    if x0 >= x1 or y0 >= y1:
        return {"status": "unavailable", "reason": "empty_candidate_crop"}
    crop_width, crop_height = x1 - x0, y1 - y0
    ratio = min(1.0, math.sqrt(MAX_EVIDENCE_CROP_PIXELS / (crop_width * crop_height)))
    sample_size = (max(1, math.floor(crop_width * ratio)),
                   max(1, math.floor(crop_height * ratio)))
    sample_pixels = sample_size[0] * sample_size[1]
    if sample_pixels > masks["remaining_budget"]:
        return {"status": "unavailable", "reason": "page_evidence_pixel_budget"}
    masks["remaining_budget"] -= sample_pixels
    fg, ink = foreground[y0:y1, x0:x1], nonrule[y0:y1, x0:x1]
    gray = masks["gray"][y0:y1, x0:x1]
    if sample_size != (crop_width, crop_height):
        fg = cv2.resize(fg, sample_size, interpolation=cv2.INTER_NEAREST)
        ink = cv2.resize(ink, sample_size, interpolation=cv2.INTER_NEAREST)
        gray = cv2.resize(gray, sample_size, interpolation=cv2.INTER_AREA)
    scale_x, scale_y = sx * sample_size[0] / crop_width, sy * sample_size[1] / crop_height
    foreground_pixels, ink_pixels = int(np.count_nonzero(fg)), int(np.count_nonzero(ink))
    result = {"status": "ok", "dark_threshold": 180,
              "crop_dimensions": [crop_width, crop_height],
              "sample_dimensions": list(sample_size), "crop_clipped": raw != (x0, y0, x1, y1),
              "scale_x": round(scale_x, 6), "scale_y": round(scale_y, 6),
              "foreground_pixels": foreground_pixels, "nonrule_ink_pixels": ink_pixels,
              "foreground_fraction": round(foreground_pixels / sample_pixels, 6),
              "nonrule_foreground_fraction": round(ink_pixels / sample_pixels, 6),
              "removed_rule_ink_fraction": round((foreground_pixels - ink_pixels) /
                                                 max(1, foreground_pixels), 6)}
    result["diagram_context"] = context_evidence([raw[0] / sx, raw[1] / sy, raw[2] / sx, raw[3] / sy],
                                                masks.get("figure_context", {}))
    midtones = gray[(gray > 30) & (gray < 230)]
    result["midtone_fraction"] = round(float(midtones.size) / sample_pixels, 6)
    result["midtone_peak_fraction"] = round(float(np.bincount(midtones, minlength=256).max()) /
                                             max(1, midtones.size), 6)
    if min(scale_x, scale_y) < 0.75:
        result.update(status="partial", reason="insufficient_scale_for_components")
        return result
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        ink, connectivity=8, ltype=cv2.CV_32S)
    result["component_count"] = int(count - 1)
    if count - 1 > MAX_EVIDENCE_COMPONENTS:
        result.update(status="partial", reason="component_limit")
        return result
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype("float64") / scale_x
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype("float64") / scale_y
    areas_pixels = stats[1:, cv2.CC_STAT_AREA].astype("float64")
    areas = areas_pixels / (scale_x * scale_y)
    # Use the same connected-component pass for a local size comparison. This
    # catches diagram primitives which fit the broad fixed-size glyph limits.
    reference = (areas_pixels >= 3) & (heights >= .5) & (widths >= .25)
    reference_height = float(np.median(heights[reference])) if np.any(reference) else 0
    height_limit = max(FIGURE_EVIDENCE_POLICY["oversized_min_height_points"],
                       FIGURE_EVIDENCE_POLICY["oversized_height_ratio"] * reference_height)
    oversized = reference & (heights > height_limit)
    centers = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT] / 2
    spread = float(np.ptp(centers[oversized])) / sample_size[1] if np.any(oversized) else 0
    result.update(reference_component_count=int(np.count_nonzero(reference)),
        reference_component_height_points=round(reference_height, 3),
        oversized_component_height_limit_points=round(height_limit, 3),
        oversized_component_count=int(np.count_nonzero(oversized)),
        oversized_component_ink_fraction=round(float(areas_pixels[oversized].sum()) / max(1, ink_pixels), 6),
        oversized_component_vertical_spread=round(spread, 6))
    longest = np.maximum(widths, heights)
    aspect = longest / np.maximum(np.minimum(widths, heights), 0.25)
    large = (longest > 32) | (areas > 200) | ((aspect >= 12) & (longest >= 12))
    character = ((widths >= 0.25) & (widths <= 18) & (heights >= 0.5) & (heights <= 20)
                 & (areas >= 0.15) & (areas <= 150) & (areas_pixels >= 3)
                 & (aspect < 12) & ~large)
    large_ink, character_ink = float(areas_pixels[large].sum()), float(areas_pixels[character].sum())
    result.update(
        large_component_count=int(np.count_nonzero(large)),
        character_component_count=int(np.count_nonzero(character)),
        tiny_component_count=int(np.count_nonzero(areas_pixels < 3)),
        large_component_ink_fraction=round(large_ink / max(1, ink_pixels), 6),
        character_component_ink_fraction=round(character_ink / max(1, ink_pixels), 6),
        other_component_ink_fraction=round((ink_pixels - large_ink - character_ink) /
                                           max(1, ink_pixels), 6),
        largest_component_ink_fraction=round(float(areas_pixels.max()) / max(1, ink_pixels), 6)
            if len(areas_pixels) else 0.0,
        largest_component_area_points2=round(float(areas.max()), 3) if len(areas) else 0.0,
        max_component_width_points=round(float(widths.max()), 3) if len(widths) else 0.0,
        max_component_height_points=round(float(heights.max()), 3) if len(heights) else 0.0)
    return result


def _layout_components(area, masks, bounds, sx, sy, cv2):
    """Read glyph-shaped components in a bounded crop, without recognizing them."""
    ink = masks["nonrule"]
    height, width = ink.shape
    x0 = max(0, math.floor((area[0] - bounds[0]) * sx))
    y0 = max(0, math.floor((area[1] - bounds[1]) * sy))
    x1 = min(width, math.ceil((area[2] - bounds[0]) * sx) + 1)
    y1 = min(height, math.ceil((area[3] - bounds[1]) * sy) + 1)
    pixels = max(0, x1 - x0) * max(0, y1 - y0)
    if not pixels or pixels > MAX_EVIDENCE_CROP_PIXELS or pixels > masks["layout_budget"]:
        return []
    masks["layout_budget"] -= pixels
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        ink[y0:y1, x0:x1], connectivity=8, ltype=cv2.CV_32S)
    if count > 20_001:
        return []
    components = []
    for x, y, w, h, area_pixels in stats[1:]:
        if 1.5 <= h / sy <= 18 and .3 <= w / sx <= 70 and area_pixels / (sx * sy) >= .5:
            components.append((float(x + x0) / sx + bounds[0], float(y + y0) / sy + bounds[1],
                float(x + x0 + w) / sx + bounds[0], float(y + y0 + h) / sy + bounds[1]))
    return components


def page_grids(region, *, dpi=144, tolerance=2.0):
    """Return ``(cell_groups, transform, diagnostics)`` in pdfplumber page points.

    Each group is a list of ``(x0, top, x1, bottom)`` rectangles. Rendering already
    applies the PDF's page rotation; the low-level extractor adds no rotation.
    The transform therefore has ``deskew_degrees=0`` and is compatible with the
    raster adapter. ``tolerance`` is recorded for interface compatibility; this
    pinned extractor exposes no line-snapping tolerance. The caller applies its
    own boundary clustering and candidate validation.
    """
    started = time.perf_counter()
    diagnostics = {"engine": "img2table", "version": None,
                   "supported_version": SUPPORTED_VERSION,
                   "status": "unavailable", "warnings": [],
                   "runtime_flags": {**EXTRACT_FLAGS, "ocr": False,
                                     "content_extraction": False,
                                     "detect_rotation": False, "cpu_only": True,
                                     "numba_used": False},
                   "requested_dpi": dpi, "tolerance_requested_points": tolerance,
                   "extractor_tolerance_configurable": False,
                   "candidate_tables": 0, "returned_tables": 0,
                   "cell_positions": 0, "unique_cells": 0,
                   "horizontal_rules": [], "horizontal_rules_truncated": False,
                   "returned_group_metadata": [],
                   "figure_evidence_policy": dict(FIGURE_EVIDENCE_POLICY),
                   "figure_context_policy": {
                       "max_contours": MAX_EVIDENCE_COMPONENTS,
                       "max_frames": MAX_CONTEXT_FRAMES, "max_shapes": MAX_CONTEXT_SHAPES,
                       "max_connector_sample_pixels": MAX_CONNECTOR_SAMPLE_PIXELS,
                       "frame_min_width_points": 25, "frame_min_height_points": 10,
                       "frame_min_area_points2": 250, "frame_fill_fraction_minimum": .88,
                       "connector_gap_points": [4, 80], "connector_ink_support_minimum": .8,
                       "shared_border_frame_fraction_maximum": .25,
                       "curved_shape_min_width_points": 18, "curved_shape_min_height_points": 18,
                       "curved_width_to_reference_height_minimum": 3,
                       "curved_height_to_reference_height_minimum": 5},
                   "pixel_evidence_policy": {
                       "dark_threshold": 180, "min_rule_length_points": 12,
                       "max_erased_rule_thickness_points": 4,
                       "character_width_points": [0.25, 18],
                       "character_height_points": [0.5, 20],
                       "character_area_points2": [0.15, 150],
                       "character_min_ink_pixels": 3, "character_max_aspect_exclusive": 12,
                       "large_dimension_exclusive_points": 32,
                       "large_area_exclusive_points2": 200,
                       "elongated_min_aspect": 12, "elongated_min_length_points": 12,
                       "min_component_pixels_per_point": 0.75,
                       "max_sample_pixels_per_crop": MAX_EVIDENCE_CROP_PIXELS,
                       "max_sample_pixels_per_page": MAX_EVIDENCE_TOTAL_PIXELS,
                       "max_components_per_crop": MAX_EVIDENCE_COMPONENTS}}

    def result(groups=None, transform=None):
        diagnostics["seconds"] = round(time.perf_counter() - started, 4)
        return groups or [], transform, diagnostics

    try:
        diagnostics["version"] = metadata.version("img2table")
    except metadata.PackageNotFoundError:
        diagnostics["warnings"].append("img2table_optional_dependency_missing")
        diagnostics["action"] = INSTALL_ACTION
        return result()
    except Exception as exc:
        diagnostics["warnings"].append(f"img2table_metadata_failed:{type(exc).__name__}")
        diagnostics["action"] = INSTALL_ACTION
        return result()
    if diagnostics["version"] != SUPPORTED_VERSION:
        diagnostics["warnings"].append("img2table_unsupported_version")
        diagnostics["action"] = INSTALL_ACTION
        return result()
    try:
        np, extractor_class, cv2 = _load_backend()
        diagnostics["runtime_flags"]["opencv_threads"] = cv2.getNumThreads()
        diagnostics["runtime_flags"]["opencv_thread_setting"] = 0
        diagnostics["runtime_flags"]["opencl"] = cv2.ocl.useOpenCL()
        diagnostics["runtime_flags"]["opencv_version"] = cv2.__version__
    except Exception as exc:
        # Library errors may contain PDF content, paths, or environment values.
        diagnostics["warnings"].append(f"img2table_backend_import_failed:{type(exc).__name__}")
        diagnostics["action"] = INSTALL_ACTION
        return result()
    if not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
        diagnostics["status"] = "failed"
        diagnostics["warnings"].append("img2table_invalid_dpi")
        return result()

    try:
        image, bounds, effective_dpi, original_size = _render(region, dpi)
        sx = image.width / (bounds[2] - bounds[0])
        sy = image.height / (bounds[3] - bounds[1])
        transform = {"deskew_degrees": 0.0, "render_bbox": list(bounds),
                     "image_size": list(image.size), "scale_x": sx, "scale_y": sy,
                     "page_rotation": getattr(region, "rotation", 0),
                     "extractor_rotation_degrees": 0.0}
        diagnostics.update(effective_dpi=round(effective_dpi, 3),
                           render_image_size=original_size, image_size=list(image.size),
                           input_dimensions=list(image.size), page_region=list(bounds),
                           page_rotation=getattr(region, "rotation", 0))
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["warnings"].append(f"img2table_render_failed:{type(exc).__name__}")
        return result()

    try:
        pixels = np.asarray(image, dtype=np.uint8)
        extractor = extractor_class(img=pixels)
        diagnostics["image_metrics_available"] = extractor.characteristics is not None
        tables = extractor.extract_tables(**EXTRACT_FLAGS)
        diagnostics["candidate_tables"] = len(tables)
        groups, positions, invalid, source_indices = _raw_cells(tables)
        diagnostics.update(cell_positions=positions, returned_tables=len(groups),
                           unique_cells=sum(len(group) for group in groups),
                           invalid_geometry_candidates=invalid)
        if invalid:
            diagnostics["warnings"].append("img2table_invalid_geometry_candidates_skipped")
        if not diagnostics["image_metrics_available"]:
            # Character dimensions come from connected components, without OCR.
            # Pure empty rule grids may lack these metrics and need raster fallback.
            diagnostics["warnings"].append("img2table_image_metrics_unavailable")
        diagnostics["status"] = "ok" if groups else "no_candidates"
        grids = [[(box[0] / sx + bounds[0], box[1] / sy + bounds[1],
                   box[2] / sx + bounds[0], box[3] / sy + bounds[1])
                  for box in group] for group in groups]
        try:
            masks = _prepare_pixel_evidence(pixels, extractor, sx, sy, np, cv2) if groups else None
            masks_error = None
        except Exception as exc:
            masks, masks_error = None, type(exc).__name__
        if masks is not None:
            diagnostics["figure_context"] = _page_context(masks.get("figure_context", {}), bounds)
        repair_notes = [{"status": "unchanged", "source_group_indices": [i]}
                        for i in range(len(groups))]
        try:
            rules, truncated = _horizontal_rules(extractor, bounds, sx, sy)
            diagnostics["horizontal_rules"] = rules
            diagnostics["horizontal_rules_truncated"] = truncated
            if truncated:
                diagnostics["warnings"].append("img2table_horizontal_rule_hints_truncated")
            if masks is not None:
                from layout_repair import repair_groups
                vertical = [(float(v.x1) / sx + bounds[0], float(v.y1) / sy + bounds[1],
                             float(v.x2) / sx + bounds[0], float(v.y2) / sy + bounds[1])
                            for v in getattr(extractor, "vertical_lines", [])[:MAX_RULE_HINTS]]
                masks["layout_budget"] = MAX_EVIDENCE_TOTAL_PIXELS
                repaired, notes = repair_groups(grids, rules, vertical,
                    lambda area: _layout_components(area, masks, bounds, sx, sy, cv2))
                repaired_pixels = [[((b[0] - bounds[0]) * sx, (b[1] - bounds[1]) * sy,
                            (b[2] - bounds[0]) * sx, (b[3] - bounds[1]) * sy) for b in grid]
                          for grid in repaired]
                grids, groups, repair_notes = repaired, repaired_pixels, notes
                diagnostics["layout_repair"] = {"status": "completed",
                    "repaired_tables": sum(n["status"] == "repaired" for n in notes),
                    "component_crop_pixel_budget": MAX_EVIDENCE_TOTAL_PIXELS,
                    "component_crop_pixels_used": MAX_EVIDENCE_TOTAL_PIXELS - masks["layout_budget"]}
        except Exception as exc:
            diagnostics["warnings"].append(f"img2table_layout_repair_failed:{type(exc).__name__}")
        diagnostics.update(returned_tables=len(grids), unique_cells=sum(len(grid) for grid in grids))
        for group, grid, repair in zip(groups, grids, repair_notes):
            source_index = source_indices[repair["source_group_indices"][0]]
            # img2table 2.0.0 has no borderless field. Preserve a bool if a
            # compatible Table supplies one; None means the source is unknown.
            borderless = getattr(tables[source_index], "borderless", None)
            item_metadata = {
                "source_candidate_index": source_index,
                "bbox": [min(box[0] for box in grid), min(box[1] for box in grid),
                         max(box[2] for box in grid), max(box[3] for box in grid)],
                "unique_cells": len(grid),
                "borderless": borderless if isinstance(borderless, bool) else None}
            if repair["status"] == "repaired":
                item_metadata["layout_repair"] = {**repair,
                    "source_candidate_indices": [source_indices[i] for i in repair["source_group_indices"]]}
            try:
                item_metadata["pixel_evidence"] = _pixel_evidence(group, masks, sx, sy, np, cv2) \
                    if masks is not None else {"status": "unavailable", "reason": masks_error}
            except Exception as exc:
                item_metadata["pixel_evidence"] = {"status": "unavailable",
                                                   "reason": type(exc).__name__}
            diagnostics["returned_group_metadata"].append(item_metadata)
        return result(grids, transform)
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["warnings"].append(f"img2table_extraction_failed:{type(exc).__name__}")
        return result(transform=transform)
