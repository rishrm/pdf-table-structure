"""Conservative figure evidence, using measured geometry rather than labels."""

FIGURE_EVIDENCE_POLICY = {
    "reference_component_minimum": 12,
    "oversized_height_ratio": 2.5,
    "oversized_min_height_points": 6,
    "oversized_ink_fraction_minimum": 0.6,
    "repeated_oversized_component_minimum": 4,
    "oversized_vertical_spread_minimum": 0.35,
    "diffuse_midtone_fraction_minimum": 0.85,
    "diffuse_midtone_peak_fraction_maximum": 0.2,
    "flow_frame_minimum": 3,
    "flow_connector_minimum": 2,
    "flow_width_coverage_minimum": 0.6,
    "flow_height_coverage_minimum": 0.6,
    "graphic_panel_minimum": 3,
    "graphic_panel_width_coverage_minimum": 0.65,
    "graphic_panel_height_coverage_minimum": 0.5,
}


def context_evidence(bbox, context):
    """Measure a candidate against bounded page geometry, without image imports.

    Context and bbox must use the same point coordinate system. Evidence from
    neighboring connected boxes can identify a fragment of a larger flowchart;
    mere image membership or a table's single graphic column is insufficient.
    """
    if context.get("status") != "ok":
        return {"status": "unavailable", "reason": context.get("reason", "missing_context")}
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return {"status": "unavailable", "reason": "empty_candidate"}

    def local(boxes, fraction):
        return [b for b in boxes if
            max(0, min(x1, b[2]) - max(x0, b[0])) * max(0, min(y1, b[3]) - max(y0, b[1])) >=
            fraction * (b[2] - b[0]) * (b[3] - b[1])]

    def coverage(boxes):
        if not boxes:
            return 0.0, 0.0
        return (round((min(x1, max(b[2] for b in boxes)) - max(x0, min(b[0] for b in boxes))) / (x1 - x0), 4),
                round((min(y1, max(b[3] for b in boxes)) - max(y0, min(b[1] for b in boxes))) / (y1 - y0), 4))

    frames, connector_count = [], 0
    for group in context.get("connected_box_groups", []):
        selected = local(group["boxes"], .2)
        if selected:
            frames.extend(selected)
            connector_count += group["connector_count"]
    panels = local(context.get("large_curved_shapes", []), .6)
    frame_width, frame_height = coverage(frames)
    panel_width, panel_height = coverage(panels)
    return {"status": "ok", "connected_frame_count": len(frames),
        "context_connector_count": connector_count, "connected_frame_width_coverage": frame_width,
        "connected_frame_height_coverage": frame_height, "large_curved_shape_count": len(panels),
        "curved_shape_width_coverage": panel_width, "curved_shape_height_coverage": panel_height}


def figure_evidence_reasons(pixel):
    """Reject strong graphic evidence; uncertain/missing evidence is advisory.

    Small circles and diagram strokes can fit fixed glyph-size limits. Compare
    them with the local text scale, and require most ink to belong to oversized
    shapes plus either repeated vertical spread or diffuse photographic tones.
    A large heading alone, flat table shading, and embedded images by themselves
    do not meet this condition. Graphics inside genuine cells remain ambiguous.
    """
    p = FIGURE_EVIDENCE_POLICY
    reasons = []
    context = pixel.get("diagram_context", {})
    if context.get("status") == "ok":
        if context.get("connected_frame_count", 0) >= p["flow_frame_minimum"] and \
                context.get("context_connector_count", 0) >= p["flow_connector_minimum"] and \
                context.get("connected_frame_width_coverage", 0) >= p["flow_width_coverage_minimum"] and \
                context.get("connected_frame_height_coverage", 0) >= p["flow_height_coverage_minimum"]:
            reasons.append("connected_box_flow_layout")
        if context.get("large_curved_shape_count", 0) >= p["graphic_panel_minimum"] and \
                context.get("curved_shape_width_coverage", 0) >= p["graphic_panel_width_coverage_minimum"] and \
                context.get("curved_shape_height_coverage", 0) >= p["graphic_panel_height_coverage_minimum"]:
            reasons.append("repeated_graphic_panels")
    if pixel.get("status") != "ok" or \
            pixel.get("reference_component_count", 0) < p["reference_component_minimum"] or \
            pixel.get("oversized_component_ink_fraction", 0) <= p["oversized_ink_fraction_minimum"]:
        return reasons
    repeated_shapes = (
        pixel.get("oversized_component_count", 0) >= p["repeated_oversized_component_minimum"] and
        pixel.get("oversized_component_vertical_spread", 0) > p["oversized_vertical_spread_minimum"])
    photographic_tones = (
        pixel.get("midtone_fraction", 0) > p["diffuse_midtone_fraction_minimum"] and
        pixel.get("midtone_peak_fraction", 1) < p["diffuse_midtone_peak_fraction_maximum"])
    if repeated_shapes or photographic_tones:
        reasons.append("graphic_components_dominate_local_text_scale")
    return reasons
