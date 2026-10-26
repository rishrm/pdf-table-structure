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
}


def figure_evidence_reasons(pixel):
    """Reject strong graphic evidence; uncertain/missing evidence is advisory.

    Small circles and diagram strokes can fit fixed glyph-size limits. Compare
    them with the local text scale, and require most ink to belong to oversized
    shapes plus either repeated vertical spread or diffuse photographic tones.
    A large heading alone, flat table shading, and embedded images by themselves
    do not meet this condition. Graphics inside genuine cells remain ambiguous.
    """
    p = FIGURE_EVIDENCE_POLICY
    if pixel.get("status") != "ok" or \
            pixel.get("reference_component_count", 0) < p["reference_component_minimum"] or \
            pixel.get("oversized_component_ink_fraction", 0) <= p["oversized_ink_fraction_minimum"]:
        return []
    repeated_shapes = (
        pixel.get("oversized_component_count", 0) >= p["repeated_oversized_component_minimum"] and
        pixel.get("oversized_component_vertical_spread", 0) > p["oversized_vertical_spread_minimum"])
    photographic_tones = (
        pixel.get("midtone_fraction", 0) > p["diffuse_midtone_fraction_minimum"] and
        pixel.get("midtone_peak_fraction", 1) < p["diffuse_midtone_peak_fraction_maximum"])
    return ["graphic_components_dominate_local_text_scale"] if repeated_shapes or photographic_tones else []
