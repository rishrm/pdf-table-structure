"""OCR-free img2table behavior on pixels, spans, crops, and backend failures."""
from contextlib import ExitStack
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import builtins
import json
import tempfile
import unittest
from unittest.mock import patch

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

import img2table_structure as adapter
from make_demo import make_demo
from table_structure import check_grid, structure_from_cells


def pixel_grid():
    image = Image.new("RGB", (660, 440), "white")
    draw = ImageDraw.Draw(image)
    for x in (80, 220, 500):
        draw.line((x, 80, x, 320), fill="black", width=2)
    draw.line((360, 140, 360, 320), fill="black", width=2)
    for y in (80, 140, 260, 320):
        draw.line((80, y, 500, y), fill="black", width=2)
    draw.line((220, 200, 500, 200), fill="black", width=2)
    for x, y, label in [(90, 100, "Header"), (230, 100, "Spanning header"),
                         (90, 180, "Spanning rows"), (230, 165, "A"),
                         (370, 165, "B"), (230, 225, "C"), (370, 225, "D"),
                         (90, 285, "Last"), (230, 285, "E")]:
        draw.text((x, y), label, fill="black")
    return image


class ImageRegion:
    def __init__(self, image, bounds=None, rotation=0, root=None):
        self.image = image
        self.bbox = bounds or (0, 0, image.width, image.height)
        self.width = self.bbox[2] - self.bbox[0]
        self.height = self.bbox[3] - self.bbox[1]
        self.rotation = rotation
        self.render_kwargs = None
        if root is not None:
            self.root_page = root

    def to_image(self, **kwargs):
        self.render_kwargs = kwargs
        return SimpleNamespace(original=self.image, bbox=self.bbox)


class GeometryCell:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def content(self):
        raise AssertionError("Cell content must never be read")


def fake_backend(tables):
    class Extractor:
        characteristics = object()
        calls = []

        def __init__(self, img):
            self.image = img

        def extract_tables(self, **kwargs):
            self.calls.append(kwargs)
            return tables

    numpy = SimpleNamespace(asarray=lambda image, dtype: image, uint8=object())
    cv2 = SimpleNamespace(getNumThreads=lambda: 1, __version__="test",
                          ocl=SimpleNamespace(useOpenCL=lambda: False))
    return (numpy, Extractor, cv2), Extractor


def backend_available():
    try:
        if metadata.version("img2table") != adapter.SUPPORTED_VERSION:
            return False
        adapter._load_backend()
        return True
    except Exception:
        return False


class AdapterContractTests(unittest.TestCase):
    def test_duplicate_grid_positions_preserve_one_merged_rectangle_and_empty_cells(self):
        header = GeometryCell(10, 10, 90, 30)
        left, right = GeometryCell(10, 30, 50, 50), GeometryCell(50, 30, 90, 50)
        tables = [SimpleNamespace(rows=[SimpleNamespace(cells=[header, header]),
                                       SimpleNamespace(cells=[left, right])])]
        backend, extractor = fake_backend(tables)
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend):
            groups, transform, diagnostics = adapter.page_grids(
                ImageRegion(Image.new("RGB", (100, 60), "white")))
        self.assertEqual(extractor.calls, [dict(implicit_rows=True, implicit_columns=True,
                                               borderless_tables=True)])
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(diagnostics["cell_positions"], 4)
        self.assertEqual(diagnostics["unique_cells"], 3)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual(check_grid(table), [])
        self.assertEqual(table["cells"][0]["column_span"], 2)
        self.assertEqual(transform["deskew_degrees"], 0)

    def test_crop_offset_and_independent_xy_scale_use_rendered_bbox(self):
        cell = GeometryCell(20, 20, 80, 80)
        backend, _ = fake_backend([SimpleNamespace(rows=[SimpleNamespace(cells=[cell])])])
        region = ImageRegion(Image.new("RGB", (200, 100), "white"),
                             bounds=(100, 200, 200, 400), rotation=270)
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend):
            groups, transform, diagnostics = adapter.page_grids(region)
        self.assertEqual(groups, [[(110.0, 240.0, 140.0, 360.0)]])
        self.assertEqual(transform["scale_x"], 2)
        self.assertEqual(transform["scale_y"], 0.5)
        self.assertEqual(transform["page_rotation"], 270)
        self.assertEqual(transform["extractor_rotation_degrees"], 0)

    def test_rule_hints_map_to_pdf_points_and_metadata_aligns_after_skipped_candidate(self):
        malformed = SimpleNamespace(rows=[SimpleNamespace(cells=[GeometryCell(0, 0, 0, 1)])])
        table = SimpleNamespace(rows=[SimpleNamespace(cells=[GeometryCell(20, 20, 80, 80)])],
                                borderless=True)
        backend, extractor = fake_backend([malformed, table])
        # Rule coordinates are endpoints; horizontal rules have zero height.
        extractor.horizontal_lines = [GeometryCell(20, 20, 80, 20)] * 2
        region = ImageRegion(Image.new("RGB", (200, 100), "white"),
                             bounds=(100, 200, 200, 400))
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend):
            groups, _, diagnostics = adapter.page_grids(region)
        self.assertEqual(len(groups), 1)
        self.assertEqual(diagnostics["horizontal_rules"], [[110.0, 240.0, 140.0, 240.0]])
        group_metadata = dict(diagnostics["returned_group_metadata"][0])
        self.assertEqual(group_metadata.pop("pixel_evidence")["status"], "unavailable")
        self.assertEqual(group_metadata, {
            "source_candidate_index": 1, "bbox": [110.0, 240.0, 140.0, 360.0],
            "unique_cells": 1, "borderless": True})
        json.dumps(diagnostics, allow_nan=False)

    def test_tiny_crop_caps_the_underlying_full_page_render(self):
        backend, _ = fake_backend([])
        root = SimpleNamespace(cropbox=(0, 0, 10_000, 20_000))
        region = ImageRegion(Image.new("RGB", (100, 50), "white"), root=root)
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend):
            _, _, diagnostics = adapter.page_grids(region)
        self.assertLessEqual(region.render_kwargs["resolution"] * 20_000 / 72,
                             adapter.MAX_RENDER_DIMENSION)
        self.assertAlmostEqual(diagnostics["effective_dpi"], 7.92)

    def test_oversized_renderer_output_is_resized_and_mapping_stays_correct(self):
        backend, _ = fake_backend([SimpleNamespace(rows=[SimpleNamespace(
            cells=[GeometryCell(0, 0, 1100, 1100)])])])
        region = ImageRegion(Image.new("RGB", (2400, 2400), "white"),
                             bounds=(10, 20, 1210, 1220))
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend):
            groups, _, diagnostics = adapter.page_grids(region)
        self.assertEqual(diagnostics["input_dimensions"], [2200, 2200])
        self.assertEqual(groups, [[(10.0, 20.0, 610.0, 620.0)]])

    def test_missing_dependency_is_actionable_and_does_not_render(self):
        region = ImageRegion(Image.new("RGB", (40, 40), "white"))
        with patch.object(adapter.metadata, "version", side_effect=metadata.PackageNotFoundError):
            groups, transform, diagnostics = adapter.page_grids(region)
        self.assertEqual(groups, [])
        self.assertIsNone(transform)
        self.assertIsNone(region.render_kwargs)
        self.assertEqual(diagnostics["status"], "unavailable")
        self.assertIn("img2table_optional_dependency_missing", diagnostics["warnings"])
        self.assertIn("img2table==2.0.0", diagnostics["action"])

    def test_unsupported_version_and_broken_native_dependency_are_sanitized(self):
        region = ImageRegion(Image.new("RGB", (40, 40), "white"))
        with patch.object(adapter.metadata, "version", return_value="9.9.9"):
            _, _, diagnostics = adapter.page_grids(region)
        self.assertEqual(diagnostics["warnings"], ["img2table_unsupported_version"])
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", side_effect=ImportError("secret content")):
            _, _, diagnostics = adapter.page_grids(region)
        self.assertEqual(diagnostics["status"], "unavailable")
        self.assertNotIn("secret", str(diagnostics))
        self.assertIn("img2table_backend_import_failed:ImportError", diagnostics["warnings"])

    def test_extractor_failure_returns_no_geometry_without_leaking_exception_text(self):
        backend, extractor = fake_backend([])
        region = ImageRegion(Image.new("RGB", (100, 60), "white"))
        with patch.object(adapter.metadata, "version", return_value="2.0.0"), \
                patch.object(adapter, "_load_backend", return_value=backend), \
                patch.object(extractor, "extract_tables", side_effect=RuntimeError("secret document")):
            groups, transform, diagnostics = adapter.page_grids(region)
        self.assertEqual(groups, [])
        self.assertIsNotNone(transform)
        self.assertEqual(diagnostics["status"], "failed")
        self.assertEqual(diagnostics["warnings"], ["img2table_extraction_failed:RuntimeError"])
        self.assertNotIn("secret", str(diagnostics))


@unittest.skipUnless(backend_available(), "optional img2table==2.0.0 backend unavailable")
class RealPixelTests(unittest.TestCase):
    def test_actual_pixels_recover_spans_partial_rules_and_empty_cell(self):
        for gap in (False, True):
            with self.subTest(small_rule_gap=gap):
                image = pixel_grid()
                if gap:
                    ImageDraw.Draw(image).rectangle((300, 139, 301, 141), fill="white")
                groups, _, diagnostics = adapter.page_grids(ImageRegion(image))
                self.assertEqual(diagnostics["status"], "ok")
                self.assertEqual(len(groups), 1)
                table = structure_from_cells(groups[0], "img2table")
                self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
                self.assertEqual(len(table["cells"]), 10)
                self.assertTrue(any(cell["row_span"] == 2 for cell in table["cells"]))
                self.assertTrue(any(cell["column_span"] == 2 for cell in table["cells"]))
                self.assertTrue(any(cell["row"] == 3 and cell["column"] == 2
                                    for cell in table["cells"]))
                self.assertEqual(check_grid(table), [])
                self.assertEqual(diagnostics["runtime_flags"]["opencv_threads"], 1)
                self.assertFalse(diagnostics["runtime_flags"]["opencl"])
                self.assertTrue(diagnostics["horizontal_rules"])
                self.assertIsNone(diagnostics["returned_group_metadata"][0]["borderless"])

    def test_actual_borderless_pixels_use_component_geometry(self):
        image = Image.new("RGB", (660, 440), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=16)
        values = [("Item", "Count", "Amount"), ("Apples", "12", "24.00"),
                  ("Pears", "8", "16.00"), ("Oranges", "10", "30.00"),
                  ("Peaches", "18", "54.00")]
        for row, labels in enumerate(values):
            for x, label in zip((80, 280, 440), labels):
                draw.text((x, 80 + row * 50), label, font=font, fill="black")
        groups, _, diagnostics = adapter.page_grids(ImageRegion(image))
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertEqual(len(table["cells"]), 15)
        self.assertEqual(check_grid(table), [])
        self.assertTrue(diagnostics["runtime_flags"]["borderless_tables"])

    def test_ocr_document_and_content_pipeline_are_never_entered(self):
        from img2table.tables.types import Table
        original_import = builtins.__import__

        def import_guard(name, *args, **kwargs):
            if name.startswith(("img2table.ocr", "img2table.document")):
                raise AssertionError(f"Forbidden pipeline import: {name}")
            return original_import(name, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch("builtins.__import__", side_effect=import_guard))
            stack.enter_context(patch.object(Table, "get_content",
                                            side_effect=AssertionError("OCR/content extraction entered")))
            stack.enter_context(patch.object(Table, "extracted_table",
                                            side_effect=AssertionError("Content serialization entered")))
            groups, _, diagnostics = adapter.page_grids(ImageRegion(pixel_grid()))
        self.assertEqual(len(groups), 1)
        self.assertFalse(diagnostics["runtime_flags"]["ocr"])
        self.assertFalse(diagnostics["runtime_flags"]["content_extraction"])

    def test_actual_pdf_crop_and_page_rotation_map_to_pdfplumber_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_demo(Path(folder) / "fixture.pdf")
            with pdfplumber.open(path) as document:
                region = document.pages[3].crop((50, 130, 490, 310))
                groups, transform, diagnostics = adapter.page_grids(region)
                self.assertEqual(len(groups), 1)
                box = structure_from_cells(groups[0], "img2table")["bbox"]
                for actual, expected in zip(box, (60, 140, 480, 300)):
                    self.assertAlmostEqual(actual, expected, delta=2)
                self.assertEqual(transform["render_bbox"], [50, 130, 490, 310])
                rotated = document.pages[4]
                groups, transform, _ = adapter.page_grids(rotated)
                expected = rotated.find_tables()[0].bbox
                actual = structure_from_cells(groups[0], "img2table")["bbox"]
                for value, target in zip(actual, expected):
                    self.assertAlmostEqual(value, target, delta=2)
                self.assertEqual(transform["page_rotation"], 90)
                self.assertEqual(transform["deskew_degrees"], 0)

    def test_empty_page_reports_metric_limitation_for_fallback(self):
        groups, _, diagnostics = adapter.page_grids(
            ImageRegion(Image.new("RGB", (400, 300), "white")))
        self.assertEqual(groups, [])
        self.assertEqual(diagnostics["status"], "no_candidates")
        self.assertIn("img2table_image_metrics_unavailable", diagnostics["warnings"])

    def test_failed_optional_pixel_evidence_keeps_geometry_and_sanitizes_error(self):
        with patch.object(adapter, "_prepare_pixel_evidence",
                          side_effect=RuntimeError("private document data")):
            groups, _, diagnostics = adapter.page_grids(ImageRegion(pixel_grid()))
        self.assertEqual(len(groups), 1)
        self.assertEqual(diagnostics["returned_group_metadata"][0]["pixel_evidence"],
                         {"status": "unavailable", "reason": "RuntimeError"})
        self.assertNotIn("private document", str(diagnostics))


@unittest.skipUnless(backend_available(), "optional img2table==2.0.0 backend unavailable")
class PixelEvidenceTests(unittest.TestCase):
    def evidence(self, image, scale=1, horizontal=(), vertical=()):
        np, _, cv2 = adapter._load_backend()
        extractor = SimpleNamespace(horizontal_lines=horizontal, vertical_lines=vertical)
        masks = adapter._prepare_pixel_evidence(np.asarray(image), extractor, scale, scale, np, cv2)
        group = [(0, 0, image.width - 1, image.height - 1)]
        return adapter._pixel_evidence(group, masks, scale, scale, np, cv2)

    def test_large_graphics_and_character_sized_ink_have_different_component_ratios(self):
        text = Image.new("RGB", (300, 180), "white")
        draw = ImageDraw.Draw(text)
        for row in range(4):
            draw.text((15, 15 + row * 35), "Alpha 123 Beta 456", font=ImageFont.load_default(size=16),
                      fill="black")
        text_evidence = self.evidence(text)
        graphic = text.copy()
        ImageDraw.Draw(graphic).rectangle((10, 10, 270, 110), fill="black")
        graphic_evidence = self.evidence(graphic)
        self.assertGreater(text_evidence["character_component_ink_fraction"], 0.8)
        self.assertLess(text_evidence["large_component_ink_fraction"], 0.05)
        self.assertGreater(graphic_evidence["large_component_ink_fraction"], 0.9)
        self.assertGreater(graphic_evidence["foreground_fraction"], 0.4)
        json.dumps(graphic_evidence, allow_nan=False)

    def test_detected_horizontal_and_vertical_rulings_are_removed_before_components(self):
        image = Image.new("RGB", (300, 180), "white")
        draw = ImageDraw.Draw(image)
        h = [GeometryCell(20, y, 280, y) for y in (20, 80, 160)]
        v = [GeometryCell(x, 20, x, 160) for x in (20, 150, 280)]
        for line in h + v:
            draw.line((line.x1, line.y1, line.x2, line.y2), fill="black", width=2)
        draw.text((35, 35), "Values 123", font=ImageFont.load_default(size=16), fill="black")
        result = self.evidence(image, horizontal=h, vertical=v)
        self.assertGreater(result["foreground_fraction"], result["nonrule_foreground_fraction"])
        self.assertGreater(result["removed_rule_ink_fraction"], 0.5)
        self.assertLess(result["large_component_ink_fraction"], 0.05)
        self.assertGreater(result["character_component_ink_fraction"], 0.8)

    def test_component_cutoffs_follow_physical_render_scale(self):
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        for x in (10, 25, 40):
            draw.rectangle((x, 10, x + 4, 18), fill="black")
        draw.rectangle((80, 50, 180, 51), fill="black")
        one = self.evidence(image)
        two = self.evidence(image.resize((400, 200), Image.Resampling.NEAREST), scale=2)
        for key in ("character_component_count", "large_component_count",
                    "character_component_ink_fraction", "large_component_ink_fraction",
                    "largest_component_area_points2"):
            self.assertEqual(one[key], two[key], key)

    def test_crop_budget_and_component_limit_produce_partial_or_unavailable_evidence(self):
        np, _, cv2 = adapter._load_backend()
        image = Image.new("RGB", (400, 400), "white")
        group = [(0, 0, 399, 399)]
        extractor = SimpleNamespace(horizontal_lines=[], vertical_lines=[])
        with patch.object(adapter, "MAX_EVIDENCE_CROP_PIXELS", 10_000), \
                patch.object(adapter, "MAX_EVIDENCE_TOTAL_PIXELS", 10_000):
            masks = adapter._prepare_pixel_evidence(np.asarray(image), extractor, 2, 2, np, cv2)
            first = adapter._pixel_evidence(group, masks, 2, 2, np, cv2)
            second = adapter._pixel_evidence(group, masks, 2, 2, np, cv2)
        self.assertEqual(first["sample_dimensions"], [100, 100])
        self.assertEqual(first["status"], "partial")
        self.assertEqual(first["reason"], "insufficient_scale_for_components")
        self.assertEqual(second, {"status": "unavailable", "reason": "page_evidence_pixel_budget"})
        draw = ImageDraw.Draw(image)
        for x in range(10, 390, 10):
            for y in range(10, 390, 10):
                draw.rectangle((x, y, x + 2, y + 2), fill="black")
        with patch.object(adapter, "MAX_EVIDENCE_COMPONENTS", 10):
            result = self.evidence(image)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reason"], "component_limit")
        self.assertNotIn("large_component_ink_fraction", result)


if __name__ == "__main__":
    unittest.main()
