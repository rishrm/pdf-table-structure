"""Behavioral tests on actual synthetic PDFs, including spans and scans."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pdfplumber

from make_demo import make_demo
from table_structure import (apply_merges, check_grid, detect_page,
                             parse_pages, run, structure_from_cells)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = make_demo(Path(cls.folder.name) / "demo.pdf")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_bordered_pdf_spans_empty_cell_and_multiple_tables(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[0], "lines")
        self.assertEqual(len(result["tables"]), 3)
        main = result["tables"][0]
        self.assertEqual((main["row_count"], main["column_count"]), (4, 3))
        self.assertEqual(len(main["cells"]), 10)
        self.assertEqual(check_grid(main), [])
        self.assertTrue(any(c["column_span"] == 2 for c in main["cells"]))
        self.assertTrue(any(c["row_span"] == 2 for c in main["cells"]))
        self.assertTrue(any(c["row"] == 3 and c["column"] == 2 for c in main["cells"]))

    def test_known_region_excludes_neighboring_tables(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[0], "lines", bbox=[50, 130, 470, 310])
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["bbox"], [60, 140, 460, 300])

    def test_borderless_known_region(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[1], "text", bbox=[50, 125, 500, 290])
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(len(table["cells"]), 12)
        self.assertEqual(check_grid(table), [])

    def test_mixed_page_detects_the_image_table(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[2], "auto")
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["method"], "raster")
        self.assertEqual(len(result["tables"][0]["cells"]), 12)

    def test_image_only_page_uses_pixel_detection(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[3], "auto")
        self.assertEqual(result["status"], "candidates_detected")
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual((result["tables"][0]["row_count"],
                          result["tables"][0]["column_count"]), (4, 3))
        self.assertEqual(check_grid(result["tables"][0]), [])

    def test_cropped_scan_coordinates_stay_in_original_page_space(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[3], "raster", bbox=[50, 130, 490, 310])
        self.assertEqual(len(result["tables"]), 1)
        bbox = result["tables"][0]["bbox"]
        for actual, expected in zip(bbox, [60, 140, 480, 300]):
            self.assertAlmostEqual(actual, expected, delta=2)

    def test_rotated_page_geometry(self):
        with pdfplumber.open(self.path) as pdf:
            result = detect_page(pdf.pages[4], "lines")
        self.assertEqual(result["rotation"], 90)
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(len(result["tables"][0]["cells"]), 4)
        self.assertEqual(check_grid(result["tables"][0]), [])
        x0, top, x1, bottom = result["tables"][0]["bbox"]
        self.assertTrue(0 <= x0 < x1 <= result["width"])
        self.assertTrue(0 <= top < bottom <= result["height"])

    def test_uncovered_grid_is_reported_not_filled(self):
        table = structure_from_cells([(0, 0, 20, 20), (20, 0, 40, 20),
                                      (0, 20, 20, 40)], "lines")
        self.assertIn("uncovered_grid_slots", table["warnings"])
        self.assertEqual(len(table["cells"]), 3)

    def test_model_candidate_preserves_baseline(self):
        baseline = structure_from_cells([(0, 0, 20, 20), (20, 0, 40, 20)], "text")
        candidate = apply_merges(baseline, [["c0000", "c0001"]])
        self.assertEqual(len(baseline["cells"]), 2)
        self.assertEqual(candidate["cells"][0]["column_span"], 2)
        self.assertEqual(check_grid(candidate), [])

    def test_page_ranges_cannot_silently_exceed_document(self):
        self.assertEqual(parse_pages("1,3-5,3", 5), [1, 3, 4, 5])
        for value in ("0", "4-2", "1-1000000", "6", "wat"):
            with self.assertRaises(ValueError):
                parse_pages(value, 5)

    def test_offline_run_exports_geometry_without_pdf_text(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(pdf=str(self.path), output=folder, strategy="lines",
                pages="1", bbox=None, tolerance=2.0, ollama_model=None,
                timeout=45, json_mode=False, no_overlays=True, dpi=120)
            with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("offline")):
                path, report = run(args)
            exported = path.read_text()
            self.assertEqual(json.loads(exported)["metrics"]["table_candidates"], 3)
            self.assertNotIn("Merged header", exported)
            self.assertNotIn("Category", exported)
            self.assertFalse(report["cell_text_in_json"])

    def test_bordered_tables_skip_model_even_if_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(pdf=str(self.path), output=folder, strategy="lines",
                pages="1", bbox=None, tolerance=2.0, ollama_model="llama3.1:latest",
                timeout=45, json_mode=False, no_overlays=True, dpi=120)
            with patch("ollama_refine.propose_merges", side_effect=AssertionError("unnecessary call")):
                _, report = run(args)
            self.assertTrue(all(t["model_review"]["status"] == "skipped"
                                for t in report["pages"][0]["tables"]))

    def test_scan_grid_skips_model_even_if_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(pdf=str(self.path), output=folder, strategy="auto",
                pages="4", bbox=None, tolerance=2.0, ollama_model="llama3.1:latest",
                timeout=45, json_mode=False, no_overlays=True, dpi=120)
            with patch("ollama_refine.propose_merges", side_effect=AssertionError("unnecessary call")):
                _, report = run(args)
            table = report["pages"][0]["tables"][0]
            self.assertEqual(table["method"], "raster")
            self.assertEqual(table["model_review"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
