"""Regression coverage for candidate acceptance, fallback routing, and prose.

Routing tests use small page doubles and deterministic geometry backends.  The
borderless/prose tests use actual PDFs so word segmentation and text-table
geometry remain part of the behavior under test.  No OCR models or network
services are needed.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pdfplumber
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from make_demo import make_demo
from table_structure import check_grid, detect_page


def rectangular_grid(x=60, top=100, rows=2, columns=2, width=70, height=30):
    return [(x + column * width, top + row * height,
             x + (column + 1) * width, top + (row + 1) * height)
            for row in range(rows) for column in range(columns)]


def words_in_grid(cells):
    return [{"x0": x0 + 5, "top": top + 8, "x1": x0 + 25,
             "bottom": top + 18, "text": "local-only"}
            for x0, top, _, _ in cells]


class DetectionPage:
    """Only the page surface needed by candidate routing is simulated."""
    page_number = 1
    width = 612
    height = 792
    rotation = 0
    bbox = (0, 0, 612, 792)

    def __init__(self, vector=(), text=(), words=(), images=False):
        self.vector = [SimpleNamespace(cells=cells) for cells in vector]
        self.text = [SimpleNamespace(cells=cells) for cells in text]
        self.words = list(words)
        self.chars = [{"x0": word["x0"], "top": word["top"],
                       "x1": word["x1"], "bottom": word["bottom"]}
                      for word in self.words]
        self.images = ([{"x0": 0, "top": 0, "x1": self.width,
                         "bottom": self.height, "width": self.width,
                         "height": self.height}] if images else [])
        self.find_calls = []

    def crop(self, bbox):
        cropped = DetectionPage(words=self.words, images=bool(self.images))
        cropped.vector, cropped.text = self.vector, self.text
        cropped.bbox = tuple(bbox)
        cropped.find_calls = self.find_calls
        return cropped

    def find_tables(self, settings):
        strategy = settings["vertical_strategy"]
        self.find_calls.append(strategy)
        return self.text if strategy == "text" else self.vector

    def extract_words(self, **kwargs):
        return self.words


@contextmanager
def detection_backends(raster=(), img2table=()):
    """Keep optional model-backed detection outside these routing tests."""
    transform = {"deskew_degrees": 0, "render_bbox": [0, 0, 612, 792],
                 "image_size": [612, 792], "scale_x": 1, "scale_y": 1}
    raster_mock = Mock(return_value=(list(raster), transform, {"warnings": []}))
    img_mock = Mock(return_value=(list(img2table), None, {"warnings": []}))
    adapter = ModuleType("img2table_structure")
    adapter.page_grids = img_mock
    with patch("raster_structure.page_grids", raster_mock), \
            patch.dict(sys.modules, {"img2table_structure": adapter}):
        yield raster_mock, img_mock


def make_article(path):
    """Aligned journal columns with variable word starts, not a data table."""
    passages = [
        "Regional planning depends on a shared account of the evidence and "
        "the local priorities. Members review each draft before a meeting "
        "and publish their findings throughout the season. The report "
        "considers several projects whose schedules vary with weather and "
        "public demand. The discussion records their rationale and leaves "
        "room for continued debate among residents and staff.",
        "The committee reviews changes in available resources every month. "
        "Its members describe the alternatives and explain the assumptions "
        "that informed their judgments. Progress depends on cooperation "
        "between local groups and on realistic schedules. Future reports "
        "will revisit these questions as the work proceeds and new evidence "
        "becomes available for discussion in public meetings.",
    ]
    document = canvas.Canvas(str(path), pagesize=(612, 792))
    document.setFont("Times-Roman", 10)
    for x, passage in zip((48, 330), passages):
        pending = (passage + " " + passage + " " + passage).split()
        for row in range(28):
            line = []
            while pending and stringWidth(" ".join(line + [pending[0]]),
                                          "Times-Roman", 10) <= 235:
                line.append(pending.pop(0))
            document.drawString(x, 730 - row * 16, " ".join(line))
    document.save()
    return Path(path)


class DetectionRoutingTests(unittest.TestCase):
    def assert_valid_tables(self, result, methods):
        self.assertEqual([table["method"] for table in result["tables"]], methods)
        self.assertEqual(result["status"], "candidates_detected")
        self.assertTrue(all(check_grid(table) == [] for table in result["tables"]))
        self.assertEqual(len({table["id"] for table in result["tables"]}),
                         len(result["tables"]))

    def test_rejected_vector_candidates_do_not_block_raster_fallback(self):
        good = rectangular_grid(x=300, top=400)
        candidates = {
            "one_box": [(30, 30, 100, 60)],
            "one_row": rectangular_grid(rows=1, columns=3),
            "one_column": rectangular_grid(rows=3, columns=1),
            "uncovered_slot": rectangular_grid()[:-1],
            "invalid_box": [(30, 30, 30, 60)],
        }
        for label, cells in candidates.items():
            with self.subTest(candidate=label), detection_backends(raster=[good]) as backends:
                result = detect_page(DetectionPage(vector=[cells], images=True), "auto")
                self.assert_valid_tables(result, ["raster"])
                self.assertEqual(result["tables"][0]["bbox"], [300, 400, 440, 460])
                self.assertTrue(backends[0].called)

    def test_rejected_raster_candidate_does_not_block_img2table_fallback(self):
        good = rectangular_grid(x=300, top=400)
        page = DetectionPage(vector=[[(10, 10, 30, 30)]], images=True)
        with detection_backends(raster=[rectangular_grid(rows=1)],
                                img2table=[good]) as backends:
            result = detect_page(page, "auto")
        self.assert_valid_tables(result, ["img2table"])
        self.assertTrue(backends[0].called)
        self.assertTrue(backends[1].called)

    def test_rejected_vector_candidate_does_not_block_text_fallback(self):
        good = rectangular_grid(rows=4, columns=3)
        page = DetectionPage(vector=[[(10, 10, 30, 30)]], text=[good],
                             words=words_in_grid(good))
        with detection_backends():
            result = detect_page(page, "auto")
        self.assert_valid_tables(result, ["text"])
        self.assertIn("text", page.find_calls)
        self.assertEqual((result["tables"][0]["row_count"],
                          result["tables"][0]["column_count"]), (4, 3))

    def test_partial_and_overlapping_grids_are_rejected(self):
        valid = rectangular_grid()
        candidates = {"missing_slot": valid[:-1],
                      "overlap": valid + [(60, 100, 130, 160)]}
        for label, cells in candidates.items():
            with self.subTest(candidate=label), detection_backends():
                result = detect_page(DetectionPage(vector=[cells]), "lines")
                self.assertEqual(len(result["tables"]), 0)
                self.assertNotEqual(result["status"], "candidates_detected")

    def test_rejected_candidate_does_not_discard_valid_neighbor(self):
        good = rectangular_grid(x=300, top=400)
        with detection_backends():
            result = detect_page(DetectionPage(vector=[rectangular_grid()[:-1], good]),
                                 "lines")
        self.assert_valid_tables(result, ["lines"])
        self.assertEqual(result["tables"][0]["bbox"], [300, 400, 440, 460])

    def test_mixed_page_keeps_distinct_vector_and_raster_tables(self):
        vector = rectangular_grid(x=40, top=50)
        scanned = rectangular_grid(x=300, top=400)
        with detection_backends(raster=[scanned]) as backends:
            result = detect_page(DetectionPage(vector=[vector], images=True), "auto")
        self.assert_valid_tables(result, ["lines", "raster"])
        self.assertTrue(backends[0].called)
        self.assertEqual([table["bbox"] for table in result["tables"]],
                         [[40, 50, 180, 110], [300, 400, 440, 460]])

    def test_overlapping_raster_duplicate_is_not_returned_twice(self):
        vector = rectangular_grid(x=40, top=50)
        scanned = rectangular_grid(x=300, top=400)
        with detection_backends(raster=[vector, scanned]):
            result = detect_page(DetectionPage(vector=[vector], images=True), "auto")
        self.assert_valid_tables(result, ["lines", "raster"])
        self.assertEqual([table["bbox"] for table in result["tables"]],
                         [[40, 50, 180, 110], [300, 400, 440, 460]])

    def test_contained_cell_subset_is_not_a_second_table(self):
        vector = rectangular_grid(rows=8, columns=3)
        subset = rectangular_grid(rows=4, columns=3)
        with detection_backends(img2table=[subset]):
            result = detect_page(DetectionPage(vector=[vector], images=True), "auto")
        self.assert_valid_tables(result, ["lines"])
        self.assertEqual(result["tables"][0]["row_count"], 8)

    def test_fuller_matching_grid_replaces_an_earlier_fragment(self):
        for fragment_rows in (4, 6):
            with self.subTest(fragment_rows=fragment_rows):
                fragment = rectangular_grid(rows=fragment_rows, columns=3)
                fuller = rectangular_grid(rows=8, columns=3)
                with detection_backends(img2table=[fuller]):
                    result = detect_page(DetectionPage(vector=[fragment], images=True), "auto")
                self.assert_valid_tables(result, ["img2table"])
                self.assertEqual(result["tables"][0]["row_count"], 8)

    def test_raster_failure_preserves_vector_and_reaches_layout_fallback(self):
        vector = rectangular_grid(x=40, top=50)
        layout = rectangular_grid(x=300, top=400)
        with detection_backends(img2table=[layout]) as (raster, img):
            raster.side_effect = RuntimeError("PRIVATE DOCUMENT TEXT")
            result = detect_page(DetectionPage(vector=[vector], images=True), "auto")
        self.assert_valid_tables(result, ["lines", "img2table"])
        self.assertTrue(img.called)
        self.assertIn("raster_backend_failed:RuntimeError", result["warnings"])
        self.assertNotIn("PRIVATE DOCUMENT TEXT", str(result))


class BorderlessEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        base = Path(cls.folder.name)
        cls.article = make_article(base / "article.pdf")
        cls.demo = make_demo(base / "demo.pdf")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_two_column_article_is_not_a_giant_table(self):
        with pdfplumber.open(self.article) as pdf, detection_backends():
            page = pdf.pages[0]
            # This fixture exercises the original false positive: repeated
            # article baselines give the text finder enough alignments.
            self.assertTrue(page.find_tables({"vertical_strategy": "text",
                                              "horizontal_strategy": "text"}))
            for strategy in ("auto", "text"):
                with self.subTest(strategy=strategy):
                    result = detect_page(page, strategy)
                    self.assertEqual(len(result["tables"]), 0)
                    self.assertNotEqual(result["status"], "candidates_detected")

    def test_simple_borderless_table_keeps_geometry(self):
        with pdfplumber.open(self.demo) as pdf, detection_backends():
            for strategy in ("auto", "text"):
                with self.subTest(strategy=strategy):
                    result = detect_page(pdf.pages[1], strategy,
                                         bbox=[50, 125, 500, 290])
                    self.assertEqual(len(result["tables"]), 1)
                    table = result["tables"][0]
                    self.assertEqual(table["method"], "text")
                    self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
                    self.assertEqual(len(table["cells"]), 12)
                    self.assertEqual(check_grid(table), [])


if __name__ == "__main__":
    unittest.main()
