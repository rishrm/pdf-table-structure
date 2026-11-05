"""Later text validation reuses exact-region evidence within the same budget."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import img2table_structure as adapter
from test_img2table_structure import ImageRegion, GeometryCell, backend_available


@unittest.skipUnless(backend_available(), "pinned image backend unavailable")
class ReusedPixelEvidenceTests(unittest.TestCase):
    def backend(self, groups=()):
        np, _, cv2 = adapter._load_backend()

        class Extractor:
            characteristics = object()
            horizontal_lines = []
            vertical_lines = []

            def __init__(self, img):
                pass

            def extract_tables(self, **kwargs):
                return list(groups)

        return np, Extractor, cv2

    def test_no_candidates_prepare_masks_lazily_and_render_only_once(self):
        region = ImageRegion(Image.new("RGB", (640, 480), "white"),
                             bounds=(100, 200, 420, 440))
        cache = {}
        with patch.object(adapter, "_load_backend", return_value=self.backend()), \
                patch.object(adapter, "_render", wraps=adapter._render) as render, \
                patch.object(adapter, "_prepare_pixel_evidence", wraps=adapter._prepare_pixel_evidence) as prepare, \
                patch.object(adapter, "_pixel_evidence", wraps=adapter._pixel_evidence) as sample:
            groups, _, details = adapter.page_grids(region, evidence_cache=cache)
            self.assertEqual(groups, [])
            self.assertEqual(prepare.call_count, 0)
            first = cache["sample_candidate"]([110, 210, 160, 250])
            self.assertEqual(first["status"], "ok")
            self.assertEqual(sample.call_args.args[0], [(20, 20, 120, 100)])
            cache["sample_candidate"]([200, 300, 250, 350])
            self.assertEqual(sample.call_args.args[0], [(200, 200, 300, 300)])
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(render.call_count, 1)
            self.assertEqual(details["figure_context"]["coordinate_system"], "pdf_page_points")
            # Image arrays and the callable are internal, never serialized.
            json.dumps(details, allow_nan=False)

    def test_existing_image_candidate_and_text_sample_share_pixel_budget(self):
        table = SimpleNamespace(rows=[SimpleNamespace(cells=[
            GeometryCell(x, y, x + 20, y + 20) for x in (10, 30)]) for y in (10, 30)])
        cache = {}
        with patch.object(adapter, "_load_backend", return_value=self.backend([table])), \
                patch.object(adapter, "MAX_EVIDENCE_TOTAL_PIXELS", 2000):
            _, _, details = adapter.page_grids(
                ImageRegion(Image.new("RGB", (100, 100), "white")), evidence_cache=cache)
            self.assertEqual(details["returned_group_metadata"][0]["pixel_evidence"]["status"], "ok")
            # Initial 41x41 sample consumed 1681 of 2000 pixels. A fresh
            # 21x21 sample cannot silently receive a new full-page allowance.
            result = cache["sample_candidate"]([10, 10, 30, 30])
            self.assertEqual(result, {"status": "unavailable", "reason": "page_evidence_pixel_budget"})

    def test_missing_optional_backend_clears_prior_page_sampler(self):
        cache = {"sample_candidate": lambda bbox: {"status": "ok"}}
        with patch.object(adapter.metadata, "version", side_effect=adapter.metadata.PackageNotFoundError):
            _, _, details = adapter.page_grids(
                ImageRegion(Image.new("RGB", (100, 100), "white")), evidence_cache=cache)
        self.assertEqual(cache, {})
        self.assertEqual(details["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
