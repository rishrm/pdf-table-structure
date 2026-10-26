"""Complete native envelopes must replace fragments without hiding valid grids."""
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock, patch
import unittest

from table_structure import detect_page
from test_detection_regressions import DetectionPage, rectangular_grid, words_in_grid, detection_backends


@contextmanager
def native_backend(groups=(), complete=True, error=None):
    module = ModuleType("native_structure")
    module.page_grids = Mock(return_value=(list(groups), {"warnings": [], "returned_group_metadata":
        [{"complete_envelope": complete, "basis": "synthetic_printed_geometry"} for _ in groups]}),
        side_effect=error)
    with patch.dict("sys.modules", {"native_structure": module}):
        yield module.page_grids


class NativeRoutingTests(unittest.TestCase):
    def test_complete_envelope_replaces_smaller_vector_fragment_with_new_partition(self):
        fragment = rectangular_grid(rows=2, columns=3, height=30)
        complete = rectangular_grid(rows=4, columns=3, height=40)
        page = DetectionPage(vector=[fragment], words=words_in_grid(complete))
        with native_backend([complete]), detection_backends():
            result = detect_page(page)
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual(table["method"], "native")
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(result["detection_attempts"][0]["retained"], 0)

    def test_existing_complete_vector_grid_remains_preferred(self):
        complete = rectangular_grid(rows=4, columns=3)
        page = DetectionPage(vector=[complete], words=words_in_grid(complete))
        with native_backend([complete]), detection_backends():
            result = detect_page(page)
        self.assertEqual([t["method"] for t in result["tables"]], ["lines"])

    def test_unverified_native_geometry_is_rejected(self):
        cells = rectangular_grid(rows=4, columns=3)
        page = DetectionPage(words=words_in_grid(cells))
        with native_backend([cells], complete=False):
            result = detect_page(page, "native")
        self.assertEqual(result["tables"], [])
        self.assertIn("native_envelope_unverified", result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_native_failure_preserves_vector_result_and_sanitizes_error(self):
        cells = rectangular_grid(rows=4, columns=3)
        page = DetectionPage(vector=[cells], words=words_in_grid(cells))
        with native_backend(error=RuntimeError("private contents")), detection_backends():
            result = detect_page(page)
        self.assertEqual([t["method"] for t in result["tables"]], ["lines"])
        self.assertIn("native_backend_failed:RuntimeError", result["warnings"])
        self.assertNotIn("private contents", str(result))

    def test_distinct_vector_table_outside_native_envelope_is_retained(self):
        first = rectangular_grid(x=40, top=100, rows=3, columns=3)
        second = rectangular_grid(x=330, top=400, rows=4, columns=3)
        page = DetectionPage(vector=[first], words=words_in_grid(first + second))
        with native_backend([second]), detection_backends():
            result = detect_page(page)
        self.assertEqual([t["method"] for t in result["tables"]], ["lines", "native"])

    def test_later_image_context_rejects_raster_flowchart_without_extra_render(self):
        cells = rectangular_grid(x=0, top=0, rows=5, columns=2, width=15, height=13)
        context = {"status": "ok", "connected_box_groups": [{
            "boxes": [[0, 0, 30, 15], [0, 25, 30, 40], [0, 50, 30, 65]],
            "connector_count": 2}], "large_curved_shapes": []}
        with detection_backends(raster=[cells]) as (raster, image):
            image.return_value = ([], None, {"warnings": [], "figure_context": context})
            result = detect_page(DetectionPage(images=True))
        self.assertEqual(result["tables"], [])
        self.assertEqual(raster.call_count, 1)
        self.assertEqual(image.call_count, 1)
        attempt = next(a for a in result["detection_attempts"] if a["method"] == "raster")
        self.assertIn("connected_box_flow_layout", attempt["rejected"][0]["reasons"])
        self.assertEqual(attempt["retained"], 0)

    def test_missing_or_partial_context_keeps_valid_raster_grid(self):
        cells = rectangular_grid(rows=3, columns=3)
        for status in ("unavailable", "partial"):
            with self.subTest(status=status), detection_backends(raster=[cells]) as (_, image):
                image.return_value = ([], None, {"warnings": [], "figure_context": {"status": status}})
                result = detect_page(DetectionPage(images=True))
            self.assertEqual([t["method"] for t in result["tables"]], ["raster"])


if __name__ == "__main__":
    unittest.main()
