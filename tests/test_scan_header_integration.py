"""Header enrichment must preserve adapter provenance and fail safely."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from PIL import Image
import img2table_structure as adapter
from table_structure import check_grid, structure_from_cells
from test_img2table_structure import GeometryCell, ImageRegion, backend_available


@unittest.skipUnless(backend_available(), "Pinned image backend is unavailable")
class ScanHeaderIntegrationTests(unittest.TestCase):
    def fixture(self, native=False):
        np, _, cv2 = adapter._load_backend()
        cells = [(x, y, x + 40, y + 10) for y in (30, 40, 50, 60) for x in (10, 50)]
        invalid = SimpleNamespace(rows=[SimpleNamespace(cells=[GeometryCell(0, 0, 0, 1)])])
        valid = SimpleNamespace(rows=[SimpleNamespace(cells=[GeometryCell(*b) for b in cells])])

        class Extractor:
            characteristics = object()
            horizontal_lines = []
            vertical_lines = []

            def __init__(self, img):
                pass

            def extract_tables(self, **kwargs):
                return [invalid, valid]

        region = ImageRegion(Image.new("RGB", (100, 90), "white"))
        region.chars = [{"x0": 10}] if native else []
        return region, cells, (np, Extractor, cv2)

    def test_added_header_keeps_original_source_index_after_invalid_group(self):
        region, cells, backend = self.fixture()
        enriched = [(10, 10, 90, 30)] + cells
        note = {"status": "repaired", "basis": "synthetic_header_geometry",
                "header_rows": 1, "body_rows": 4, "columns": 2}
        with patch.object(adapter, "_load_backend", return_value=backend), \
                patch("scan_headers.repair_headers", return_value=([enriched], [note])):
            groups, _, details = adapter.page_grids(region)
        self.assertEqual(groups, [enriched])
        metadata = details["returned_group_metadata"][0]
        self.assertEqual(metadata["source_candidate_index"], 1)
        self.assertEqual(metadata["layout_repair"]["source_candidate_indices"], [1])
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 2))
        self.assertEqual(check_grid(table), [])
        self.assertEqual(table["cells"][0]["column_span"], 2)

    def test_header_failure_keeps_existing_grid_and_omits_exception_text(self):
        region, cells, backend = self.fixture()
        with patch.object(adapter, "_load_backend", return_value=backend), \
                patch("scan_headers.repair_headers", side_effect=RuntimeError("PRIVATE CONTENT")):
            groups, _, details = adapter.page_grids(region)
        self.assertEqual(groups, [cells])
        self.assertIn("scan_header_repair_failed:RuntimeError", details["warnings"])
        self.assertNotIn("PRIVATE CONTENT", str(details))
        self.assertEqual(details["returned_group_metadata"][0]["source_candidate_index"], 1)

    def test_scan_only_header_pass_does_not_run_on_native_text_page(self):
        region, cells, backend = self.fixture(native=True)
        with patch.object(adapter, "_load_backend", return_value=backend), \
                patch("scan_headers.repair_headers") as repair:
            groups, _, _ = adapter.page_grids(region)
        self.assertEqual(groups, [cells])
        repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
