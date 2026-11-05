"""Word repair must validate its new region without reusing stale pixels."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

from table_structure import check_grid, detect_page
from test_detection_regressions import (DetectionPage, detection_backends,
                                       rectangular_grid, words_in_grid)


TEXT_PIXELS = {"status": "ok", "large_component_ink_fraction": 0,
               "character_component_ink_fraction": .95}
GRAPHIC_PIXELS = {"status": "ok", "large_component_ink_fraction": .95,
                  "character_component_ink_fraction": .01}


class RepairedImageValidationTests(unittest.TestCase):
    def fixture(self):
        raw = [rectangular_grid(x=40, top=40),
               rectangular_grid(x=130, top=150, rows=2, columns=2)]
        rebuilt = rectangular_grid(x=100, top=120, rows=4, columns=3, width=60)
        note = {"status": "repaired", "basis": "printed_rules_and_word_positions",
                "source_group_indices": [1, 0], "source_candidate_indices": [29, 7],
                "bbox": [100, 120, 280, 240], "header_rows": 1,
                "body_rows": 3, "columns": 3, "semantic_structure_verified": False}
        old_metadata = [{"source_candidate_index": 7, "bbox": [40, 40, 180, 100],
                         "pixel_evidence": dict(TEXT_PIXELS)},
                        {"source_candidate_index": 29, "bbox": [130, 150, 270, 210],
                         "borderless": False, "pixel_evidence": dict(GRAPHIC_PIXELS)}]
        details = {"warnings": [], "horizontal_rules": [[100, 120, 280, 120]],
                   "returned_group_metadata": old_metadata}
        page = DetectionPage(words=words_in_grid(rebuilt))
        page.to_image = Mock(side_effect=AssertionError("Unexpected second render"))
        return page, raw, rebuilt, note, details, old_metadata

    def run_repaired(self, sampler):
        page, raw, rebuilt, note, details, old_metadata = self.fixture()
        original_metadata = copy.deepcopy(old_metadata)

        def provide_cache(region, **kwargs):
            if sampler is not None:
                kwargs["evidence_cache"]["sample_candidate"] = sampler
            return raw, None, details

        with detection_backends() as (raster, image), \
                patch("layout_repair.repair_word_groups", return_value=([rebuilt], [note])) as repair:
            image.side_effect = provide_cache
            result = detect_page(page, "img2table")
            self.assertEqual(image.call_count, 1)
            raster.assert_not_called()
            self.assertEqual(repair.call_count, 1)
        page.to_image.assert_not_called()
        self.assertEqual(old_metadata, original_metadata)
        json.dumps(result, allow_nan=False)
        return result, note

    def test_repaired_bbox_is_sampled_once_and_original_provenance_is_preserved(self):
        sampler = Mock(return_value=dict(TEXT_PIXELS))
        result, note = self.run_repaired(sampler)
        sampler.assert_called_once_with(note["bbox"])
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual(table["bbox"], note["bbox"])
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(check_grid(table), [])
        self.assertEqual(table["evidence"]["pixel"], TEXT_PIXELS)
        self.assertEqual(table["layout_repair"]["source_group_indices"], [1, 0])
        self.assertEqual(table["layout_repair"]["source_candidate_indices"], [29, 7])
        metadata = result["img2table_detection"]["returned_group_metadata"][0]
        self.assertEqual(metadata["source_candidate_index"], 29)
        self.assertFalse(metadata["borderless"])
        self.assertEqual(metadata["unique_cells"], 12)
        self.assertEqual(metadata["pixel_evidence"], TEXT_PIXELS)

    def test_graphic_evidence_for_the_repaired_region_rejects_the_candidate(self):
        sampler = Mock(return_value=dict(GRAPHIC_PIXELS))
        result, note = self.run_repaired(sampler)
        sampler.assert_called_once_with(note["bbox"])
        self.assertEqual(result["tables"], [])
        rejected = result["detection_attempts"][0]["rejected"]
        self.assertEqual(rejected[0]["bbox"], note["bbox"])
        self.assertIn("graphic_dominated_ink", rejected[0]["reasons"])
        self.assertEqual(result["img2table_detection"]["returned_group_metadata"][0]
                         ["pixel_evidence"], GRAPHIC_PIXELS)

    def test_sampling_failure_preserves_repaired_geometry_and_sanitizes_error(self):
        sampler = Mock(side_effect=RuntimeError("PRIVATE DOCUMENT CONTENT"))
        result, note = self.run_repaired(sampler)
        sampler.assert_called_once_with(note["bbox"])
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["bbox"], note["bbox"])
        self.assertEqual(result["tables"][0]["evidence"]["pixel"],
                         {"status": "unavailable", "reason": "RuntimeError"})
        self.assertNotIn("PRIVATE DOCUMENT CONTENT", str(result))

    def test_budget_exhaustion_or_partial_evidence_does_not_discard_repaired_table(self):
        samples = [{"status": "unavailable", "reason": "page_evidence_pixel_budget"},
                   {**GRAPHIC_PIXELS, "status": "partial", "reason": "component_limit"}]
        for sample in samples:
            with self.subTest(status=sample["status"]):
                sampler = Mock(return_value=sample)
                result, note = self.run_repaired(sampler)
                sampler.assert_called_once_with(note["bbox"])
                self.assertEqual(len(result["tables"]), 1)
                self.assertEqual(result["tables"][0]["bbox"], note["bbox"])
                self.assertEqual(result["tables"][0]["evidence"]["pixel"], sample)

    def test_unavailable_sampler_keeps_geometry_without_reusing_old_graphic_ratios(self):
        result, note = self.run_repaired(None)
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["bbox"], note["bbox"])
        pixel = result["tables"][0]["evidence"]["pixel"]
        self.assertEqual(pixel["status"], "unavailable")
        self.assertNotIn("large_component_ink_fraction", pixel)

    def test_unchanged_ruled_candidate_does_not_request_an_extra_sample(self):
        cells = rectangular_grid(x=100, top=120, rows=4, columns=3, width=60)
        sampler = Mock(return_value=dict(TEXT_PIXELS))
        page = DetectionPage(words=words_in_grid(cells))
        page.to_image = Mock(side_effect=AssertionError("Unexpected second render"))

        def provide_cache(region, **kwargs):
            kwargs["evidence_cache"]["sample_candidate"] = sampler
            return [cells], None, {"warnings": [], "horizontal_rules": [[100, 120, 280, 120]],
                "returned_group_metadata": [{"source_candidate_index": 4,
                    "pixel_evidence": dict(TEXT_PIXELS)}]}

        with detection_backends() as (raster, image), \
                patch("layout_repair.repair_word_groups", return_value=(
                    [cells], [{"status": "unchanged", "source_group_indices": [0]}])):
            image.side_effect = provide_cache
            result = detect_page(page, "img2table")
            self.assertEqual(image.call_count, 1)
            raster.assert_not_called()
        self.assertEqual(len(result["tables"]), 1)
        sampler.assert_not_called()
        page.to_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
