"""Unruled image grids may share text-boundary correction without hiding evidence."""
import copy
import unittest
from unittest.mock import Mock

from table_structure import check_grid, detect_page, structure_from_cells
from test_detection_regressions import DetectionPage, detection_backends, rectangular_grid, words_in_grid


TEXT_PIXEL = {"status": "ok", "large_component_ink_fraction": 0,
              "character_component_ink_fraction": .95}
GRAPHIC_PIXEL = {"status": "ok", "large_component_ink_fraction": .9,
                 "character_component_ink_fraction": .05}


def rejected_reasons(result):
    return {reason for attempt in result["detection_attempts"]
            for item in attempt["rejected"] for reason in item["reasons"]}


def fixture(*, clipped=False, spans=False):
    cells = rectangular_grid(rows=4, columns=3)
    if spans:
        cells = [b for b in cells if b[1] != 100 and not (b[0] == 60 and b[1] in (130, 160))]
        cells += [(60, 100, 270, 130), (60, 130, 130, 190)]
    words = words_in_grid(cells)
    if clipped:
        for word in words:
            if word["top"] == 198:
                word["top"], word["bottom"] = 215, 223
    return cells, words


class UnruledWordBoundaryTests(unittest.TestCase):
    def run_candidate(self, cells, words, *, metadata=None, rules=(), sample=TEXT_PIXEL):
        sampler = Mock(return_value=copy.deepcopy(sample))
        details = {"warnings": [], "horizontal_rules": list(rules)}
        if metadata is not None:
            details["returned_group_metadata"] = [metadata]

        def backend(region, **kwargs):
            kwargs["evidence_cache"]["sample_candidate"] = sampler
            return [cells], None, details

        with detection_backends() as (_, image):
            image.side_effect = backend
            result = detect_page(DetectionPage(words=words, images=True), "img2table")
        return result, sampler

    def test_clipped_native_words_fit_final_box_and_exact_region_is_resampled(self):
        cells, words = fixture(clipped=True)
        result, sampler = self.run_candidate(cells, words, metadata={"pixel_evidence": TEXT_PIXEL})
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(table["bbox"][3], 224)
        self.assertEqual(table["evidence"]["word_cut_fraction"], 0)
        self.assertEqual(check_grid(table), [])
        sampler.assert_called_once_with(table["bbox"])
        self.assertIn("native_word_boundaries_require_review", table["warnings"])
        for word in words:
            matches = [c for c in table["cells"] if c["bbox"][0] <= word["x0"] and
                       c["bbox"][1] <= word["top"] and c["bbox"][2] >= word["x1"] and
                       c["bbox"][3] >= word["bottom"]]
            self.assertEqual(len(matches), 1)

    def test_normalization_preserves_row_and_column_spans(self):
        cells, words = fixture(clipped=True, spans=True)
        result, _ = self.run_candidate(cells, words, metadata={"pixel_evidence": TEXT_PIXEL})
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(sorted((c["row"], c["column"], c["row_span"], c["column_span"])
            for c in table["cells"] if c["row_span"] > 1 or c["column_span"] > 1),
            [(0, 0, 1, 3), (1, 0, 2, 1)])
        self.assertEqual(check_grid(table), [])

    def test_small_columns_are_kept_as_distinct_fields(self):
        cells = rectangular_grid(rows=4, columns=3, width=14, height=12)
        words = [{"x0": x0+2, "top": top+3, "x1": x0+8, "bottom": top+9,
                  "text": "local-only"} for x0, top, _, _ in cells]
        result, sampler = self.run_candidate(cells, words, metadata={"pixel_evidence": TEXT_PIXEL})
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(len(table["cells"]), 12)
        self.assertEqual(check_grid(table), [])
        sampler.assert_called_once_with(table["bbox"])

    def test_printed_rule_keeps_original_geometry_without_normalization(self):
        cells, words = fixture()
        result, sampler = self.run_candidate(cells, words, metadata={"pixel_evidence": TEXT_PIXEL},
                                             rules=[[60, 100, 270, 100]])
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        expected = structure_from_cells(cells, "img2table")
        for key in ("bbox", "x_boundaries", "y_boundaries", "cells"):
            self.assertEqual(table[key], expected[key])
        sampler.assert_not_called()
        self.assertNotIn("native_word_boundaries_require_review", table["warnings"])

    def test_image_only_grid_is_not_changed_by_native_word_normalization(self):
        cells, _ = fixture()
        result, sampler = self.run_candidate(cells, [], metadata={"pixel_evidence": TEXT_PIXEL})
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        expected = structure_from_cells(cells, "img2table")
        for key in ("bbox", "x_boundaries", "y_boundaries", "cells"):
            self.assertEqual(table[key], expected[key])
        sampler.assert_not_called()

    def test_previously_repaired_grid_keeps_its_existing_boundaries(self):
        cells, words = fixture()
        metadata = {"pixel_evidence": TEXT_PIXEL, "layout_repair": {
            "status": "repaired", "bbox": [60, 100, 270, 220],
            "basis": "printed_rules_and_word_positions"}}
        result, sampler = self.run_candidate(cells, words, metadata=metadata)
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["bbox"], [60, 100, 270, 220])
        self.assertNotIn("native_word_boundaries_require_review", result["tables"][0]["warnings"])
        sampler.assert_not_called()

    def test_native_caption_outside_image_table_does_not_move_its_boundaries(self):
        cells, _ = fixture()
        words = [{"x0": 70, "top": 70, "x1": 180, "bottom": 80,
                  "text": "outside-caption"}]
        result, sampler = self.run_candidate(cells, words, metadata={"pixel_evidence": TEXT_PIXEL})
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["bbox"], [60, 100, 270, 220])
        self.assertNotIn("native_word_boundaries_require_review", result["tables"][0]["warnings"])
        sampler.assert_not_called()

    def test_original_graphic_rejection_cannot_be_erased_by_missing_resample(self):
        cells, words = fixture()
        metadata = {"pixel_evidence": GRAPHIC_PIXEL}
        result, sampler = self.run_candidate(cells, words, metadata=metadata,
                                             sample={"status": "unavailable"})
        self.assertEqual(result["tables"], [])
        self.assertIn("graphic_dominated_ink", rejected_reasons(result))
        sampler.assert_not_called()

    def test_raw_candidate_metadata_is_not_rewritten_for_the_adjusted_box(self):
        cells, words = fixture(clipped=True)
        metadata = {"bbox": [60, 100, 270, 220], "source_candidate_index": 7,
                    "pixel_evidence": copy.deepcopy(TEXT_PIXEL)}
        original = copy.deepcopy(metadata)
        result, sampler = self.run_candidate(cells, words, metadata=metadata)
        self.assertEqual(metadata, original)
        raw = result["img2table_detection"]["returned_group_metadata"][0]
        self.assertEqual(raw, original)
        self.assertEqual(len(result["tables"]), 1)
        self.assertNotEqual(result["tables"][0]["bbox"], raw["bbox"])
        sampler.assert_called_once_with(result["tables"][0]["bbox"])

    def test_final_box_graphic_evidence_is_honored_without_original_metadata(self):
        cells, words = fixture()
        result, sampler = self.run_candidate(cells, words, metadata=None, sample=GRAPHIC_PIXEL)
        self.assertEqual(result["tables"], [])
        self.assertIn("graphic_dominated_ink", rejected_reasons(result))
        sampler.assert_called_once()
        rejected = result["detection_attempts"][0]["rejected"][0]
        self.assertEqual(sampler.call_args.args[0], rejected["bbox"])


if __name__ == "__main__":
    unittest.main()
