"""Independent image evidence must control layout candidate acceptance."""
import unittest

from test_detection_regressions import (DetectionPage, detection_backends,
                                       rectangular_grid, words_in_grid)
from table_structure import detect_page


class CandidateEvidenceTests(unittest.TestCase):
    def result(self, cells, pixel, rules=()):
        with detection_backends(img2table=[cells]) as (_, backend):
            backend.return_value = ([cells], None, {"warnings": [],
                "returned_group_metadata": [{"pixel_evidence": pixel}],
                "horizontal_rules": list(rules)})
            return detect_page(DetectionPage(images=True), "img2table")

    def test_few_aligned_marks_without_rule_support_need_review(self):
        result = self.result(rectangular_grid(rows=3, columns=3),
            {"status": "ok", "large_component_ink_fraction": 0,
             "character_component_ink_fraction": 0.9})
        self.assertEqual(result["tables"], [])
        self.assertIn("insufficient_repeated_layout_or_rules",
                      result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_small_table_with_independent_ruling_is_retained(self):
        result = self.result(rectangular_grid(rows=3, columns=3),
            {"status": "ok", "large_component_ink_fraction": 0,
             "character_component_ink_fraction": 0.9}, [[60, 100, 270, 100]])
        self.assertEqual(len(result["tables"]), 1)

    def test_graphic_dominated_candidate_is_not_a_table(self):
        result = self.result(rectangular_grid(rows=8, columns=4),
            {"status": "ok", "large_component_ink_fraction": 0.9,
             "character_component_ink_fraction": 0.05})
        self.assertEqual(result["tables"], [])
        self.assertIn("graphic_dominated_ink",
                      result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_repeated_borderless_rows_can_be_retained_without_rules(self):
        result = self.result(rectangular_grid(rows=5, columns=3),
            {"status": "ok", "large_component_ink_fraction": 0,
             "character_component_ink_fraction": 0.9})
        self.assertEqual(len(result["tables"]), 1)

    def test_image_and_separate_prose_are_not_joined_as_table_columns(self):
        cells = rectangular_grid(rows=5, columns=4)
        words = words_in_grid(cells)
        # The last column consists of several separate prose words per band.
        for box in cells[3::4]:
            words.extend({"x0": box[0] + 30 + offset, "top": box[1] + 8,
                          "x1": box[0] + 34 + offset, "bottom": box[1] + 18,
                          "text": "local-only"} for offset in (0, 8, 16))
        page = DetectionPage(words=words, images=True)
        page.images = [{"x0": 50, "top": 70, "x1": 250, "bottom": 300}]
        with detection_backends(img2table=[cells]):
            result = detect_page(page, "img2table")
        self.assertEqual(result["tables"], [])
        self.assertIn("layout_crosses_image_and_neighboring_prose",
                      result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_unruled_short_cells_and_large_prose_column_are_rejected(self):
        cells = [(x0, top, x1 + (70 if x0 == 270 else 0), bottom)
                 for x0, top, x1, bottom in rectangular_grid(rows=5, columns=4)]
        words = words_in_grid(cells)
        for box in (box for index, box in enumerate(cells) if index % 4 != 3):
            words.extend({"x0": box[0] + offset, "top": box[1] + 8,
                          "x1": box[0] + offset + 2, "bottom": box[1] + 18,
                          "text": "local-only"} for offset in (35, 40))
        for box in cells[3::4]:
            words.extend({"x0": box[0] + 30 + offset, "top": box[1] + 8,
                          "x1": box[0] + 32 + offset, "bottom": box[1] + 18,
                          "text": "local-only"} for offset in range(0, 60, 5))
        for ruled in (False, True):
            with self.subTest(ruled=ruled), detection_backends(img2table=[cells]) as (_, backend):
                backend.return_value = ([cells], None, {"warnings": [],
                    "horizontal_rules": [[60, 100, 410, 100]] if ruled else []})
                result = detect_page(DetectionPage(words=words), "img2table")
            if ruled:
                self.assertEqual(len(result["tables"]), 1)
            else:
                self.assertEqual(result["tables"], [])
                self.assertIn("unruled_layout_includes_long_prose_column",
                              result["detection_attempts"][0]["rejected"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
