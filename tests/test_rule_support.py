"""An unrelated page rule or image stroke must not validate a tiny grid."""
import unittest

from table_structure import detect_page, image_evidence, structure_from_cells
from test_detection_regressions import DetectionPage, detection_backends, rectangular_grid


TEXT_PIXELS = {"status": "ok", "large_component_ink_fraction": 0,
               "character_component_ink_fraction": .95}


class RuleSupportTests(unittest.TestCase):
    def inspect(self, cells, rules, pixel=None):
        table = structure_from_cells(cells, "img2table")
        return image_evidence(table, {"pixel_evidence": TEXT_PIXELS if pixel is None else pixel}, rules)

    def image_result(self, cells, rules):
        with detection_backends() as (_, backend):
            backend.return_value = ([cells], None, {"warnings": [],
                "horizontal_rules": rules,
                "returned_group_metadata": [{"pixel_evidence": TEXT_PIXELS}]})
            return detect_page(DetectionPage(images=True), "img2table")

    def assert_weak_grid(self, cells, rules):
        reasons, evidence = self.inspect(cells, rules)
        self.assertFalse(evidence["long_horizontal_rule_support"])
        self.assertIn("insufficient_repeated_layout_or_rules", reasons)
        result = self.image_result(cells, rules)
        self.assertEqual(result["tables"], [])
        self.assertIn("insufficient_repeated_layout_or_rules",
                      result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_running_header_above_candidate_does_not_supply_table_evidence(self):
        cells = rectangular_grid(rows=3, columns=2)
        self.assert_weak_grid(cells, [[60, 88, 200, 88]])

    def test_axis_outside_candidate_does_not_validate_its_aligned_labels(self):
        cells = rectangular_grid(x=50, top=100, rows=3, columns=7, width=45, height=16)
        self.assert_weak_grid(cells, [[36, 97.4, 358, 97.4]])

    def test_rule_below_candidate_is_not_its_bottom_border(self):
        cells = rectangular_grid(rows=3, columns=3)
        self.assert_weak_grid(cells, [[60, 192.5, 270, 192.5]])

    def test_short_texture_strokes_and_overshooting_stroke_do_not_validate_tiny_grid(self):
        cells = rectangular_grid(x=100, top=100, rows=2, columns=2, width=5, height=5)
        rules = [[101.5, 98.2, 112, 98.2], [99.7, 103.3, 123.4, 103.3],
                 [99.7, 110, 110.1, 110]]
        self.assert_weak_grid(cells, rules)

    def test_real_internal_header_separator_supports_a_partial_small_table(self):
        cells = rectangular_grid(x=20, top=100, rows=2, columns=5, width=80, height=25)
        # A detected table may end before its actual printed rule does. Modest
        # overhang is allowed; complete outer-border equality is not required.
        rules = [[19.5, 127, 504, 127]]
        reasons, evidence = self.inspect(cells, rules)
        self.assertTrue(evidence["long_horizontal_rule_support"])
        self.assertEqual(reasons, [])
        self.assertEqual(len(self.image_result(cells, rules)["tables"]), 1)

    def test_real_top_rule_and_rounding_tolerance_preserve_small_table(self):
        cells = rectangular_grid(rows=3, columns=3)
        for top in (100, 99.2):
            with self.subTest(rule_y=top):
                reasons, evidence = self.inspect(cells, [[60, top, 270, top]])
                self.assertTrue(evidence["long_horizontal_rule_support"])
                self.assertEqual(reasons, [])

    def test_rule_decision_is_translation_and_scale_independent_for_normal_size_tables(self):
        original = rectangular_grid(rows=3, columns=3)
        for scale, dx, dy in ((.5, 0, 0), (1, 120, -50), (1.5, -20, 80)):
            move = lambda box: [box[0]*scale+dx, box[1]*scale+dy,
                                box[2]*scale+dx, box[3]*scale+dy]
            cells = [move(box) for box in original]
            with self.subTest(scale=scale, offset=(dx, dy)):
                reasons, evidence = self.inspect(cells, [move([60, 100, 270, 100])])
                self.assertEqual(reasons, [])
                self.assertTrue(evidence["long_horizontal_rule_support"])
                reasons, evidence = self.inspect(cells, [move([60, 80, 270, 80])])
                self.assertIn("insufficient_repeated_layout_or_rules", reasons)
                self.assertFalse(evidence["long_horizontal_rule_support"])

    def test_missing_or_partial_pixels_do_not_turn_rule_absence_into_rejection(self):
        cells = rectangular_grid(rows=3, columns=3)
        for pixel in ({}, {"status": "unavailable"}, {"status": "partial"}):
            with self.subTest(pixel=pixel):
                reasons, evidence = self.inspect(cells, [[60, 88, 270, 88]], pixel)
                self.assertEqual(reasons, [])
                self.assertFalse(evidence["long_horizontal_rule_support"])


if __name__ == "__main__":
    unittest.main()
