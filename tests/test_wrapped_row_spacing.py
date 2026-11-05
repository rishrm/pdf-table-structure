"""Recover variable-height records without swallowing separate sparse records."""
import copy
import json
from pathlib import Path
import unittest

from layout_repair import repair_word_groups
from table_structure import check_grid, structure_from_cells


FIXTURE = Path(__file__).parent / "fixtures" / "967862-word-layout.json"


def fixture():
    data = json.loads(FIXTURE.read_text())
    return data["groups"], data["horizontal"], data["word_boxes"]


def transformed(boxes, scale, dx, dy):
    return [[b[0] * scale + dx, b[1] * scale + dy,
             b[2] * scale + dx, b[3] * scale + dy] for b in boxes]


def grid(xs, ys):
    return [(a, c, b, d) for c, d in zip(ys, ys[1:])
            for a, b in zip(xs, xs[1:])]


class WrappedRowSpacingTests(unittest.TestCase):
    def assert_repaired_record_geometry(self, groups, rules, words,
                                        scale=1.0, dx=0.0, dy=0.0):
        original = copy.deepcopy((groups, rules, words))
        repaired, notes = repair_word_groups(groups, rules, words)
        self.assertEqual((groups, rules, words), original)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(notes[0]["status"], "repaired")
        table = structure_from_cells(repaired[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (21, 3))
        self.assertEqual(check_grid(table), [])
        self.assertEqual(len(table["cells"]), 63)
        self.assertEqual(notes[0]["header_rows"], 1)
        self.assertEqual(notes[0]["body_rows"], 20)
        expected = transformed([[50.74601934703748, 244.78328629949934,
                                  289.36028053204353, 708.4316285844334]],
                               scale, dx, dy)[0]
        for actual, wanted in zip(table["bbox"], expected):
            self.assertAlmostEqual(actual, wanted, delta=.002)
        inside, outside = [], []
        for word in words:
            cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
            (inside if expected[0] <= cx <= expected[2] and
             expected[1] <= cy <= expected[3] else outside).append(word)
        self.assertEqual(len(inside), 159)
        self.assertEqual(len(outside), 18)  # Two-line caption stays outside.
        for word in inside:
            matching = [cell for cell in table["cells"] if
                cell["bbox"][0] - .002 <= word[0] and
                cell["bbox"][1] - .002 <= word[1] and
                cell["bbox"][2] + .002 >= word[2] and
                cell["bbox"][3] + .002 >= word[3]]
            self.assertEqual(len(matching), 1, word)
        for word in outside:
            self.assertLessEqual(word[3], expected[1] + .002)
        self.assertTrue(all(c["row_span"] == 1 and c["column_span"] == 1
                            for c in table["cells"]))

    def test_variable_height_table_keeps_twenty_records_and_printed_envelope(self):
        self.assert_repaired_record_geometry(*fixture())

    def test_translation_and_render_scale_preserve_records_and_caption_exclusion(self):
        groups, rules, words = fixture()
        for scale, dx, dy in ((1.0, 125, -170), (.8, 20, 130), (1.5, -40, 37)):
            with self.subTest(scale=scale, offset=(dx, dy)):
                self.assert_repaired_record_geometry(
                    [transformed(g, scale, dx, dy) for g in groups],
                    transformed(rules, scale, dx, dy),
                    transformed(words, scale, dx, dy), scale, dx, dy)

    def test_a_missing_middle_field_does_not_absorb_an_independent_record(self):
        groups, rules, words = fixture()
        # Remove the single-line middle field from three different records:
        # a short record and two records whose labels/references wrap.
        for start in (291.399, 357.676, 646.510):
            with self.subTest(record_start=start):
                reduced = [w for w in words if not
                    (170 < (w[0] + w[2]) / 2 < 233 and abs(w[1] - start) < .01)]
                self.assertEqual(len(words) - len(reduced), 1)
                repaired, notes = repair_word_groups(groups, rules, reduced)
                self.assertEqual(repaired, groups)
                self.assertEqual(notes[0]["status"], "unchanged")

    def test_equal_spacing_does_not_merge_a_sparse_record_with_its_predecessor(self):
        xs = [10, 80, 150, 230]
        ys = [40, 70] + list(range(80, 201, 10))
        groups = [grid(xs, ys)]
        rules = [[10, y, 230, y] for y in (40, 70, 200)]
        words = [(x + 5, 50, x + 25, 57) for x in xs[:-1]]
        for top in (80, 100, 120, 140, 160, 180):
            words += [(x + 5, top, x + 25, top + 7) for x in xs[:-1]
                      if not (x == 80 and top == 120)]
            words += [(x + 5, top + 10, x + 25, top + 17)
                      for x in (10, 150)]
        repaired, notes = repair_word_groups(groups, rules, words)
        self.assertEqual(repaired, groups)
        self.assertEqual(notes[0]["status"], "unchanged")

    def test_absent_bottom_rule_preserves_original_candidate(self):
        groups, rules, words = fixture()
        rules = [r for r in rules if abs(r[1] - 708.432) > 1]
        repaired, notes = repair_word_groups(groups, rules, words)
        self.assertEqual(repaired, groups)
        self.assertEqual(notes[0]["status"], "unchanged")

    def test_conflicting_header_preserves_original_candidate(self):
        groups, rules, words = fixture()
        words = [w for w in words if not 244.783 < (w[1] + w[3]) / 2 < 256.303]
        words.append([60, 247.777, 280, 256.277])
        repaired, notes = repair_word_groups(groups, rules, words)
        self.assertEqual(repaired, groups)
        self.assertEqual(notes[0]["status"], "unchanged")


if __name__ == "__main__":
    unittest.main()
