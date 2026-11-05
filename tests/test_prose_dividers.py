"""Reject invented prose dividers without penalizing real compact fields."""
import unittest

from table_structure import image_evidence, structure_from_cells, text_evidence


def grid(xs, ys):
    return [(left, top, right, bottom)
            for top, bottom in zip(ys, ys[1:])
            for left, right in zip(xs, xs[1:])]


def wide_grid(cut_rows):
    xs = [10 + 60 * col for col in range(7)]
    ys = [100 + 24 * row for row in range(5)]
    cells, words = grid(xs, ys), []
    for row in range(4):
        top = ys[row] + 6
        for col, left in enumerate(xs[:-1]):
            words.extend([(left + 5, top, left + 15, top + 8),
                          (left + 23, top, left + 33, top + 8)])
            words.append((left + 51, top, left + 67, top + 8)
                         if col == 0 and row in cut_rows else
                         (left + 39, top, left + 49, top + 8))
    return structure_from_cells(cells, "text"), words


def narrow_wordy_grid(aligned=False, header_offset=False):
    xs, ys = [10, 110, 220], [100, 124, 148, 172, 196]
    words = []
    for row in range(4):
        top = ys[row] + 6
        left_start = 20 if aligned else 20 + row * 6
        right_start = 111.5 if aligned else 111.5 + row * 3
        # The negative fixture must lack left, right AND center alignment in
        # at least one field. Fixed ends would instead describe legitimate
        # right-aligned values, not sufficient evidence of flowing prose.
        left_end = 109 if aligned else (109, 105, 101, 107)[row]
        if header_offset and row == 0:
            left_start, right_start = 40, 120.5
        words.extend([(left_start, top, left_start + 20, top + 10),
                      (65, top, 85, top + 10), (95, top, left_end, top + 10),
                      (right_start, top, right_start + 18, top + 10),
                      (150, top, 174, top + 10), (185, top, 210, top + 10)])
    return structure_from_cells(grid(xs, ys), "img2table"), words


def aligned_three_token_fields(alignments):
    """Vary field width while keeping a legitimate alignment convention.

    Three tokens can represent a numeric summary such as mean, +/- and SD.
    Their meaning is deliberately absent; the test supplies boxes only.
    """
    xs, ys = [10, 110, 220], [100, 124, 148, 172, 196]
    widths = [(98, 92, 86, 80), (107, 99, 91, 83)]
    words = []
    for row in range(4):
        for column, alignment in enumerate(alignments):
            width = widths[column][row]
            if alignment == "right":
                left = (109, 218.5)[column] - width
            elif alignment == "center":
                left = (60, 165)[column] - width / 2
            else:
                left = (11, 111.5)[column]
            top = ys[row] + 6
            words.extend([(left, top, left + .4 * width, top + 10),
                          (left + .4 * width + 2, top, left + .6 * width, top + 10),
                          (left + .6 * width + 2, top, left + width, top + 10)])
    return structure_from_cells(grid(xs, ys), "img2table"), words


def evaluated(table, words, rules=()):
    reasons, evidence = text_evidence(table, words, [0, 0, 612, 792])
    table["evidence"] = evidence
    image_reasons, _ = image_evidence(table, {}, list(rules))
    return reasons + image_reasons, evidence


class ProseDividerTests(unittest.TestCase):
    def test_wide_grid_cannot_dilute_a_repeatedly_cutting_divider(self):
        table, words = wide_grid({0, 1, 2, 3})
        reasons, evidence = evaluated(table, words)
        self.assertLess(evidence["word_cut_fraction"], .18)
        self.assertIn("column_divider_repeatedly_cuts_words", reasons)

    def test_one_isolated_word_crossing_is_not_a_repeated_divider_failure(self):
        table, words = wide_grid({2})
        reasons, _ = evaluated(table, words)
        self.assertEqual(reasons, [])

    def test_merged_header_text_can_cross_interior_logical_columns(self):
        xs, ys = [10, 70, 130, 190], [100, 124, 148, 172, 196]
        cells = [(10, 100, 190, 124)] + grid(xs, ys[1:])
        words = [(30, 106, 95, 114), (100, 106, 160, 114)]
        words += [(left + 5, top + 6, left + 25, top + 14)
                  for left, top, _, _ in cells[1:]]
        table = structure_from_cells(cells, "img2table")
        self.assertEqual(table["cells"][0]["column_span"], 3)
        reasons, evidence = evaluated(table, words)
        self.assertEqual(reasons, [])
        self.assertEqual(evidence["word_cut_fraction"], 0)

    def test_narrow_spaces_between_unaligned_wordy_cells_do_not_form_columns(self):
        for scale, dx, dy in ((1, 0, 0), (2, 31, 43)):
            with self.subTest(scale=scale):
                table, words = narrow_wordy_grid()
                if scale != 1:
                    move = lambda b: (b[0] * scale + dx, b[1] * scale + dy,
                                      b[2] * scale + dx, b[3] * scale + dy)
                    table = structure_from_cells([move(c["bbox"]) for c in table["cells"]], "img2table")
                    words = [move(w) for w in words]
                reasons, evidence = evaluated(table, words)
                self.assertEqual(evidence["word_cut_fraction"], 0)
                self.assertIn("unruled_columns_split_flowing_prose", reasons)

    def test_compact_numeric_fields_do_not_require_prose_sized_gutters(self):
        xs, ys = [10, 110, 220], [100, 124, 148, 172, 196]
        words = []
        for row in range(4):
            top = ys[row] + 6
            words.extend([(101 + row, top, 109, top + 10),
                          (111.5 + row * 3, top, 121.5 + row * 3, top + 10)])
        table = structure_from_cells(grid(xs, ys), "img2table")
        reasons, _ = evaluated(table, words)
        self.assertEqual(reasons, [])

    def test_consistently_aligned_multiword_fields_keep_narrow_gutters(self):
        table, words = narrow_wordy_grid(aligned=True)
        reasons, _ = evaluated(table, words)
        self.assertEqual(reasons, [])

    def test_right_aligned_three_token_values_keep_narrow_gutters(self):
        table, words = aligned_three_token_fields(("right", "right"))
        reasons, evidence = evaluated(table, words)
        self.assertEqual(evidence["word_cut_fraction"], 0)
        self.assertEqual(reasons, [])

    def test_centered_three_token_values_keep_narrow_gutters(self):
        table, words = aligned_three_token_fields(("center", "center"))
        reasons, evidence = evaluated(table, words)
        self.assertEqual(evidence["word_cut_fraction"], 0)
        self.assertEqual(reasons, [])

    def test_mixed_left_and_right_aligned_fields_keep_narrow_gutters(self):
        table, words = aligned_three_token_fields(("left", "right"))
        reasons, evidence = evaluated(table, words)
        self.assertEqual(evidence["word_cut_fraction"], 0)
        self.assertEqual(reasons, [])

    def test_three_aligned_records_with_differently_aligned_header_are_retained(self):
        table, words = narrow_wordy_grid(aligned=True, header_offset=True)
        reasons, _ = evaluated(table, words)
        self.assertEqual(reasons, [])

    def test_compatible_printed_rule_preserves_narrow_narrative_columns(self):
        table, words = narrow_wordy_grid()
        reasons, _ = evaluated(table, words, [[10, 100, 220, 100]])
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
