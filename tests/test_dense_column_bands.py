"""Separate dense records while preserving headers, blanks and true footers."""
import copy
import json
from pathlib import Path
import unittest

from layout_repair import repair_dense_word_rows
from table_structure import check_grid, structure_from_cells, text_evidence


def synthetic(bands=3):
    xs = list(range(0, 161, 20))
    ys = [0, 20, 40] + [40 + i * 40 for i in range(1, bands + 1)]
    end = ys[-1]
    ys += [end + 14, end + 24, end + 60]
    cells = [(0, 0, 160, 20)]
    cells += [(a, top, b, bottom) for top, bottom in zip(ys[1:-3], ys[2:-2])
              for a, b in zip(xs, xs[1:])]
    cells += [(0, end + 14, 160, end + 24), (0, end + 24, 160, end + 60)]
    words = [(50, 5, 100, 13)] + [(a + 3, 24, a + 17, 32) for a in xs[:-1]]
    for band in range(bands):
        for top in (42 + band * 40, 54 + band * 40, 66 + band * 40):
            words += [(a + 7, top, a + 17, top + 8) for col, a in enumerate(xs[:-1]) if col not in (5, 6)]
    words += [(a + 3, end + 2, a + 17, end + 10) for a in xs[:-1]]
    words += [(a, top, b, top + 8) for top in (end + 30, end + 42)
              for a, b in ((1, 45), (47, 90), (93, 145))]
    return structure_from_cells(cells, 'lines'), words


def rebuild(table, words, rules=()):
    proposal = repair_dense_word_rows(table, words, rules)
    if proposal is None:
        return None, None
    cells, note = proposal
    result = structure_from_cells(cells, 'lines')
    assert check_grid(result) == []
    return result, note


class DenseColumnBandTests(unittest.TestCase):
    def test_three_bands_restore_records_and_keep_empty_columns(self):
        table, words = synthetic()
        original = copy.deepcopy((table, words))
        result, note = rebuild(table, words)
        self.assertEqual((result['row_count'], result['column_count']), (12, 8))
        self.assertEqual(result['bbox'], [0, 0, 160, 174])
        self.assertEqual([b['body_records'] for b in note['split_bands']], [3, 3, 3])
        self.assertEqual(result['x_boundaries'], table['x_boundaries'])
        self.assertEqual((table, words), original)
        for row in range(2, 11):
            for col in (5, 6):
                self.assertTrue(any(c['row'] == row and c['column'] == col and
                                    c['column_span'] == c['row_span'] == 1 for c in result['cells']))

    def test_real_grid_recovers_rows_header_tier_and_note_boundary(self):
        data = json.loads((Path(__file__).parent/'fixtures/dense-bands-word-layout.json').read_text())
        for scale, dx, dy in ((1, 0, 0), (.8, 20, -10), (1.3, -40, 25)):
            with self.subTest(scale=scale):
                def move(b):
                    return [b[0]*scale+dx, b[1]*scale+dy, b[2]*scale+dx, b[3]*scale+dy]
                old = structure_from_cells([move(b) for b in data['cells']], 'lines')
                words = [move(b) for b in data['word_boxes']]
                result, note = rebuild(old, words, [move(b) for b in data['vertical_rules']])
                self.assertEqual((result['row_count'], result['column_count']), (59, 25))
                self.assertEqual(result['x_boundaries'], old['x_boundaries'])
                self.assertAlmostEqual(result['bbox'][3], 482.75*scale+dy, delta=.002)
                self.assertAlmostEqual(note['header_tier_divider'], 70.34*scale+dy, delta=.002)
                self.assertEqual(text_evidence(result, words, move(data['page_bbox']))[0], [])
                for col, span in ((7, 3), (10, 3), (13, 3), (16, 2), (18, 3), (21, 3)):
                    self.assertTrue(any(c['row'] == 1 and c['column'] == col and
                                        c['column_span'] == span for c in result['cells']))
                for row in range(3, 58):
                    for col in (16, 17):
                        self.assertTrue(any(c['row'] == row and c['column'] == col and
                                            c['row_span'] == c['column_span'] == 1 for c in result['cells']))

    def test_header_tier_requires_all_physical_lower_dividers(self):
        data = json.loads((Path(__file__).parent/'fixtures/dense-bands-word-layout.json').read_text())
        old = structure_from_cells(data['cells'], 'lines')
        for rules in ([], [r for r in data['vertical_rules'] if abs(r[0] - 330.31) > 1]):
            with self.subTest(rule_count=len(rules)):
                result, note = rebuild(old, data['word_boxes'], rules)
                self.assertEqual(result['row_count'], 58)
                self.assertNotIn('header_tier_divider', note)

    def test_header_words_crossing_the_proposed_tier_prevent_header_repair(self):
        data = json.loads((Path(__file__).parent/'fixtures/dense-bands-word-layout.json').read_text())
        old = structure_from_cells(data['cells'], 'lines')
        words = data['word_boxes'] + [[340, 68, 360, 74]]
        result, note = rebuild(old, words, data['vertical_rules'])
        self.assertEqual(result['row_count'], 58)
        self.assertNotIn('header_tier_divider', note)

    def test_less_than_three_repeated_bands_is_not_enough(self):
        table, words = synthetic(bands=2)
        self.assertIsNone(repair_dense_word_rows(table, words))

    def test_sparse_record_or_nonuniform_spacing_preserves_ambiguous_band(self):
        table, words = synthetic()
        for case in ('missing_field', 'different_pitch'):
            if case == 'missing_field':
                changed = [w for w in words if not (w[0] == 7 and w[1] == 94)]
            else:
                changed = [(a, b + 4, c, d + 4) if b == 94 else (a, b, c, d) for a, b, c, d in words]
            with self.subTest(case=case):
                self.assertIsNone(repair_dense_word_rows(table, changed))

    def test_wrapped_prose_and_unaligned_compact_fields_do_not_supply_dense_records(self):
        table, words = synthetic()
        extra = [(a - 5, b, a - 1, d) for a, b, c, d in words if 40 <= b < 160]
        self.assertIsNone(repair_dense_word_rows(table, words + extra))
        shifted = [(a + (b % 7), b, c + (b % 7), d) if 40 <= b < 160 else (a, b, c, d)
                   for a, b, c, d in words]
        self.assertIsNone(repair_dense_word_rows(table, shifted))

    def test_nonempty_spacer_retains_following_notes(self):
        table, words = synthetic()
        result, note = rebuild(table, words + [(4, 176, 30, 182)])
        self.assertEqual(result['bbox'], table['bbox'])
        self.assertNotIn('excluded_terminal_note_bbox', note)

    def test_missing_total_record_does_not_trim_terminal_prose(self):
        table, words = synthetic()
        words = [w for w in words if not 160 <= w[1] < 174]
        result, note = rebuild(table, words)
        self.assertEqual(result['bbox'], table['bbox'])
        self.assertNotIn('excluded_terminal_note_bbox', note)

    def test_tabular_trailing_summary_is_not_treated_as_footnote(self):
        table, words = synthetic()
        words = [w for w in words if w[1] < 184]
        words += [(a + 5, y, a + 15, y + 8) for y in (190, 202) for a in range(0, 160, 20)]
        result, note = rebuild(table, words)
        self.assertEqual(result['bbox'], table['bbox'])
        self.assertNotIn('excluded_terminal_note_bbox', note)

    def test_later_separate_column_cells_prevent_note_trimming(self):
        table, words = synthetic()
        cells = [c['bbox'] for c in table['cells'] if c['bbox'][1] < 184]
        cells.extend((a, 184, a + 20, 220) for a in range(0, 160, 20))
        table = structure_from_cells(cells, 'lines')
        result, note = rebuild(table, words)
        self.assertEqual(result['bbox'], table['bbox'])
        self.assertNotIn('excluded_terminal_note_bbox', note)

    def test_word_crossings_keep_existing_column_cells(self):
        table, words = synthetic()
        words = [(a, b, c + 8, d) if 40 <= b < 160 else (a, b, c, d) for a, b, c, d in words]
        self.assertIsNone(repair_dense_word_rows(table, words))


if __name__ == '__main__':
    unittest.main()
