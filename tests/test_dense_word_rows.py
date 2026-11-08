"""Logical records inside full-width body boxes need independent anchors."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from layout_repair import repair_dense_word_rows
from table_structure import check_grid, detect_page, structure_from_cells, text_evidence
from test_detection_regressions import DetectionPage


def fixture(wrapped=True):
    xs = [0, 60, 180, 210]
    cells = [(a, 0, b, 20) for a, b in zip(xs, xs[1:])] + [(0, 20, 210, 150)]
    words = [(a + 3, 4, a + 25, 12) for a in xs[:-1]]
    for y in (30, 60, 90, 120):
        words += [(5, y, 25, y + 8), (70, y, 100, y + 8),
                  (105, y, 145, y + 8), (190, y, 202, y + 8)]
        if wrapped:
            words.append((70, y + 10, 140, y + 18))
    return structure_from_cells(cells, 'lines'), words


class DenseWordRowTests(unittest.TestCase):
    def test_repeated_records_keep_wrapped_descriptions_and_empty_cells(self):
        table, words = fixture()
        original = copy.deepcopy((table, words))
        cells, note = repair_dense_word_rows(table, words)
        rebuilt = structure_from_cells(cells, 'lines')
        self.assertEqual((rebuilt['row_count'], rebuilt['column_count']), (5, 3))
        self.assertEqual(rebuilt['bbox'], table['bbox'])
        self.assertEqual(note['split_bands'][0]['body_records'], 4)
        self.assertEqual(check_grid(rebuilt), [])
        self.assertEqual(text_evidence(rebuilt, words, (0, 0, 400, 500))[0], [])
        self.assertEqual((table, words), original)
        for row in range(1, 5):
            description = next(c for c in rebuilt['cells'] if c['row'] == row and c['column'] == 1)
            self.assertGreater(description['bbox'][3] - description['bbox'][1], 18)

    def test_full_expense_geometry_recovers_records_and_preserves_header_spans(self):
        path = Path(__file__).parent/'fixtures/dense-expense-word-layout.json'
        data = json.loads(path.read_text())
        original = structure_from_cells(data['cells'], 'lines')
        self.assertEqual((original['row_count'], original['column_count']), (4, 7))
        for scale, dx, dy in ((1, 0, 0), (.8, 120, 50), (1.5, -20, 10)):
            with self.subTest(scale=scale):
                def move(b):
                    return [b[0]*scale+dx, b[1]*scale+dy, b[2]*scale+dx, b[3]*scale+dy]
                table = structure_from_cells([move(b) for b in data['cells']], 'lines')
                cells, note = repair_dense_word_rows(table, [move(b) for b in data['word_boxes']])
                rebuilt = structure_from_cells(cells, 'lines')
                self.assertEqual((rebuilt['row_count'], rebuilt['column_count']), (34, 7))
                self.assertEqual([b['body_records'] for b in note['split_bands']], [10, 22])
                self.assertEqual(rebuilt['bbox'], table['bbox'])
                headers = [c for c in table['cells'] if c['row'] < 2]
                for h in headers:
                    self.assertTrue(any(all(c[k] == h[k] for k in
                        ('bbox', 'row', 'column', 'row_span', 'column_span')) for c in rebuilt['cells']))
                self.assertEqual(check_grid(rebuilt), [])

    def test_missing_field_record_is_not_absorbed_into_neighbor(self):
        table, words = fixture()
        # Three complete records are insufficient; a fourth sparse record must
        # not be absorbed into a surrounding description.
        words = [w for w in words if not (w[0] == 190 and w[1] == 60)]
        self.assertIsNone(repair_dense_word_rows(table, words))

    def test_many_complete_records_still_do_not_hide_a_sparse_label_start(self):
        table, words = fixture()
        words += [(5, 52, 25, 60), (70, 52, 100, 60)]
        self.assertIsNone(repair_dense_word_rows(table, words))

    def test_wrapped_amount_or_unaligned_labels_keep_original_geometry(self):
        table, words = fixture(False)
        for case in ('multiword_amount', 'shift_labels', 'shift_amounts'):
            changed = list(words)
            if case == 'multiword_amount':
                changed += [(183, y, 187, y + 8) for y in (30, 60, 90, 120)]
            else:
                target = 5 if case == 'shift_labels' else 190
                changed = [(a + i*2, b, c + i*2, d) if a == target else (a, b, c, d)
                           for i, (a, b, c, d) in enumerate(changed)]
            with self.subTest(case=case):
                self.assertIsNone(repair_dense_word_rows(table, changed))

    def test_ordinary_cells_vertical_spans_and_headerless_boxes_are_not_split(self):
        table, words = fixture()
        for cells in (
            [(0, 0, 210, 150)],
            [(0, 0, 60, 150), (60, 0, 180, 20), (180, 0, 210, 20), (60, 20, 210, 150)],
            [(a, top, b, bottom) for a, b in zip([0, 60, 180], [60, 180, 210])
             for top, bottom in ((0, 20), (20, 150))]):
            with self.subTest(cells=cells):
                self.assertIsNone(repair_dense_word_rows(structure_from_cells(cells, 'lines'), words))

    def test_conflicting_column_words_cannot_create_new_column_cuts(self):
        table, words = fixture()
        words = [(a, b, 185, d) if a == 105 else (a, b, c, d) for a, b, c, d in words]
        self.assertIsNone(repair_dense_word_rows(table, words))

    def test_processing_budget_preserves_candidate(self):
        table, words = fixture()
        self.assertIsNone(repair_dense_word_rows(table, words * 1001))

    def test_ambiguous_header_columns_or_competing_patterns_are_not_reconstructed(self):
        table, words = fixture()
        # The left two header fields are one physical cell: their independent
        # positions are not established above the full-width body band.
        ambiguous = structure_from_cells([(0, 0, 180, 20), (180, 0, 210, 20),
                                         (0, 20, 210, 150)], 'lines')
        self.assertIsNone(repair_dense_word_rows(ambiguous, words))
        # Distinct record field patterns of equal size are not enough to pick
        # one grouping as the table's authoritative structure.
        xs = [0, 40, 80, 180, 210]
        original = structure_from_cells([(a, 0, b, 20) for a, b in zip(xs, xs[1:])] +
                                        [(0, 20, 210, 180)], 'lines')
        mixed = [(5, y, 20, y + 8) for y in (30, 50, 70, 90, 110, 130)]
        mixed += [(190, y, 202, y + 8) for y in (30, 50, 70, 90, 110, 130)]
        mixed += [(45 if i % 2 else 90, y, 65 if i % 2 else 150, y + 8)
                  for i, y in enumerate((30, 50, 70, 90, 110, 130))]
        self.assertIsNone(repair_dense_word_rows(original, mixed))

    def test_route_preserves_vector_table_on_repair_error(self):
        table, words = fixture()
        page = DetectionPage(vector=[[c['bbox'] for c in table['cells']]], words=[
            dict(zip(('x0', 'top', 'x1', 'bottom'), w), text='local') for w in words])
        with patch('layout_repair.repair_dense_word_rows', side_effect=RuntimeError('PRIVATE DATA')):
            result = detect_page(page, 'lines')
        self.assertEqual((result['tables'][0]['row_count'], result['tables'][0]['column_count']), (2, 3))
        self.assertIn('dense_word_repair_failed:RuntimeError', result['warnings'])
        self.assertNotIn('PRIVATE DATA', str(result))

    def test_failed_grid_validation_keeps_original_vector_candidate(self):
        table, words = fixture()
        page = DetectionPage(vector=[[c['bbox'] for c in table['cells']]], words=[
            dict(zip(('x0', 'top', 'x1', 'bottom'), w), text='local') for w in words])
        with patch('layout_repair.repair_dense_word_rows', return_value=(
                [(0, 0, 60, 20), (60, 0, 210, 20), (0, 20, 60, 150)], {'status': 'repaired'})):
            result = detect_page(page, 'lines')
        self.assertEqual(result['tables'][0]['row_count'], 2)
        self.assertNotIn('layout_repair', result['tables'][0])


if __name__ == '__main__':
    unittest.main()
