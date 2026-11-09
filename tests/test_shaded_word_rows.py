"""Paint spacers and footnotes must not displace real empty data fields."""
import copy
import json
from pathlib import Path
import unittest

from layout_repair import repair_shaded_word_rows
from table_structure import check_grid, structure_from_cells, text_evidence


def fixture():
    data = json.loads((Path(__file__).parent/'fixtures/shaded-word-layout.json').read_text())
    return structure_from_cells(data['cells'], 'lines'), data['word_boxes'], data['rectangles']


class ShadedWordRowsTests(unittest.TestCase):
    def test_real_header_rows_and_empty_fields_are_preserved(self):
        table, words, rects = fixture()
        original = copy.deepcopy((table, words, rects))
        cells, note = repair_shaded_word_rows(table, words, rects)
        new = structure_from_cells(cells, 'lines')
        self.assertEqual((new['row_count'], new['column_count']), (40, 5))
        self.assertEqual(note['section_rows'], [2, 6, 10, 33, 36])
        self.assertEqual(note['body_records'], 33)
        self.assertEqual(check_grid(new), [])
        self.assertEqual(text_evidence(new, words, [0, 0, 612, 792])[0], [])
        self.assertEqual((table, words, rects), original)
        for row in [2, 6, 10, 33, 36]:
            self.assertTrue(any(c['row'] == row and c['column_span'] == 5 for c in new['cells']))
        for row in set(range(2, 40)) - {2, 6, 10, 33, 36}:
            for col in (3, 4):
                cell = next(c for c in new['cells'] if c['row'] == row and c['column'] == col)
                b = cell['bbox']
                self.assertFalse(any(b[0] <= (w[0]+w[2])/2 < b[2] and
                                     b[1] <= (w[1]+w[3])/2 < b[3] for w in words))
        self.assertTrue(any(c['row'] == 0 and c['column'] == 0 and c['row_span'] == 2 for c in new['cells']))
        self.assertEqual(new['bbox'][3], 531.917)

    def test_translation_and_scale_do_not_encode_document_coordinates(self):
        table, words, rects = fixture()
        for scale, dx, dy in ((.8, 17, 25), (1.2, -10, 60)):
            with self.subTest(scale=scale):
                def move(b):return [b[0]*scale+dx,b[1]*scale+dy,b[2]*scale+dx,b[3]*scale+dy]
                transformed = structure_from_cells([move(c['bbox']) for c in table['cells']], 'lines')
                painted = [{**r, **dict(zip(('x0','top','x1','bottom'),
                    move([r[k] for k in ('x0','top','x1','bottom')])))} for r in rects]
                cells, _ = repair_shaded_word_rows(transformed, [move(w) for w in words], painted)
                new = structure_from_cells(cells, 'lines')
                self.assertEqual((new['row_count'],new['column_count']), (40,5))

    def test_labeled_or_populated_narrow_field_is_never_removed(self):
        table, words, rects = fixture()
        for y in (86, 110):
            with self.subTest(y=y):
                self.assertIsNone(repair_shaded_word_rows(table, words + [[468,y,472,y+7]], rects))

    def test_real_strokes_and_different_header_pattern_prevent_rebuild(self):
        table, words, rects = fixture()
        stroked = copy.deepcopy(rects)
        next(r for r in stroked if r['x0'] > 390 and r['bottom']-r['top'] > 3)['stroke'] = True
        self.assertIsNone(repair_shaded_word_rows(table, words, stroked))
        self.assertIsNone(repair_shaded_word_rows(table, words, rects,
            [{'x0':467,'top':90,'x1':467,'bottom':520}]))
        changed = copy.deepcopy(table)
        changed['cells'][0]['row_span'] = 1
        self.assertIsNone(repair_shaded_word_rows(changed, words, rects))

    def test_missing_section_or_missing_data_field_keeps_original_grid(self):
        table, words, rects = fixture()
        fewer_sections = [r for r in rects if not 140 < r['top'] < 494]
        self.assertIsNone(repair_shaded_word_rows(table, words, fewer_sections))
        sparse = [w for w in words if not (433 < (w[0]+w[2])/2 < 467 and 107 < w[1] < 117)]
        self.assertIsNone(repair_shaded_word_rows(table, sparse, rects))

    def test_footer_needs_printed_boundary_and_distinct_small_prose(self):
        table, words, rects = fixture()
        no_boundary = [r for r in rects if not (0 < r['bottom']-r['top'] <= 1 and
                       abs((r['top']+r['bottom'])/2-531.917) <= 1)]
        self.assertIsNone(repair_shaded_word_rows(table, words, no_boundary))
        same_fill = copy.deepcopy(rects)
        for r in same_fill:
            if r['top'] > 531 and r['bottom']-r['top'] > 2:
                r['non_stroking_color'] = [.72157,.8,.89412]
        self.assertIsNone(repair_shaded_word_rows(table, words, same_fill))

    def test_terminal_numeric_summary_is_not_treated_as_a_footnote(self):
        table, words, rects = fixture()
        words = [w for w in words if w[1] < 531.917]
        words += [[x,535,x+10,542.5] for x in (80,410,445,480,515)]
        self.assertIsNone(repair_shaded_word_rows(table, words, rects))

    def test_budget_failure_preserves_geometry(self):
        table, words, rects = fixture()
        self.assertIsNone(repair_shaded_word_rows(table, words, rects * 3))
        self.assertIsNone(repair_shaded_word_rows(table, words * 100, rects))


if __name__ == '__main__':unittest.main()
