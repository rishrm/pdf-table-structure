"""Redundant background paint must not become extra table borders."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pdfplumber
from reportlab.pdfgen import canvas

from table_structure import check_grid, detect_page, vector_geometry_region
from test_detection_regressions import DetectionPage, detection_backends, rectangular_grid


def rect(bbox, color=(.8, 1, 1), stroke=False):
    return dict(zip(('x0', 'top', 'x1', 'bottom'), bbox),
                fill=True, stroke=stroke, non_stroking_color=color, object_type='rect')


class Region:
    def __init__(self, rects, **drawings):
        self.rects = rects
        for name in ('lines', 'curves', 'images'):
            setattr(self, name, drawings.get(name, []))

    def filter(self, predicate):
        return Region([r for r in self.rects if predicate(r)],
                      **{n: getattr(self, n) for n in ('lines', 'curves', 'images')})


class VectorRepaintTests(unittest.TestCase):
    def setUp(self):
        self.outer = rect((10, 10, 110, 60))
        self.inner = rect((20, 20, 100, 35))

    def test_same_color_contained_repaint_is_filtered_without_mutating_source(self):
        source = Region([self.outer, self.inner])
        view, notes = vector_geometry_region(source)
        self.assertEqual(view.rects, [self.outer])
        self.assertEqual(source.rects, [self.outer, self.inner])
        self.assertEqual(notes['ignored_repaint_rectangles'], 1)

    def test_strokes_thin_rules_different_colors_and_partial_overlaps_are_retained(self):
        cases = [rect((20, 20, 100, 35), stroke=True),
                 rect((20, 20, 100, 20.5)),
                 rect((20, 20, 100, 35), color=(1, 0, 0)),
                 rect((100, 20, 130, 35))]
        for inner in cases:
            with self.subTest(inner=inner):
                source = Region([self.outer, inner])
                view, notes = vector_geometry_region(source)
                self.assertIs(view, source)
                self.assertEqual(notes['ignored_repaint_rectangles'], 0)

    def test_intervening_drawing_protects_a_fill_that_may_cover_it(self):
        drawing = dict(zip(('x0', 'top', 'x1', 'bottom'), (25, 25, 90, 25)))
        for kind in ('lines', 'curves', 'images'):
            with self.subTest(kind=kind):
                source = Region([self.outer, self.inner], **{kind: [drawing]})
                view, _ = vector_geometry_region(source)
                self.assertIs(view, source)

    def test_border_touching_only_the_edge_does_not_count_as_interior_drawing(self):
        line = dict(zip(('x0', 'top', 'x1', 'bottom'), (20, 20, 100, 20)))
        view, _ = vector_geometry_region(Region([self.outer, self.inner], lines=[line]))
        self.assertEqual(view.rects, [self.outer])

    def test_another_color_inside_repaint_prevents_removal(self):
        source = Region([self.outer, self.inner, rect((30, 22, 60, 30), color=0)])
        view, _ = vector_geometry_region(source)
        self.assertIs(view, source)

    def test_nested_chain_retains_largest_fill_and_unknown_colors_are_untouched(self):
        smallest = rect((30, 23, 60, 30))
        view, notes = vector_geometry_region(Region([self.outer, self.inner, smallest]))
        self.assertEqual(view.rects, [self.outer])
        self.assertEqual(notes['ignored_repaint_rectangles'], 2)
        for color in (None, ('pattern',), (float('nan'),), (1, 2)):
            with self.subTest(color=color):
                source = Region([rect((10, 10, 110, 60), color), rect((20, 20, 100, 35), color)])
                self.assertIs(vector_geometry_region(source)[0], source)

    def test_identical_rectangles_are_not_both_removed(self):
        source = Region([self.outer, dict(self.outer)])
        self.assertIs(vector_geometry_region(source)[0], source)

    def test_budgets_and_absent_geometry_preserve_original_view(self):
        for source in (Region([self.outer] * 1001),
                       Region([self.outer, self.inner], lines=[self.outer] * 6001)):
            with self.subTest(count=len(source.rects)):
                view, notes = vector_geometry_region(source)
                self.assertIs(view, source)
                self.assertEqual(notes['status'], 'budget_exceeded')
        page = DetectionPage()
        self.assertIs(vector_geometry_region(page)[0], page)

    def test_cleanup_failure_preserves_vector_detection_and_sanitizes_error(self):
        cells = rectangular_grid(rows=3, columns=3)
        with patch('table_structure.vector_geometry_region', side_effect=ValueError('PRIVATE VALUE')):
            result = detect_page(DetectionPage(vector=[cells]), 'lines')
        self.assertEqual(len(result['tables']), 1)
        self.assertIn('vector_fill_cleanup_failed:ValueError', result['warnings'])
        self.assertNotIn('PRIVATE VALUE', str(result))

    def test_repeated_background_paints_do_not_split_real_ruled_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'paint.pdf'
            doc = canvas.Canvas(str(path), pagesize=(400, 300))
            for x in (40, 160):
                for y in (80, 140, 200):
                    doc.setFillColorRGB(.8, 1, 1)
                    doc.rect(x, y, 120, 60, stroke=0, fill=1)
                    # A same-color repaint touches the row borders; treating
                    # its invisible sides as rulings invents extra columns.
                    doc.rect(x + 7, y, 106, 60, stroke=0, fill=1)
                    for offset in (7, 23, 39):
                        doc.rect(x + 7, y + offset, 106, 12, stroke=0, fill=1)
            doc.setStrokeColorRGB(0, 0, 0)
            for x in (40, 160, 280):
                doc.line(x, 80, x, 260)
            for y in (80, 140, 200, 260):
                doc.line(40, y, 280, y)
            doc.save()
            with pdfplumber.open(path) as pdf:
                self.assertTrue(any(len(t.columns) > 2 for t in pdf.pages[0].find_tables()))
                result = detect_page(pdf.pages[0], 'lines')
            self.assertEqual(result['vector_fill_cleanup']['ignored_repaint_rectangles'], 24)
            self.assertEqual(len(result['tables']), 1)
            table = result['tables'][0]
            self.assertEqual((table['row_count'], table['column_count']), (3, 2))
            self.assertEqual(check_grid(table), [])


if __name__ == '__main__':
    unittest.main()
