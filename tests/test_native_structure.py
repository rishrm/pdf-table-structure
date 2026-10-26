"""Native shaded-cell partitions support completion without lexical rules."""
import unittest

from native_structure import page_grids
from table_structure import check_grid, structure_from_cells


def rectangle(x0, top, x1, bottom, color=(.8, .85, .9)):
    return dict(x0=x0, top=top, x1=x1, bottom=bottom, non_stroking_color=color, fill=True)


def word(x, top, width=20):
    # Content is deliberately inaccessible to the geometry helper.
    return dict(x0=x, top=top, x1=x + width, bottom=top + 8)


class Page:
    bbox = (0, 0, 600, 800)

    def __init__(self, rects, words, rules=()):
        self.rects = rects
        self.words = words
        self.chars = [object()] if words else []
        self.lines = [dict(x0=a, top=y, x1=b, bottom=y) for a, y, b in rules]

    def extract_words(self, **kwargs):
        return self.words


def simple_page():
    xs = [10, 80, 150, 220]
    rects = [rectangle(a, 50, b, 70) for a, b in zip(xs, xs[1:])]
    rects += [rectangle(a, 100, b, 130, (.95, .95, .95)) for a, b in zip(xs, xs[1:])]
    words = [word(x + 5, y) for y in (56, 76, 106, 136) for x in xs[:-1]]
    words += [word(85, 86), word(155, 86), word(155, 146)]
    return Page(rects, words, [(a, 160, b) for a, b in zip(xs, xs[1:])])


class NativeStructureTests(unittest.TestCase):
    def table(self, page):
        groups, diagnostics = page_grids(page)
        self.assertEqual(len(groups), 1)
        result = structure_from_cells(groups[0], "native")
        self.assertEqual(check_grid(result), [])
        self.assertTrue(diagnostics["returned_group_metadata"][0]["complete_envelope"])
        return result, diagnostics

    def test_alternating_shaded_rows_define_complete_table_and_wrapped_cells(self):
        table, details = self.table(simple_page())
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(table["bbox"], [10, 50, 220, 160])
        self.assertEqual(table["y_boundaries"], [50, 70, 100, 130, 160])
        self.assertTrue(details["returned_group_metadata"][0]["printed_bottom_rule"])

    def test_wrapped_leading_label_does_not_split_a_record(self):
        page = simple_page()
        page.words += [word(20, 86)]
        table, _ = self.table(page)
        self.assertEqual(table["row_count"], 4)

    def test_full_width_section_band_is_one_spanning_row(self):
        page = simple_page()
        page.rects += [rectangle(10, 130, 220, 144, (.95, .95, .95))]
        page.words = [w for w in page.words if w["top"] != 136]
        page.words += [word(15, 133, 75)] + [word(x, 149) for x in (15, 85, 155)]
        table, _ = self.table(page)
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertTrue(any(c["row"] == 3 and c["column_span"] == 3 for c in table["cells"]))

    def test_whole_row_background_does_not_merge_independent_data_cells(self):
        page = simple_page()
        page.rects = [r for r in page.rects if r["top"] != 100]
        page.rects.append(rectangle(10, 100, 220, 130, (.95, .95, .95)))
        table, _ = self.table(page)
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertTrue(all(c["column_span"] == 1 for c in table["cells"]))

    def test_two_level_header_preserves_group_span(self):
        page = simple_page()
        page.rects = [r for r in page.rects if r["top"] != 50]
        page.rects += [rectangle(10, 30, 80, 50), rectangle(80, 30, 220, 50)]
        page.rects += [rectangle(a, 50, b, 70) for a, b in zip([10, 80, 150], [80, 150, 220])]
        page.words += [word(100, 36, 50)]
        table, details = self.table(page)
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertTrue(any(c["row"] == 0 and c["column"] == 1 and c["column_span"] == 2 for c in table["cells"]))
        self.assertEqual(details["returned_group_metadata"][0]["header_bands"], 2)

    def test_two_stacked_tables_are_not_joined(self):
        first = simple_page()
        second = simple_page()
        for obj in second.rects + second.words + second.lines:
            obj["top"] += 200
            obj["bottom"] += 200
        first.rects += second.rects
        first.words += second.words
        first.lines += second.lines
        groups, _ = page_grids(first)
        self.assertEqual(len(groups), 2)
        self.assertEqual([structure_from_cells(g, "native")["bbox"] for g in groups],
                         [[10, 50, 220, 160], [10, 250, 220, 360]])

    def test_terminal_colored_band_needs_no_unexplained_text_below(self):
        page = simple_page()
        page.lines = []
        page.rects += [rectangle(a, 130, b, 160, (.95, .95, .95))
                       for a, b in zip([10, 80, 150], [80, 150, 220])]
        table, _ = self.table(page)
        self.assertEqual(table["bbox"], [10, 50, 220, 160])
        page.words += [word(15, 180, 150)]
        groups, _ = page_grids(page)
        self.assertEqual(groups, [])

    def test_single_column_callout_is_not_a_table(self):
        page = Page([rectangle(10, 50, 220, 70)], [word(15, y, 180) for y in (56, 76, 96)], [(10, 120, 220)])
        self.assertEqual(page_grids(page)[0], [])

    def test_body_prose_crossing_header_dividers_is_not_forced_into_cells(self):
        page = simple_page()
        page.words += [word(60, y, 50) for y in (76, 86, 106, 116, 136, 146)]
        self.assertEqual(page_grids(page)[0], [])

    def test_fully_ruled_body_is_left_to_vector_backend(self):
        page = simple_page()
        page.lines += [dict(x0=10, top=y, x1=220, bottom=y) for y in (90, 120)]
        self.assertEqual(page_grids(page)[0], [])

    def test_geometry_budget_and_missing_native_text_fail_closed(self):
        page = simple_page()
        page.chars = []
        self.assertEqual(page_grids(page)[1]["status"], "no_native_text")
        page.chars = [object()]
        page.rects = page.rects * 1000
        groups, diagnostics = page_grids(page)
        self.assertEqual(groups, [])
        self.assertEqual(diagnostics["status"], "budget_exceeded")

    def test_translation_preserves_topology(self):
        page = simple_page()
        for obj in page.rects + page.words + page.lines:
            obj["x0"] += 30
            obj["x1"] += 30
            obj["top"] += 40
            obj["bottom"] += 40
        table, _ = self.table(page)
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(table["bbox"], [40, 90, 250, 200])


if __name__ == "__main__":
    unittest.main()
