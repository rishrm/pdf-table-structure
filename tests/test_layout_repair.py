"""Layout repairs must follow physical evidence, including negative controls."""
import unittest
from unittest.mock import patch

from layout_repair import repair_groups, repair_word_groups
from table_structure import check_grid, structure_from_cells


def glyphs(x, y, count=4):
    return [(x + 4 * i, y, x + 4 * i + 3, y + 6) for i in range(count)]


def grid(xs, ys):
    return [(a, c, b, d) for c, d in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]


def run_repair(groups, horizontal, vertical, components):
    def reader(area):
        return [b for b in components if area[0] <= (b[0] + b[2]) / 2 <= area[2]
                and area[1] <= (b[1] + b[3]) / 2 <= area[3]]
    result, notes = repair_groups(groups, horizontal, vertical, reader)
    return [structure_from_cells(group, "img2table") for group in result], notes


class LayoutRepairTests(unittest.TestCase):
    def wrapped_fixture(self):
        groups = [grid([10, 80, 150, 230], [40, 50, 60, 70]),
                  grid([10, 120, 230], [80, 90, 100, 110, 120, 130, 140, 150, 173])]
        components = [box for y in (48, 58) for x in (95, 175) for box in glyphs(x, y)]
        for y in (80, 100, 120, 140):
            components += glyphs(20, y) + glyphs(95, y) + glyphs(175, y)
            components += glyphs(95, y + 8) + glyphs(175, y + 8)
        components += glyphs(20, 165, 20)  # Footnote below the printed bottom rule.
        return groups, [[10, y, 230, y] for y in (40, 70, 160)], \
            [[x, 40, x, 72] for x in (80, 150)], components

    def test_header_body_join_restores_columns_and_keeps_continuations_in_record(self):
        tables, notes = run_repair(*self.wrapped_fixture())
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertEqual(table["x_boundaries"], [10, 80, 150, 230])
        self.assertEqual(table["y_boundaries"], [40, 70, 97, 117, 137, 160])
        self.assertEqual(check_grid(table), [])
        self.assertEqual(notes[0]["source_group_indices"], [0, 1])
        self.assertEqual(notes[0]["body_text_lines"], 8)
        self.assertEqual(notes[0]["body_rows"], 4)

    def test_caption_and_footnote_stay_outside_repaired_grid(self):
        args = list(self.wrapped_fixture())
        args[3] += glyphs(20, 28, 20)
        tables, _ = run_repair(*args)
        self.assertEqual(tables[0]["bbox"], [10, 40, 230, 160])

    def test_persistent_body_gutters_restore_subcolumns_and_grouped_headers(self):
        components = []
        for y in (110, 130, 150, 170):
            for x, count in ((15, 8), (95, 1), (145, 10), (235, 1), (275, 10)):
                components += glyphs(x, y, count)
        for x, count in ((120, 12), (245, 12)):
            components += glyphs(x, 58, count)
        for x, count in ((95, 1), (145, 10), (235, 1), (275, 10)):
            components += glyphs(x, 78, count)
        tables, notes = run_repair([grid([10, 200, 330], [55, 75, 95, 120, 140, 160, 178])],
            [[10, 50, 330, 50], [10, 90, 330, 90]], [], components)
        table = tables[0]
        self.assertEqual((table["row_count"], table["column_count"]), (6, 5))
        self.assertEqual([(c["row"], c["column"], c["column_span"]) for c in table["cells"]
                          if c["column_span"] > 1], [(0, 1, 2), (0, 3, 2)])
        self.assertEqual(check_grid(table), [])
        self.assertEqual(notes[0]["column_source"], "persistent_body_whitespace")

    def test_missing_rule_evidence_preserves_original_fragments(self):
        groups, _, vertical, components = self.wrapped_fixture()
        tables, notes = run_repair(groups, [], vertical, components)
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(n["status"] == "unchanged" for n in notes))

    def test_vertical_rules_continuing_into_body_block_speculative_rebuild(self):
        groups, horizontal, vertical, components = self.wrapped_fixture()
        vertical += [[80, 40, 80, 115]]
        tables, notes = run_repair(groups, horizontal, vertical, components)
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(n["status"] == "unchanged" for n in notes))

    def test_sparse_leading_labels_do_not_silently_collapse_unlabeled_records(self):
        groups, horizontal, vertical, components = self.wrapped_fixture()
        components = [b for b in components if not (b[0] < 80 and b[1] in (100, 120, 140))]
        tables, notes = run_repair(groups, horizontal, vertical, components)
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(n["status"] == "unchanged" for n in notes))

    def test_translation_does_not_encode_specific_page_coordinates(self):
        groups, horizontal, vertical, components = self.wrapped_fixture()
        dx, dy = 300, 200
        move = lambda b: (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        tables, _ = run_repair([[move(b) for b in group] for group in groups],
            [move(b) for b in horizontal], [move(b) for b in vertical], [move(b) for b in components])
        self.assertEqual(tables[0]["bbox"], [310, 240, 530, 360])
        self.assertEqual((tables[0]["row_count"], tables[0]["column_count"]), (5, 3))

    def test_pixel_adapter_preserves_candidates_if_repair_fails(self):
        from test_img2table_structure import ImageRegion, backend_available, pixel_grid
        if not backend_available():
            self.skipTest("Pinned image backend is unavailable")
        import img2table_structure as adapter
        with patch("layout_repair.repair_groups", side_effect=RuntimeError("private details")):
            groups, _, diagnostics = adapter.page_grids(ImageRegion(pixel_grid()))
        self.assertTrue(groups)
        self.assertEqual(diagnostics["status"], "ok")
        self.assertIn("img2table_layout_repair_failed:RuntimeError", diagnostics["warnings"])
        self.assertNotIn("private details", str(diagnostics))

    def word_fixture(self):
        groups = [grid([10, 90, 150, 210, 250], [40, 55, 70]),
                  grid([10, 130, 250], [80, 95, 115, 135, 168])]
        words = [(x, y, x + 20, y + 7) for y in (48, 58) for x in (10, 100, 160, 220)]
        for y in (80, 100, 120, 140):
            words += [(20, y, 75, y + 7), (30, y + 8, 70, y + 15)]
            words += [(x, y, x + 12, y + 7) for x in (100, 160, 220)]
        words += [(20, 174, 230, 181)]
        return groups, [[10, y, 250, y] for y in (40, 70, 170)], words

    def test_word_positions_keep_wrapped_labels_in_one_row_and_join_fragments(self):
        groups, notes = repair_word_groups(*self.word_fixture())
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 4))
        self.assertEqual(table["y_boundaries"], [40, 70, 97.5, 117.5, 137.5, 170])
        self.assertEqual(notes[0]["body_text_lines"], 8)
        self.assertEqual(notes[0]["body_rows"], 4)
        self.assertEqual(check_grid(table), [])

    def test_word_repair_needs_bottom_rule_and_matching_header_column_pattern(self):
        groups, rules, words = self.word_fixture()
        for case in ("no_bottom_rule", "prose_header"):
            with self.subTest(case=case):
                these_rules = rules[:-1] if case == "no_bottom_rule" else rules
                these_words = words if case == "no_bottom_rule" else \
                    [b for b in words if b[1] > 70] + [(20, 48, 230, 55)]
                repaired, notes = repair_word_groups(groups, these_rules, these_words)
                self.assertEqual(len(repaired), 2)
                self.assertTrue(all(n["status"] == "unchanged" for n in notes))


if __name__ == "__main__":
    unittest.main()
