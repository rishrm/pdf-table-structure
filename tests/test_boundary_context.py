"""Word-gap normalization must not absorb neighboring page content."""
import copy
import unittest

from table_structure import boxes_in_table, check_grid, detect_page, space_text_boundaries, structure_from_cells
from test_detection_regressions import DetectionPage, detection_backends, rectangular_grid, words_in_grid


def fixture():
    cells = rectangular_grid(rows=4, columns=3)
    words = [[w[k] for k in ("x0", "top", "x1", "bottom")]
             for w in words_in_grid(cells)]
    return cells, words


class BoundaryContextTests(unittest.TestCase):
    def normalized(self, neighbors=(), extra_assigned=()):
        cells, words = fixture()
        table = structure_from_cells(cells, "img2table")
        words = words + list(extra_assigned) + list(neighbors)
        original_words = copy.deepcopy(words)
        assigned_before = boxes_in_table(words, table["bbox"])
        space_text_boundaries(table, words, [0, 0, 612, 792])
        self.assertEqual(words, original_words)
        self.assertEqual(check_grid(table), [])
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(boxes_in_table(words, table["bbox"]), assigned_before)
        # External words must not change the data-derived internal gutters.
        self.assertEqual(table["x_boundaries"][1:-1], [110, 180])
        return table, assigned_before

    def test_caption_stays_outside_while_unconflicted_edges_can_expand(self):
        table, _ = self.normalized(neighbors=[[65, 94, 95, 99]])
        self.assertEqual(table["bbox"], [56, 100, 274, 224])

    def test_footnote_stays_outside_while_other_edges_can_expand(self):
        table, _ = self.normalized(neighbors=[[65, 221, 95, 225]])
        self.assertEqual(table["bbox"], [56, 96, 274, 220])

    def test_left_and_right_neighbors_do_not_become_table_fields(self):
        for neighbor, expected in (
                ([55, 108, 61, 118], [60, 96, 274, 224]),
                ([270, 108, 276, 118], [56, 96, 270, 224])):
            with self.subTest(neighbor=neighbor):
                table, _ = self.normalized(neighbors=[neighbor])
                self.assertEqual(table["bbox"], expected)

    def test_diagonal_neighbor_cannot_enter_through_two_expanded_edges(self):
        table, _ = self.normalized(neighbors=[[56, 96, 59, 99]])
        self.assertEqual(table["bbox"], [60, 100, 274, 224])

    def test_padding_covers_already_assigned_clipped_words_when_unconflicted(self):
        clipped = [[58, 108, 66, 118], [268, 108, 272, 118],
                   [65, 98, 85, 106], [65, 216, 85, 222]]
        table, _ = self.normalized(extra_assigned=clipped)
        self.assertEqual(table["bbox"], [56, 96, 274, 224])
        for word in clipped:
            self.assertTrue(any(c["bbox"][0] <= word[0] and c["bbox"][1] <= word[1]
                                and c["bbox"][2] >= word[2] and c["bbox"][3] >= word[3]
                                for c in table["cells"]), word)

    def test_text_and_native_image_routes_pass_surrounding_caption_to_guard(self):
        cells, words = fixture()
        words.append([65, 94, 95, 99])
        native_words = [{**dict(zip(("x0", "top", "x1", "bottom"), w)), "text": "local-only"}
                        for w in words]
        for strategy in ("text", "img2table"):
            with self.subTest(strategy=strategy), detection_backends(img2table=[cells]):
                result = detect_page(DetectionPage(text=[cells], words=native_words), strategy)
            self.assertEqual(len(result["tables"]), 1)
            table = result["tables"][0]
            self.assertEqual(table["bbox"][1], 100)
            self.assertEqual(table["evidence"]["word_boxes"], 12)
            self.assertEqual(table["evidence"]["word_cut_fraction"], 0)

    def test_translated_region_preserves_context_guard_and_gutter_correction(self):
        cells, words = fixture()
        words.append([56, 96, 59, 99])
        move = lambda b: [b[0] + 100, b[1] - 50, b[2] + 100, b[3] - 50]
        table = structure_from_cells([move(c) for c in cells], "img2table")
        moved_words = [move(w) for w in words]
        before = boxes_in_table(moved_words, table["bbox"])
        space_text_boundaries(table, moved_words, [100, -50, 712, 742])
        self.assertEqual(table["bbox"], [160, 50, 374, 174])
        self.assertEqual(table["x_boundaries"][1:-1], [210, 280])
        self.assertEqual(boxes_in_table(moved_words, table["bbox"]), before)
        self.assertEqual(check_grid(table), [])


if __name__ == "__main__":
    unittest.main()
