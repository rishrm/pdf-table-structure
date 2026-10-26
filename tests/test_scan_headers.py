"""A centered summary inside printed rules is different from a caption."""
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from PIL import Image, ImageDraw, ImageFont

from scan_headers import repair_headers
from table_structure import check_grid, structure_from_cells
from test_img2table_structure import backend_available


def glyphs(x, y, count=4, height=6):
    return [(x + 4 * i, y, x + 4 * i + 3, y + height) for i in range(count)]


def fixture(columns=2):
    xs = [20, 180, 210] if columns == 2 else [20, 90, 150, 210]
    ys = [120, 140, 160, 180, 200]
    cells = [(a, c, b, d) for c, d in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]
    header = glyphs(104, 108, 8) if columns == 2 else glyphs(80, 108, 20)
    body = []
    for y in (124, 144, 164, 184):
        for x in ([25, 191] if columns == 2 else [25, 105, 185]):
            body += glyphs(x, y)
    return cells, [[20, 100, 220, 100], [20, 204, 220, 204]], header + body


def run(cells, rules, components):
    return repair_headers([cells], rules, lambda _area: components)


class ScanHeaderGeometryTests(unittest.TestCase):
    def test_centered_header_can_be_inside_wide_first_body_column(self):
        cells, rules, components = fixture()
        groups, notes = run(cells, rules, components)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 2))
        self.assertEqual(table["cells"][0]["column_span"], 2)
        self.assertEqual(table["bbox"], [20, 100, 220, 204])
        self.assertEqual(check_grid(table), [])
        self.assertEqual(notes[0]["header_rows_added"], 1)
        self.assertFalse(notes[0]["semantic_structure_verified"])
        self.assertNotIn("source_group_indices", notes[0])

    def test_three_column_header_and_existing_body_span_are_preserved(self):
        cells, rules, components = fixture(3)
        cells = [b for b in cells if b not in ((20, 120, 90, 140), (20, 140, 90, 160))]
        cells.append((20, 120, 90, 160))
        groups, notes = run(cells, rules, components)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertEqual(table["cells"][0]["column_span"], 3)
        span = next(c for c in table["cells"] if c["row"] == 1 and c["column"] == 0)
        self.assertEqual(span["row_span"], 2)
        self.assertEqual(check_grid(table), [])
        self.assertEqual(notes[0]["body_rows"], 4)

    def test_caption_above_top_rule_cannot_supply_header_evidence(self):
        cells, rules, components = fixture()
        components = [b for b in components if b[1] >= 120] + glyphs(104, 88, 8)
        groups, notes = run(cells, rules, components)
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["status"], "unchanged")

    def test_existing_wrapped_or_grouped_header_is_not_prepended_again(self):
        cells, rules, components = fixture()
        cells += [(20, 100, 210, 120)]
        components += glyphs(104, 102, 8)
        reader = Mock(return_value=components)
        groups, notes = repair_headers([cells], rules, reader)
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["status"], "unchanged")
        reader.assert_not_called()

    def test_short_gap_above_candidate_does_not_mean_missing_header(self):
        cells, _, components = fixture()
        rules = [[20, 117, 220, 117], [20, 204, 220, 204]]
        reader = Mock(return_value=components)
        groups, notes = repair_headers([cells], rules, reader)
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["reason"], "no_compatible_rule_bracket")
        reader.assert_not_called()

    def test_unrelated_left_aligned_or_separate_labels_do_not_form_summary(self):
        cells, rules, components = fixture()
        body = [b for b in components if b[1] >= 120]
        headers = [glyphs(25, 108, 10), glyphs(40, 108, 8) + glyphs(170, 108, 8)]
        for header in headers:
            with self.subTest(header=header):
                groups, notes = run(cells, rules, body + header)
                self.assertEqual(groups, [cells])
                self.assertEqual(notes[0]["status"], "unchanged")

    def test_multiple_header_lines_and_intervening_rule_remain_ambiguous(self):
        cells, rules, components = fixture()
        for kind in ("two_lines", "intervening_rule"):
            with self.subTest(kind=kind):
                local_rules = rules + ([[20, 118, 220, 118]] if kind == "intervening_rule" else [])
                local_ink = components + (glyphs(104, 101, 8) if kind == "two_lines" else [])
                groups, notes = run(cells, local_rules, local_ink)
                self.assertEqual(groups, [cells])
                self.assertEqual(notes[0]["status"], "unchanged")

    def test_missing_bottom_rule_or_unpopulated_body_column_blocks_repair(self):
        cells, rules, components = fixture()
        for kind in ("no_bottom", "sparse_column"):
            with self.subTest(kind=kind):
                ink = components if kind == "no_bottom" else [b for b in components if not (b[0] > 180 and b[1] > 124)]
                groups, notes = run(cells, rules[:1] if kind == "no_bottom" else rules, ink)
                self.assertEqual(groups, [cells])
                self.assertEqual(notes[0]["status"], "unchanged")

    def test_translation_preserves_shape_and_diagnostics_export_no_text(self):
        cells, rules, components = fixture(3)
        shifted = lambda b: (b[0] + 100, b[1] + 200, b[2] + 100, b[3] + 200)
        groups, notes = run([shifted(b) for b in cells], [shifted(b) for b in rules], [shifted(b) for b in components])
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual(table["bbox"], [120, 300, 320, 404])
        self.assertEqual((table["row_count"], table["column_count"]), (5, 3))
        self.assertNotIn("text", notes[0])

    def test_reader_failures_preserve_groups_and_sanitize_details(self):
        cells, rules, _ = fixture()
        groups, notes = repair_headers([cells], rules, Mock(side_effect=RuntimeError("PRIVATE")))
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["reason"], "header_components_failed:RuntimeError")
        self.assertNotIn("PRIVATE", str(notes))

    def test_outer_bottom_expansion_cannot_absorb_an_additional_record(self):
        cells, rules, components = fixture()
        rules[-1] = [20, 212, 220, 212]
        components += glyphs(25, 204) + glyphs(191, 204)
        groups, notes = run(cells, rules, components)
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["reason"], "additional_body_ink_below_grid")

    def test_outer_right_expansion_cannot_absorb_an_additional_field(self):
        cells, rules, components = fixture()
        components += glyphs(214, 124, 1)
        groups, notes = run(cells, rules, components)
        self.assertEqual(groups, [cells])
        self.assertEqual(notes[0]["reason"], "additional_body_ink_beside_grid")

    def test_small_component_rounding_does_not_block_blank_padding_expansion(self):
        cells, rules, components = fixture()
        # A sub-point boundary discrepancy can result from pixel-to-point
        # rounding. It must not look like a separate row or column.
        components += [(208.5, 124, 210.5, 130), (25, 198.5, 28, 200.5)]
        groups, notes = run(cells, rules, components)
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 2))
        self.assertEqual(notes[0]["status"], "repaired")


@unittest.skipUnless(backend_available(), "Pinned image backend is unavailable")
class ScanHeaderPixelTests(unittest.TestCase):
    def test_real_components_restore_centered_summary_without_recognizing_text(self):
        from img2table_structure import _load_backend, _prepare_pixel_evidence, _layout_components
        np, _, cv2 = _load_backend()
        image = Image.new("RGB", (480, 480), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=18)
        cells, rules, _ = fixture()
        for a, top, b, _ in rules:
            draw.line((2*a, 2*top, 2*b, 2*top), fill="black", width=2)
        text = "12 records"
        length = draw.textbbox((0, 0), text, font=font)[2]
        draw.text((240 - length/2, 210), text, font=font, fill="black")
        for y in (124, 144, 164, 184):
            draw.text((50, 2*y), "Body label", font=font, fill="black")
            draw.text((390, 2*y), "8", font=font, fill="black")
        extractor = SimpleNamespace(horizontal_lines=[SimpleNamespace(x1=a*2,y1=b*2,x2=c*2,y2=d*2,thickness=2)
            for a,b,c,d in rules], vertical_lines=[])
        masks = _prepare_pixel_evidence(np.asarray(image), extractor, 2, 2, np, cv2)
        masks["layout_budget"] = 4_000_000
        groups, notes = repair_headers([cells], rules,
            lambda area: _layout_components(area, masks, (0,0,240,240), 2, 2, cv2))
        table = structure_from_cells(groups[0], "img2table")
        self.assertEqual((table["row_count"], table["column_count"]), (5, 2))
        self.assertEqual(table["cells"][0]["column_span"], 2)
        self.assertEqual(notes[0]["status"], "repaired")


if __name__ == "__main__":
    unittest.main()
