"""Rule-seeded scan recovery must require supported columns and body records."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

from scan_recovery import (_normalize_wrapped_header, _rule_envelopes,
                           recover_ruled_groups, MAX_RULES)
from table_structure import check_grid, image_evidence, structure_from_cells
from test_img2table_structure import ImageRegion, backend_available


def scan_fixture(prose=False, blank=False):
    # Pixel size is twice the physical PDF-point size, independent of DPI hints.
    image = Image.new("RGB", (1200, 640), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    rules = [[20, y, 560, y] for y in (30, 67, 285)]
    for left, y, right, _ in rules:
        draw.line((2 * left, 2 * y, 2 * right, 2 * y), fill="black", width=2)
    if not blank:
        for x, title in ((25, "Category"), (140, "First group"), (245, "Second group"),
                         (350, "Third group"), (455, "Last group")):
            draw.text((2 * x, 2 * 36), title, font=font, fill="black")
        draw.text((2 * 245, 2 * 48), "continued", font=font, fill="black")
        for row, y in enumerate(range(82, 263, 30)):
            if prose:
                # Filled lines crossing all possible gutters do not form columns.
                draw.text((50, 2 * y), "The paragraph continues with ordinary words across this space.",
                          font=font, fill="black")
            else:
                draw.text((50, 2 * y), f"Row label {row + 1}", font=font, fill="black")
                for index, x in enumerate((140, 245, 350, 455)):
                    draw.text((2 * x, 2 * y), str(10 + row + index), font=font, fill="black")
    return ImageRegion(image, bounds=(0, 0, 600, 320)), rules


class RuleEnvelopeTests(unittest.TestCase):
    def test_requires_three_compatible_printed_boundaries(self):
        rules = [[20, 30, 560, 30], [21, 67, 560, 67], [20, 285, 559, 285]]
        envelopes = _rule_envelopes(rules, (0, 0, 600, 320))
        self.assertEqual(len(envelopes), 1)
        self.assertEqual(envelopes[0]["bbox"], [21, 30, 559, 285])
        self.assertEqual(_rule_envelopes(rules[:2], (0, 0, 600, 320)), [])
        self.assertEqual(_rule_envelopes([rules[0], [100, 67, 560, 67], rules[2]], (0, 0, 600, 320)), [])

    def test_invalid_rules_and_unbounded_iterators_do_not_trigger_render(self):
        region = SimpleNamespace(bbox=(0, 0, 600, 800))
        with patch("img2table_structure._render") as render:
            groups, diagnostics = recover_ruled_groups(region,
                iter([[float("nan"), 40, 500, 40]] * (MAX_RULES + 1000)))
        self.assertEqual(groups, [])
        self.assertTrue(diagnostics["input_rules_truncated"])
        render.assert_not_called()

    def test_header_normalization_does_not_erase_grouped_header_spans(self):
        cells = [(20, 30, 230, 45), (230, 30, 440, 45),
                 (20, 45, 125, 65), (125, 45, 230, 65),
                 (230, 45, 335, 65), (335, 45, 440, 65)]
        for y in (65, 85, 105):
            cells += [(x, y, x + 105, y + 20) for x in (20, 125, 230, 335)]
        note = {"header_rows": 2, "body_rows": 3, "columns": 4, "bbox": [20, 30, 440, 125]}
        components = [(110 + 4 * i, 34, 113 + 4 * i, 40) for i in range(8)]
        components += [(320 + 4 * i, 34, 323 + 4 * i, 40) for i in range(8)]
        components += [(x + 4 * i, 53, x + 3 + 4 * i, 59) for x in (40, 145, 250, 355) for i in range(5)]
        components += [(x + 4 * i, y, x + 3 + 4 * i, y + 6) for y in (70, 90, 110)
                       for x in (40, 145, 250, 355) for i in range(5)]
        result, metadata = _normalize_wrapped_header(cells, note, components,
            [[20, 30, 440, 30], [20, 65, 440, 65], [20, 125, 440, 125]])
        self.assertEqual(result, cells)
        self.assertEqual(metadata["header_rows"], 2)

    def test_dependency_errors_are_sanitized(self):
        region = SimpleNamespace(bbox=(0, 0, 600, 320))
        rules = [[20, y, 560, y] for y in (30, 67, 285)]
        with patch("img2table_structure._load_backend", side_effect=ImportError("PRIVATE document details")):
            groups, details = recover_ruled_groups(region, rules)
        self.assertEqual(groups, [])
        self.assertEqual(details["status"], "failed")
        self.assertIn("scan_recovery_failed:ImportError", details["warnings"])
        self.assertNotIn("PRIVATE", str(details))


@unittest.skipUnless(backend_available(), "Pinned image backend is unavailable")
class PixelRuleRecoveryTests(unittest.TestCase):
    def test_recovers_numeric_body_and_single_wrapped_header(self):
        region, rules = scan_fixture()
        groups, details = recover_ruled_groups(region, rules)
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "scan_recovery")
        self.assertEqual((table["row_count"], table["column_count"]), (8, 5))
        self.assertEqual(check_grid(table), [])
        self.assertEqual(table["bbox"], [20, 30, 560, 285])
        metadata = details["returned_group_metadata"][0]
        self.assertEqual(metadata["layout_repair"]["header_rows"], 1)
        self.assertEqual(image_evidence(table, metadata, details["horizontal_rules"])[0], [])
        self.assertFalse(details["runtime_flags"]["second_table_extractor_pass"])

    def test_prose_and_blank_bands_do_not_become_tables(self):
        for kind in ("prose", "blank"):
            with self.subTest(kind=kind):
                region, rules = scan_fixture(**{kind: True})
                groups, _ = recover_ruled_groups(region, rules)
                self.assertEqual(groups, [])

    def test_pdf_coordinate_offset_maps_back_correctly(self):
        region, rules = scan_fixture()
        dx, dy = 100, 200
        region.bbox = (dx, dy, dx + 600, dy + 320)
        rules = [[a + dx, b + dy, c + dx, d + dy] for a, b, c, d in rules]
        groups, details = recover_ruled_groups(region, rules)
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "scan_recovery")
        self.assertEqual(table["bbox"], [120, 230, 660, 485])
        self.assertEqual((table["row_count"], table["column_count"]), (8, 5))
        self.assertLessEqual(details["layout_crop_pixels_used"], 4_000_000)


if __name__ == "__main__":
    unittest.main()
