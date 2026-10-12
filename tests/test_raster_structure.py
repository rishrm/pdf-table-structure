"""Pixel-grid behavior, including spans, skew, and bounded failure modes."""
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from raster_structure import detect_image_grids, map_to_original
from table_structure import check_grid, structure_from_cells


def grid_image():
    image = Image.new("RGB", (660, 440), "white")
    draw = ImageDraw.Draw(image)
    for x in (80, 220, 500):
        draw.line((x, 80, x, 320), fill="black", width=2)
    draw.line((360, 140, 360, 320), fill="black", width=2)
    for y in (80, 140, 260, 320):
        draw.line((80, y, 500, y), fill="black", width=2)
    draw.line((220, 200, 500, 200), fill="black", width=2)
    for x, y, label in [(90, 100, "Header"), (230, 100, "Spanning header"),
                         (90, 180, "Spanning rows"), (230, 165, "A"),
                         (370, 165, "B"), (230, 225, "C"), (370, 225, "D"),
                         (90, 285, "Last"), (230, 285, "E")]:
        draw.text((x, y), label, fill="black")
    return image


class RasterTests(unittest.TestCase):
    def test_pixels_preserve_merged_cells_and_empty_cell(self):
        groups, angle, diagnostics = detect_image_grids(grid_image(), scale=1)
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "raster")
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(len(table["cells"]), 10)
        self.assertTrue(any(c["row_span"] == 2 for c in table["cells"]))
        self.assertTrue(any(c["column_span"] == 2 for c in table["cells"]))
        self.assertTrue(any(c["row"] == 3 and c["column"] == 2 for c in table["cells"]))
        self.assertEqual(check_grid(table), [])
        self.assertEqual(angle, 0)

    def test_mild_skew_is_corrected_in_both_directions(self):
        for original_angle in (-1.5, 0.7):
            with self.subTest(angle=original_angle):
                image = grid_image().rotate(original_angle, resample=Image.Resampling.BICUBIC,
                                             fillcolor="white")
                groups, correction, diagnostics = detect_image_grids(image, scale=1)
                self.assertAlmostEqual(correction, -original_angle, delta=0.3)
                self.assertEqual(len(groups), 1)
                table = structure_from_cells(groups[0], "raster")
                self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
                self.assertEqual(len(table["cells"]), 10)
                self.assertEqual(check_grid(table), [])
                transform = {"deskew_degrees": correction, "render_bbox": [0, 0, 660, 440],
                             "image_size": [660, 440], "scale_x": 1, "scale_y": 1}
                map_to_original(table, transform)
                self.assertIn("polygon", table["cells"][0])
                # A known top-left corner, rotated in the original image, must
                # land near the mapped detector corner (not the deskewed one).
                import math
                a = math.radians(original_angle)
                x, y = 80 - 330, 80 - 220
                expected = [math.cos(a) * x + math.sin(a) * y + 330,
                            -math.sin(a) * x + math.cos(a) * y + 220]
                for actual, target in zip(table["polygon"][0], expected):
                    self.assertAlmostEqual(actual, target, delta=3)

    def test_small_line_gap_does_not_destroy_the_grid(self):
        image = grid_image()
        ImageDraw.Draw(image).rectangle((300, 139, 301, 141), fill="white")
        groups, _, _ = detect_image_grids(image, scale=1)
        self.assertEqual(len(groups), 1)
        table = structure_from_cells(groups[0], "raster")
        self.assertEqual(len(table["cells"]), 10)
        self.assertEqual(check_grid(table), [])

    def test_neighboring_image_tables_stay_separate(self):
        image = Image.new("RGB", (1320, 440), "white")
        image.paste(grid_image(), (0, 0))
        image.paste(grid_image(), (660, 0))
        groups, _, _ = detect_image_grids(image, scale=1)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 10 for group in groups))

    def test_text_without_grid_does_not_become_cells(self):
        image = Image.new("RGB", (660, 440), "white")
        draw = ImageDraw.Draw(image)
        for y in range(60, 350, 20):
            draw.text((40, y), "This is a paragraph. It has words but no drawn table grid.", fill="black")
        groups, _, _ = detect_image_grids(image, scale=1)
        self.assertEqual(groups, [])

    def test_horizontal_rules_alone_are_not_a_closed_table(self):
        image = Image.new("RGB", (660, 440), "white")
        draw = ImageDraw.Draw(image)
        for y in (80, 140, 200, 260):
            draw.line((80, y, 500, y), fill="black", width=2)
        groups, _, _ = detect_image_grids(image, scale=1)
        self.assertEqual(groups, [])

    def test_complexity_limit_skips_instead_of_hanging(self):
        with patch("raster_structure.MAX_MERGED_EDGES", 1):
            groups, _, diagnostics = detect_image_grids(grid_image(), scale=1, deskew=False)
        self.assertEqual(groups, [])
        self.assertIn("raster_merged_line_limit", diagnostics["warnings"])


if __name__ == "__main__":
    unittest.main()
