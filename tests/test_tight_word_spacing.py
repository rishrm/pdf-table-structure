"""Publisher positioning gaps must not turn whole prose lines into one word."""
from types import SimpleNamespace
import unittest

from pdfplumber.utils import extract_words

from table_structure import detect_page, word_boxes_only
from test_detection_regressions import DetectionPage


def positioned_chars(x, top, tokens, size=10):
    """PDF character objects with no literal whitespace glyphs between words."""
    chars = []
    for token in tokens:
        for letter in token:
            chars.append({"text": letter, "x0": x, "x1": x + size * .4,
                          "top": top, "bottom": top + size * .8,
                          "doctop": top, "upright": True, "size": size})
            x += size * .4
        x += size * .2
    return chars


class TightWordSpacingTests(unittest.TestCase):
    def test_word_separation_scales_with_font_without_exporting_text(self):
        for size in (6, 10, 20):
            with self.subTest(font_size=size):
                chars = positioned_chars(20, 40, ["one", "two", "six"], size)
                page = SimpleNamespace(extract_words=lambda **kwargs: extract_words(chars, **kwargs))
                boxes = word_boxes_only(page)
                self.assertEqual(len(boxes), 3)
                self.assertTrue(all(len(box) == 4 and all(isinstance(v, (float, int)) for v in box)
                                    for box in boxes))
        chars = positioned_chars(20, 40, ["one", "two", "six"])
        self.assertEqual(len(extract_words(chars)), 1)  # The old fixed tolerance merges them.

    def test_reference_style_prose_is_rejected_with_positioned_spaces(self):
        xs = [40, 60, 290, 310, 540]
        ys = [100 + 20 * row for row in range(13)]
        cells = [(a, c, b, d) for c, d in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]
        chars = []
        for row in range(12):
            top = ys[row] + 5
            for x in (64, 314):
                chars += positioned_chars(x, top, ["one"] * 9)
            if row % 3 == 0:
                for x in (44, 294):
                    chars += positioned_chars(x, top, ["1"])
        page = DetectionPage(text=[cells])
        page.chars = chars
        page.extract_words = lambda **kwargs: extract_words(chars, **kwargs)
        result = detect_page(page, "text")
        self.assertEqual(result["tables"], [])
        self.assertIn("prose_like_columns", result["detection_attempts"][0]["rejected"][0]["reasons"])

    def test_real_compact_table_stays_accepted_with_positioned_spaces(self):
        xs, ys = [40, 180, 320, 460], [100, 125, 150, 175, 200]
        cells = [(a, c, b, d) for c, d in zip(ys, ys[1:]) for a, b in zip(xs, xs[1:])]
        chars = [char for a, c, _, _ in cells for char in positioned_chars(a + 5, c + 5, ["one", "two"])]
        page = DetectionPage(text=[cells])
        page.chars = chars
        page.extract_words = lambda **kwargs: extract_words(chars, **kwargs)
        result = detect_page(page, "text")
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual((result["tables"][0]["row_count"], result["tables"][0]["column_count"]), (4, 3))


if __name__ == "__main__":
    unittest.main()
