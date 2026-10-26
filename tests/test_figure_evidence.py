"""Pixel-size evidence distinguishes graphics from text tables without OCR."""
from types import SimpleNamespace
import unittest

from PIL import Image, ImageDraw, ImageFont

import img2table_structure as adapter
from figure_quality import figure_evidence_reasons
from test_img2table_structure import backend_available


@unittest.skipUnless(backend_available(), "pinned image backend unavailable")
class FigureEvidenceTests(unittest.TestCase):
    def evidence(self, image, scale=1):
        np, _, cv2 = adapter._load_backend()
        extractor = SimpleNamespace(horizontal_lines=[], vertical_lines=[])
        masks = adapter._prepare_pixel_evidence(np.asarray(image), extractor, scale, scale, np, cv2)
        return adapter._pixel_evidence([(0, 0, image.width - 1, image.height - 1)],
                                       masks, scale, scale, np, cv2)

    def schematic(self):
        image = Image.new("RGB", (360, 260), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=10)
        for y in (40, 125, 210):
            draw.line((40, y, 320, y), fill=(195, 195, 195), width=3)
            for x in (40, 320):
                draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=(130, 130, 130))
            draw.text((115, y - 18), "Small shape labels", font=font, fill="black")
        return image

    def test_repeated_diagram_shapes_dominate_local_label_scale(self):
        evidence = self.evidence(self.schematic())
        self.assertGreater(evidence["oversized_component_ink_fraction"], .6)
        self.assertGreater(evidence["oversized_component_vertical_spread"], .35)
        self.assertEqual(figure_evidence_reasons(evidence),
                         ["graphic_components_dominate_local_text_scale"])

    def test_diffuse_photo_tones_and_large_nontext_shapes_supply_independent_evidence(self):
        np, _, _ = adapter._load_backend()
        y, x = np.indices((220, 360))
        gray = (190 + (7 * x + 3 * y) % 31).astype(np.uint8)
        for center in (100, 260):
            shape = ((x - center) / 45) ** 2 + ((y - 95) / 55) ** 2 < 1
            gray[shape] = (95 + (3 * x[shape] + y[shape]) % 60).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text((30, 180), "small annotation across the picture", font=ImageFont.load_default(size=10),
                  fill="black")
        evidence = self.evidence(image)
        self.assertGreater(evidence["midtone_fraction"], .85)
        self.assertLess(evidence["midtone_peak_fraction"], .2)
        self.assertLess(evidence["oversized_component_count"], 4)
        self.assertEqual(figure_evidence_reasons(evidence),
                         ["graphic_components_dominate_local_text_scale"])

    def test_large_header_and_flat_table_shading_are_not_graphic_evidence(self):
        for shade in (255, 215):
            with self.subTest(background=shade):
                image = Image.new("RGB", (500, 260), (shade, shade, shade))
                draw = ImageDraw.Draw(image)
                draw.text((15, 10), "BIG HEADER", font=ImageFont.load_default(size=68), fill="black")
                for y in (100, 130, 160, 190, 220):
                    for x in (30, 180, 330):
                        draw.text((x, y), "12", font=ImageFont.load_default(size=12), fill="black")
                evidence = self.evidence(image)
                self.assertGreater(evidence["oversized_component_ink_fraction"], .6)
                self.assertLess(evidence["oversized_component_vertical_spread"], .35)
                self.assertEqual(figure_evidence_reasons(evidence), [])

    def test_noisy_gray_scan_background_with_regular_text_is_retained(self):
        np, _, _ = adapter._load_backend()
        y, x = np.indices((260, 360))
        image = Image.fromarray((200 + (x * 3 + y * 7) % 25).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for y in (20, 65, 110, 155, 200):
            for x in (20, 130, 240):
                draw.text((x, y), "Data 123", font=ImageFont.load_default(size=14), fill="black")
        evidence = self.evidence(image)
        self.assertGreater(evidence["midtone_fraction"], .85)
        self.assertLess(evidence["oversized_component_ink_fraction"], .1)
        self.assertEqual(figure_evidence_reasons(evidence), [])

    def test_physical_render_scale_preserves_the_graphic_decision(self):
        image = self.schematic()
        one = self.evidence(image)
        two = self.evidence(image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST), 2)
        for key in ("reference_component_height_points", "oversized_component_count",
                    "oversized_component_ink_fraction", "oversized_component_vertical_spread"):
            self.assertEqual(one[key], two[key], key)
        self.assertEqual(figure_evidence_reasons(one), figure_evidence_reasons(two))


class FigureGateContractTests(unittest.TestCase):
    def test_missing_partial_or_insufficient_evidence_cannot_reject_a_candidate(self):
        strong = {"status": "ok", "reference_component_count": 40,
                  "oversized_component_ink_fraction": .9, "oversized_component_count": 6,
                  "oversized_component_vertical_spread": .8}
        self.assertTrue(figure_evidence_reasons(strong))
        for evidence in ({}, {**strong, "status": "partial"},
                         {**strong, "reference_component_count": 4}):
            self.assertEqual(figure_evidence_reasons(evidence), [])


if __name__ == "__main__":
    unittest.main()
