"""Box connectors and distributed graphics are evidence, not semantic labels."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

import img2table_structure as adapter
from figure_quality import context_evidence, figure_evidence_reasons
from test_img2table_structure import backend_available


def flow_boxes(connect=True):
    image = Image.new("RGB", (340, 330), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 130, 240):
        draw.rounded_rectangle((75, y, 265, y + 55), radius=8, outline="black", width=2)
        draw.text((100, y + 20), "ordinary boxed text", font=ImageFont.load_default(size=12), fill="black")
    if connect:
        for y in (75, 185):
            draw.line((170, y, 170, y + 55), fill="black", width=2)
            draw.polygon([(170, y + 55), (166, y + 47), (174, y + 47)], fill="black")
    return image


def table_with_graphics(graphic_columns):
    image = Image.new("RGB", (370, 250), "white")
    draw = ImageDraw.Draw(image)
    for x in (10, 125, 240, 355):
        draw.line((x, 10, x, 235), fill="black", width=2)
    for y in (10, 85, 160, 235):
        draw.line((10, y, 355, y), fill="black", width=2)
    for row, y in enumerate((47, 122, 197)):
        for col, x in enumerate((67, 182, 297)):
            if col in graphic_columns:
                draw.ellipse((x - 26, y - 29, x + 26, y + 29), outline="black", width=2)
                draw.text((x - 17, y - 4), "label", font=ImageFont.load_default(size=9), fill="black")
            else:
                draw.text((x - 40, y - 14), "Ordinary text", font=ImageFont.load_default(size=11), fill="black")
                draw.text((x - 40, y + 2), "1234 values", font=ImageFont.load_default(size=11), fill="black")
    return image


@unittest.skipUnless(backend_available(), "pinned image backend unavailable")
class FlowchartPixelEvidenceTests(unittest.TestCase):
    def context(self, image, scale=1):
        np, _, cv2 = adapter._load_backend()
        pixels = np.asarray(image)
        foreground = (cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) < 180).astype(np.uint8) * 255
        return adapter._diagram_context(foreground, scale, scale, np, cv2)

    def reasons(self, image, box=None):
        box = box or [0, 0, image.width, image.height]
        return figure_evidence_reasons({"diagram_context": context_evidence(box, self.context(image))})

    def test_separate_boxes_with_interior_connectors_are_a_flow_network(self):
        self.assertEqual(self.reasons(flow_boxes(), [70, 15, 270, 300]), ["connected_box_flow_layout"])

    def test_unconnected_boxed_form_is_not_rejected_as_flowchart(self):
        self.assertEqual(self.reasons(flow_boxes(connect=False)), [])

    def test_adjacent_table_cells_and_graphics_confined_to_one_column_are_retained(self):
        self.assertEqual(self.reasons(table_with_graphics(set())), [])
        self.assertEqual(self.reasons(table_with_graphics({2})), [])

    def test_repeated_large_graphic_panels_across_rows_and_columns_are_rejected(self):
        self.assertEqual(self.reasons(table_with_graphics({0, 1, 2})), ["repeated_graphic_panels"])

    def test_large_heading_over_regular_data_is_retained(self):
        image = Image.new("RGB", (500, 260), "white")
        draw = ImageDraw.Draw(image)
        draw.text((15, 10), "BIG HEADER", font=ImageFont.load_default(size=68), fill="black")
        for y in (100, 130, 160, 190, 220):
            for x in (30, 180, 330):
                draw.text((x, y), "123", font=ImageFont.load_default(size=12), fill="black")
        self.assertEqual(self.reasons(image), [])

    def test_physical_scale_and_crop_offset_preserve_the_flow_decision(self):
        image = flow_boxes()
        bigger = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
        context = self.context(bigger, 2)
        relocated = adapter._page_context(context, [100, 200, 440, 530])
        self.assertEqual(figure_evidence_reasons({"diagram_context":
            context_evidence([170, 215, 370, 500], relocated)}), ["connected_box_flow_layout"])
        self.assertEqual(figure_evidence_reasons({"diagram_context":
            context_evidence([500, 600, 840, 930], relocated)}), [])

    def test_context_budget_failure_does_not_turn_missing_evidence_into_a_rejection(self):
        with patch.object(adapter, "MAX_CONTEXT_FRAMES", 1):
            context = self.context(flow_boxes())
        self.assertEqual(context["status"], "partial")
        self.assertEqual(figure_evidence_reasons({"diagram_context":
            context_evidence([0, 0, 340, 330], context)}), [])

    def test_context_failure_preserves_existing_pixel_evidence(self):
        np, _, cv2 = adapter._load_backend()
        image = flow_boxes()
        with patch.object(adapter, "_diagram_context", side_effect=RuntimeError("private diagnostic")):
            masks = adapter._prepare_pixel_evidence(np.asarray(image),
                SimpleNamespace(horizontal_lines=[], vertical_lines=[]), 1, 1, np, cv2)
        evidence = adapter._pixel_evidence([(0, 0, 340, 330)], masks, 1, 1, np, cv2)
        self.assertEqual(evidence["status"], "ok")
        self.assertEqual(evidence["diagram_context"]["status"], "unavailable")
        self.assertNotIn("private diagnostic", str(evidence))


class FlowchartContextContractTests(unittest.TestCase):
    def test_rebuilt_native_grid_can_use_context_without_old_pixel_statistics(self):
        context = {"status": "ok", "connected_box_groups": [{"boxes":
            [[0, 0, 30, 15], [0, 25, 30, 40], [0, 50, 30, 65]], "connector_count": 2}],
            "large_curved_shapes": []}
        self.assertEqual(figure_evidence_reasons({"status": "unavailable", "diagram_context":
            context_evidence([0, 0, 30, 65], context)}), ["connected_box_flow_layout"])


if __name__ == "__main__":
    unittest.main()
