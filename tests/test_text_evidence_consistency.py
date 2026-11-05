"""Text fallback must honor available evidence without rejecting its neighbors."""
import unittest
from unittest.mock import Mock

from table_structure import check_grid, detect_page
from test_detection_regressions import (DetectionPage, detection_backends,
                                       rectangular_grid, words_in_grid)


def rejection_reasons(result, method):
    return {reason for attempt in result["detection_attempts"]
            if attempt["method"] == method
            for rejection in attempt["rejected"] for reason in rejection["reasons"]}


def narrative_column():
    cells = [(x0, top, x1 + (70 if x0 == 270 else 0), bottom)
             for x0, top, x1, bottom in rectangular_grid(rows=5, columns=4)]
    words = words_in_grid(cells)
    for index, box in enumerate(cells):
        offsets = (35, 40) if index % 4 != 3 else range(30, 90, 5)
        words.extend({"x0": box[0] + offset, "top": box[1] + 8,
                      "x1": box[0] + offset + 2, "bottom": box[1] + 18,
                      "text": "local-only"} for offset in offsets)
    return cells, words


def connected_context(status="ok"):
    return {"status": status, "connected_box_groups": [{
        "boxes": [[100, 100, 260, 130], [100, 155, 260, 185],
                  [100, 210, 260, 240]], "connector_count": 2}],
        "large_curved_shapes": []}


class TextEvidenceConsistencyTests(unittest.TestCase):
    def test_picture_and_separate_prose_are_rejected_by_both_inferred_routes(self):
        cells = rectangular_grid(rows=5, columns=4)
        words = words_in_grid(cells)
        for box in cells[3::4]:
            words.extend({"x0": box[0] + 30 + offset, "top": box[1] + 8,
                          "x1": box[0] + 34 + offset, "bottom": box[1] + 18,
                          "text": "local-only"} for offset in (0, 8, 16))
        for strategy in ("text", "img2table", "auto"):
            with self.subTest(strategy=strategy):
                page = DetectionPage(text=[cells], words=words, images=True)
                page.images = [{"x0": 50, "top": 70, "x1": 250, "bottom": 300}]
                with detection_backends(img2table=[cells]):
                    result = detect_page(page, strategy)
                self.assertEqual(result["tables"], [])
                method = "img2table" if strategy == "img2table" else "text"
                self.assertIn("layout_crosses_image_and_neighboring_prose",
                              rejection_reasons(result, method))

    def test_unruled_long_prose_evidence_is_honored_by_text_and_image_routes(self):
        cells, words = narrative_column()
        for strategy in ("text", "img2table", "auto"):
            with self.subTest(strategy=strategy), detection_backends(img2table=[cells]):
                result = detect_page(DetectionPage(text=[cells], words=words), strategy)
            self.assertEqual(result["tables"], [])
            method = "img2table" if strategy == "img2table" else "text"
            self.assertIn("unruled_layout_includes_long_prose_column",
                          rejection_reasons(result, method))

    def test_printed_rule_support_preserves_a_table_with_a_narrative_column(self):
        cells, words = narrative_column()
        for strategy in ("img2table", "auto"):
            groups = [cells] if strategy == "img2table" else []
            with self.subTest(strategy=strategy), detection_backends() as (_, backend):
                backend.return_value = (groups, None, {"warnings": [],
                    "horizontal_rules": [[60, 100, 410, 100]]})
                result = detect_page(DetectionPage(text=[cells], words=words), strategy)
            self.assertEqual(len(result["tables"]), 1)
            table = result["tables"][0]
            self.assertTrue(table["evidence"]["long_horizontal_rule_support"])
            self.assertEqual((table["row_count"], table["column_count"]), (5, 4))
            self.assertEqual(check_grid(table), [])

    def test_regular_borderless_text_table_remains_accepted(self):
        cells = rectangular_grid(rows=4, columns=3)
        for strategy in ("text", "auto"):
            with self.subTest(strategy=strategy), detection_backends():
                result = detect_page(DetectionPage(text=[cells], words=words_in_grid(cells)), strategy)
            self.assertEqual(len(result["tables"]), 1)
            table = result["tables"][0]
            self.assertEqual(table["method"], "text")
            self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
            self.assertEqual(check_grid(table), [])

    def test_forced_text_uses_native_printed_rule_without_an_image_pass(self):
        cells, words = narrative_column()
        page = DetectionPage(text=[cells], words=words)
        page.horizontal_edges = [{"x0": 60, "top": 100, "x1": 410, "bottom": 100}]
        with detection_backends() as (_, backend):
            result = detect_page(page, "text")
        backend.assert_not_called()
        self.assertEqual(len(result["tables"]), 1)
        self.assertTrue(result["tables"][0]["evidence"]["long_horizontal_rule_support"])

    def test_native_rule_support_survives_unavailable_or_failed_image_backend(self):
        cells, words = narrative_column()
        for status in ("unavailable", "failed"):
            page = DetectionPage(text=[cells], words=words)
            page.horizontal_edges = [{"x0": 60, "top": 100, "x1": 410, "bottom": 100}]
            with self.subTest(status=status), detection_backends() as (_, backend):
                backend.return_value = ([], None, {"status": status, "warnings": [],
                                                   "horizontal_rules": []})
                result = detect_page(page, "auto")
            self.assertEqual(len(result["tables"]), 1)
            self.assertEqual(result["tables"][0]["method"], "text")
            self.assertTrue(result["tables"][0]["evidence"]["long_horizontal_rule_support"])

    def test_unrelated_image_rules_do_not_hide_native_support_for_text_table(self):
        cells, words = narrative_column()
        page = DetectionPage(text=[cells], words=words)
        page.horizontal_edges = [{"x0": 60, "top": 100, "x1": 410, "bottom": 100}]
        with detection_backends() as (_, backend):
            backend.return_value = ([], None, {"status": "ok", "warnings": [],
                                               "horizontal_rules": [[10, 500, 200, 500]]})
            result = detect_page(page, "auto")
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["method"], "text")
        self.assertTrue(result["tables"][0]["evidence"]["long_horizontal_rule_support"])

    def test_rejected_diagram_cannot_return_through_text_fallback(self):
        cells = rectangular_grid(x=100, top=100, rows=5, columns=2, width=80, height=30)
        page = DetectionPage(text=[cells], words=words_in_grid(cells))
        with detection_backends() as (_, backend):
            backend.return_value = ([cells], None, {"warnings": [],
                "figure_context": connected_context()})
            result = detect_page(page, "auto")
        self.assertEqual(result["tables"], [])
        for method in ("img2table", "text"):
            self.assertIn("connected_box_flow_layout", rejection_reasons(result, method))

    def test_diagram_context_does_not_suppress_a_distinct_text_table(self):
        diagram = rectangular_grid(x=100, top=100, rows=5, columns=2, width=80, height=30)
        good = rectangular_grid(x=350, top=350, rows=4, columns=3, width=60)
        page = DetectionPage(text=[diagram, good], words=words_in_grid(diagram + good))
        with detection_backends() as (_, backend):
            backend.return_value = ([diagram], None, {"warnings": [],
                "figure_context": connected_context()})
            result = detect_page(page, "auto")
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual(table["method"], "text")
        self.assertGreater(table["bbox"][0], 300)
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertEqual(check_grid(table), [])
        self.assertIn("connected_box_flow_layout", rejection_reasons(result, "text"))

    def test_missing_or_partial_context_does_not_reject_valid_text_geometry(self):
        cells = rectangular_grid(x=100, top=100, rows=5, columns=2, width=80, height=30)
        for context in ({}, connected_context("partial"), connected_context("unavailable")):
            with self.subTest(context=context), detection_backends() as (_, backend):
                backend.return_value = ([], None, {"warnings": [], "figure_context": context})
                result = detect_page(DetectionPage(text=[cells], words=words_in_grid(cells)), "auto")
            self.assertEqual(len(result["tables"]), 1)
            self.assertEqual(result["tables"][0]["method"], "text")
            self.assertEqual(check_grid(result["tables"][0]), [])

    def test_rejected_fragment_pixel_statistics_do_not_contaminate_other_crop(self):
        graphic = rectangular_grid(x=40, top=50, rows=5, columns=3)
        good = rectangular_grid(x=350, top=350, rows=4, columns=3, width=60)
        page = DetectionPage(text=[good], words=words_in_grid(good))
        with detection_backends() as (_, backend):
            backend.return_value = ([graphic], None, {"warnings": [],
                "returned_group_metadata": [{"bbox": [40, 50, 250, 200],
                    "pixel_evidence": {"status": "ok", "large_component_ink_fraction": .9,
                                       "character_component_ink_fraction": .05}}]})
            result = detect_page(page, "auto")
        self.assertIn("graphic_dominated_ink", rejection_reasons(result, "img2table"))
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["method"], "text")
        self.assertGreater(result["tables"][0]["bbox"][0], 300)

    def test_live_pixel_sampler_checks_each_text_crop_without_suppressing_neighbor(self):
        graphic = rectangular_grid(x=40, top=50, rows=5, columns=3)
        good = rectangular_grid(x=350, top=350, rows=4, columns=3, width=60)
        bad_pixel = {"status": "ok", "large_component_ink_fraction": .9,
                     "character_component_ink_fraction": .05}
        good_pixel = {"status": "ok", "large_component_ink_fraction": 0,
                      "character_component_ink_fraction": .95}
        sampler = Mock(side_effect=lambda bbox: bad_pixel if bbox[0] < 300 else good_pixel)

        def provide_cache(region, **kwargs):
            kwargs["evidence_cache"]["sample_candidate"] = sampler
            return [graphic], None, {"warnings": [], "returned_group_metadata": [
                {"pixel_evidence": bad_pixel}]}

        page = DetectionPage(text=[graphic, good], words=words_in_grid(graphic + good))
        with detection_backends() as (_, backend):
            backend.side_effect = provide_cache
            result = detect_page(page, "auto")
            self.assertEqual(backend.call_count, 1)
        self.assertIn("graphic_dominated_ink", rejection_reasons(result, "img2table"))
        self.assertIn("graphic_dominated_ink", rejection_reasons(result, "text"))
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (4, 3))
        self.assertGreater(table["bbox"][0], 300)
        self.assertEqual(sampler.call_count, 2)
        rejected = next(a for a in result["detection_attempts"] if a["method"] == "text")["rejected"]
        self.assertEqual(sampler.call_args_list[0].args[0], rejected[0]["bbox"])
        self.assertEqual(sampler.call_args_list[1].args[0], table["bbox"])

    def test_partial_unavailable_or_failed_pixel_sample_preserves_text_candidate(self):
        cells = rectangular_grid(rows=4, columns=3)
        strong = {"large_component_ink_fraction": .95, "character_component_ink_fraction": .01}
        for sample in ({**strong, "status": "partial"},
                       {**strong, "status": "unavailable"},
                       RuntimeError("PRIVATE PIXEL DIAGNOSTIC")):
            with self.subTest(sample=sample):
                sampler = Mock(side_effect=sample) if isinstance(sample, Exception) else Mock(return_value=sample)

                def provide_cache(region, **kwargs):
                    kwargs["evidence_cache"]["sample_candidate"] = sampler
                    return [], None, {"warnings": []}

                with detection_backends() as (_, backend):
                    backend.side_effect = provide_cache
                    result = detect_page(DetectionPage(text=[cells], words=words_in_grid(cells)), "auto")
                self.assertEqual(sampler.call_count, 1)
                self.assertEqual(len(result["tables"]), 1)
                self.assertEqual(result["tables"][0]["method"], "text")
                self.assertNotIn("PRIVATE PIXEL DIAGNOSTIC", str(result))

    def test_pixel_evidence_does_not_impose_image_only_minimum_rows_on_text(self):
        cells = rectangular_grid(rows=3, columns=3)
        sampler = Mock(return_value={"status": "ok", "large_component_ink_fraction": 0,
                                     "character_component_ink_fraction": .95})

        def provide_cache(region, **kwargs):
            kwargs["evidence_cache"]["sample_candidate"] = sampler
            return [], None, {"warnings": [], "horizontal_rules": []}

        with detection_backends() as (_, backend):
            backend.side_effect = provide_cache
            result = detect_page(DetectionPage(text=[cells], words=words_in_grid(cells)), "auto")
        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(len(result["tables"]), 1)
        table = result["tables"][0]
        self.assertEqual((table["row_count"], table["column_count"]), (3, 3))
        self.assertFalse(table["evidence"]["long_horizontal_rule_support"])


if __name__ == "__main__":
    unittest.main()
