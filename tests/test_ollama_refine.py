"""Offline contract tests. No test contacts an Ollama server."""

import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ollama_refine as refine


def table():
    return {
        "bbox": [100, 200, 300, 400],
        "row_count": 2,
        "column_count": 2,
        "cells": [
            {"id": "a", "row": 0, "column": 0, "row_span": 1, "column_span": 1, "bbox": [100, 200, 200, 300]},
            {"id": "b", "row": 0, "column": 1, "row_span": 1, "column_span": 1, "bbox": [200, 200, 300, 300]},
            {"id": "c", "row": 1, "column": 0, "row_span": 1, "column_span": 1, "bbox": [100, 300, 200, 400]},
            {"id": "d", "row": 1, "column": 1, "row_span": 1, "column_span": 1, "bbox": [200, 300, 300, 400]},
        ],
    }


def response(merges, **extra):
    body = {"message": {"content": json.dumps({"merges": merges})}, "done": True}
    body.update(extra)
    return io.BytesIO(json.dumps(body).encode())


class ProposeMergesTests(unittest.TestCase):
    def call(self, source=None, words=None, **kwargs):
        return refine.propose_merges(
            source if source is not None else table(),
            words if words is not None else [],
            model="configured-alias",
            url="https://example.test/ollama/api/chat",
            **kwargs,
        )

    def test_valid_adjacent_groups_and_no_input_mutation(self):
        source = table()
        before = copy.deepcopy(source)
        with patch.object(refine, "_open_request", return_value=response([["d", "c"], ["b", "a"]])):
            result = self.call(source)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["merges"], [["a", "b"], ["c", "d"]])
        self.assertEqual(source, before)

    def test_valid_existing_span(self):
        source = table()
        source["cells"] = [
            {"id": "wide", "row": 0, "column": 0, "row_span": 1, "column_span": 2, "bbox": [100, 200, 300, 300]},
            *source["cells"][2:],
        ]
        with patch.object(refine, "_open_request", return_value=response([["wide", "c", "d"]])):
            result = self.call(source)
        self.assertEqual(result["status"], "proposed")

    def test_nonrectangular_groups_rejected(self):
        for group in (["a", "d"], ["a", "b", "c"]):
            with self.subTest(group=group), patch.object(refine, "_open_request", return_value=response([group])):
                result = self.call()
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["metrics"]["reason"], "non_rectangular_merge")
            self.assertEqual(result["merges"], [])

    def test_unknown_repeated_bool_and_reused_ids_rejected(self):
        for groups in ([["a", "invented"]], [["a", "a"]], [["a", True]], [["a", "b"], ["b", "d"]]):
            with self.subTest(groups=groups), patch.object(refine, "_open_request", return_value=response(groups)):
                self.assertEqual(self.call()["status"], "rejected")

    def test_physical_gaps_and_overlap_rejected(self):
        for left in (190, 210):
            source = table()
            source["cells"][1]["bbox"][0] = left
            with self.subTest(left=left), patch.object(refine, "_open_request", return_value=response([["a", "b"]])):
                result = self.call(source)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["metrics"]["reason"], "physical_geometry_mismatch")

    def test_single_cell_and_too_many_groups_rejected(self):
        for groups in ([["a"]], [["a", "b"]] * (refine.MAX_MERGE_GROUPS + 1)):
            with self.subTest(count=len(groups)), patch.object(refine, "_open_request", return_value=response(groups)):
                self.assertEqual(self.call()["status"], "rejected")

    def test_no_change(self):
        with patch.object(refine, "_open_request", return_value=response([])):
            self.assertEqual(self.call()["status"], "no_change")

    def test_request_privacy_normalization_and_options(self):
        source = table()
        source["text"] = "PRIVATE_DOCUMENT_CONTENT"
        source["header"] = {"names": ["PRIVATE_DOCUMENT_CONTENT"]}
        source["cells"][0]["text"] = "PRIVATE_DOCUMENT_CONTENT"
        words = [[120, 220, 150, 240], [0, 0, 10, 10]]
        with patch.object(refine, "_open_request", return_value=response([], prompt_eval_count=25)) as opened:
            result = self.call(source, words, api_key="dummy-test-token", timeout=12)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/ollama/api/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-token")
        self.assertEqual(opened.call_args.kwargs["timeout"], 12)
        payload = json.loads(request.data)
        self.assertNotIn("PRIVATE_DOCUMENT_CONTENT", request.data.decode())
        self.assertNotIn("dummy-test-token", request.data.decode())
        self.assertNotIn("dummy-test-token", json.dumps(result))
        self.assertEqual(payload["model"], "configured-alias")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096})
        self.assertIsInstance(payload["format"], dict)
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(user["table"]), {"bbox", "row_count", "column_count", "cells"})
        self.assertEqual(user["word_boxes"], [[0.1, 0.1, 0.25, 0.2]])
        self.assertEqual(user["table"]["cells"][0]["bbox"], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(result["metrics"]["prompt_eval_count"], 25)

    def test_json_mode(self):
        with patch.object(refine, "_open_request", return_value=response([])) as opened:
            self.call(json_schema=False)
        self.assertEqual(json.loads(opened.call_args.args[0].data)["format"], "json")

    def test_prompt_limit_skips_without_request(self):
        with patch.object(refine, "_open_request") as opened:
            result = self.call(max_prompt_chars=20)
        opened.assert_not_called()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["metrics"]["reason"], "prompt_limit")

    def test_invalid_word_box_and_overlapping_input_skip(self):
        source = table()
        source["cells"][1]["column"] = 0
        for candidate, words in ((table(), [[1, 2, 3, 4, "PRIVATE"]]), (source, [])):
            with self.subTest(words=words), patch.object(refine, "_open_request") as opened:
                self.assertEqual(self.call(candidate, words)["status"], "skipped")
                opened.assert_not_called()

    def test_truncation_rejected(self):
        for extra in ({"done": False}, {"done_reason": "length"}):
            with self.subTest(extra=extra), patch.object(refine, "_open_request", return_value=response([["a", "b"]], **extra)):
                self.assertEqual(self.call()["metrics"]["reason"], "truncated_response")

    def test_timeout_and_transport_errors_do_not_echo_secrets(self):
        errors = (
            TimeoutError("PRIVATE"),
            urllib.error.URLError(TimeoutError("PRIVATE")),
            urllib.error.URLError("PRIVATE"),
            urllib.error.HTTPError("https://example.test", 401, "PRIVATE", {}, None),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), patch.object(refine, "_open_request", side_effect=error):
                result = self.call(api_key="dummy-test-token")
            self.assertEqual(result["status"], "error")
            self.assertNotIn("PRIVATE", json.dumps(result))
            self.assertNotIn("dummy-test-token", json.dumps(result))

    def test_malformed_oversized_and_extra_output_rejected(self):
        bodies = (
            b"not json",
            b"x" * (refine.MAX_RESPONSE_BYTES + 1),
            json.dumps({"done": True, "message": {"content": '{"merges":[],"text":"PRIVATE"}'}}).encode(),
            json.dumps({"done": True, "message": {"content": "x" * (refine.MAX_CONTENT_CHARS + 1)}}).encode(),
        )
        for body in bodies:
            with self.subTest(length=len(body)), patch.object(refine, "_open_request", return_value=io.BytesIO(body)):
                result = self.call()
            self.assertEqual(result["status"], "rejected")
            self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
