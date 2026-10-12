"""Optional Ollama merge proposals from geometry alone.

This module does not extract or transmit PDF text.  Proposals are deliberately
returned separately from the input table so callers can compare both versions.
"""

from __future__ import annotations

import json
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


MAX_CELLS = 512
MAX_WORD_BOXES = 1000
MAX_GRID_AREA = 65536
MAX_MERGE_GROUPS = 64
MAX_GROUP_SIZE = 128
MAX_RESPONSE_BYTES = 65536
MAX_CONTENT_CHARS = 16000

_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SYSTEM = (
    "Suggest only highly justified merges of adjacent cells in a PDF table. "
    "You receive structure and word bounding boxes, never cell text. Coordinates "
    "are normalized to the table, with origin at top left. Existing cells may "
    "already span rows or columns. A merge must form a completely tiled rectangle "
    "in the existing grid, without holes or overlap. Use each existing cell ID at "
    "most once; never invent IDs. Do not merge merely because cells are empty or "
    "have similar dimensions. When geometry is ambiguous, leave it unchanged. "
    'Return only JSON with exactly the key "merges", containing groups of at least '
    'two existing cell IDs, or {"merges":[]} when no merge is justified.'
)


class _InvalidInput(ValueError):
    pass


class _InvalidProposal(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep authentication and geometry on the exact configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_request(request: urllib.request.Request, timeout: float):
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _InvalidInput("invalid_input")
    return value


def _bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _InvalidInput("invalid_input")
    if any(type(v) not in (int, float) for v in value):
        raise _InvalidInput("invalid_input")
    try:
        box = [float(v) for v in value]
    except (ValueError, OverflowError):
        raise _InvalidInput("invalid_input") from None
    if any(not math.isfinite(v) for v in box):
        raise _InvalidInput("invalid_input")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise _InvalidInput("invalid_input")
    return box


def _normalized(box: list[float], table_box: list[float]) -> list[float]:
    x0, y0, x1, y1 = table_box
    width, height = x1 - x0, y1 - y0
    return [
        round((box[0] - x0) / width, 5),
        round((box[1] - y0) / height, 5),
        round((box[2] - x0) / width, 5),
        round((box[3] - y0) / height, 5),
    ]


def _sanitize(table: dict, word_boxes: list[list[float]]) -> tuple[dict, dict]:
    if not isinstance(table, dict) or not isinstance(word_boxes, list):
        raise _InvalidInput("invalid_input")
    table_box = _bbox(table.get("bbox"))
    rows = _integer(table.get("row_count"), 1)
    columns = _integer(table.get("column_count"), 1)
    if rows * columns > MAX_GRID_AREA:
        raise _InvalidInput("input_limit")
    cells = table.get("cells")
    if not isinstance(cells, list) or len(cells) > MAX_CELLS:
        raise _InvalidInput("input_limit")
    if len(word_boxes) > MAX_WORD_BOXES:
        raise _InvalidInput("input_limit")

    cleaned, by_id, occupied = [], {}, set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise _InvalidInput("invalid_input")
        cell_id = cell.get("id")
        if not (
            (type(cell_id) is int and cell_id >= 0)
            or (isinstance(cell_id, str) and _ID_PATTERN.fullmatch(cell_id))
        ):
            raise _InvalidInput("invalid_input")
        if cell_id in by_id:
            raise _InvalidInput("invalid_input")
        row, column = _integer(cell.get("row")), _integer(cell.get("column"))
        row_span = _integer(cell.get("row_span"), 1)
        column_span = _integer(cell.get("column_span"), 1)
        if row + row_span > rows or column + column_span > columns:
            raise _InvalidInput("invalid_input")
        box = _bbox(cell.get("bbox"))
        tolerance = 1e-5
        if (
            box[0] < table_box[0] - tolerance
            or box[1] < table_box[1] - tolerance
            or box[2] > table_box[2] + tolerance
            or box[3] > table_box[3] + tolerance
        ):
            raise _InvalidInput("invalid_input")
        grid = {
            (r, c)
            for r in range(row, row + row_span)
            for c in range(column, column + column_span)
        }
        if occupied.intersection(grid):
            raise _InvalidInput("invalid_input")
        occupied.update(grid)
        safe_cell = {
            "id": cell_id,
            "row": row,
            "column": column,
            "row_span": row_span,
            "column_span": column_span,
            "bbox": _normalized(box, table_box),
        }
        cleaned.append(safe_cell)
        by_id[cell_id] = safe_cell

    safe_words = []
    for word in word_boxes:
        box = _bbox(word)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (table_box[0] <= cx <= table_box[2] and table_box[1] <= cy <= table_box[3]):
            continue
        clipped = [
            max(box[0], table_box[0]),
            max(box[1], table_box[1]),
            min(box[2], table_box[2]),
            min(box[3], table_box[3]),
        ]
        safe_words.append(_normalized(clipped, table_box))

    return {
        "table": {
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "row_count": rows,
            "column_count": columns,
            "cells": cleaned,
        },
        "word_boxes": safe_words,
    }, by_id


def _schema(ids: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "merges": {
                "type": "array",
                "maxItems": MAX_MERGE_GROUPS,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_GROUP_SIZE,
                    "uniqueItems": True,
                    "items": {"enum": ids},
                },
            }
        },
        "required": ["merges"],
        "additionalProperties": False,
    }


def _validate_proposal(proposal: Any, by_id: dict) -> list[list]:
    if not isinstance(proposal, dict) or set(proposal) != {"merges"}:
        raise _InvalidProposal("invalid_schema")
    groups = proposal["merges"]
    if not isinstance(groups, list) or len(groups) > MAX_MERGE_GROUPS:
        raise _InvalidProposal("output_limit")
    seen, result = set(), []
    for group in groups:
        if not isinstance(group, list) or not (2 <= len(group) <= MAX_GROUP_SIZE):
            raise _InvalidProposal("invalid_group")
        local_ids, grid, selected = set(), set(), []
        for cell_id in group:
            if type(cell_id) not in (str, int) or cell_id not in by_id:
                raise _InvalidProposal("unknown_cell_id")
            if cell_id in local_ids or cell_id in seen:
                raise _InvalidProposal("repeated_cell_id")
            local_ids.add(cell_id)
            cell = by_id[cell_id]
            selected.append(cell)
            for row in range(cell["row"], cell["row"] + cell["row_span"]):
                for column in range(cell["column"], cell["column"] + cell["column_span"]):
                    if (row, column) in grid:
                        raise _InvalidProposal("overlapping_grid")
                    grid.add((row, column))
        top = min(c["row"] for c in selected)
        left = min(c["column"] for c in selected)
        bottom = max(c["row"] + c["row_span"] for c in selected)
        right = max(c["column"] + c["column_span"] for c in selected)
        expected = {
            (row, column)
            for row in range(top, bottom)
            for column in range(left, right)
        }
        if grid != expected:
            raise _InvalidProposal("non_rectangular_merge")
        # A logical rectangle must also tile the physical rectangle. This
        # catches inconsistent grid metadata or bboxes with gaps/overlap.
        boxes = [c["bbox"] for c in selected]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        epsilon = 1e-8
        if abs(area - (x1 - x0) * (y1 - y0)) > epsilon:
            raise _InvalidProposal("physical_geometry_mismatch")
        for index, first in enumerate(boxes):
            for second in boxes[index + 1:]:
                overlap_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
                overlap_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
                if overlap_width * overlap_height > epsilon:
                    raise _InvalidProposal("physical_geometry_mismatch")
        seen.update(local_ids)
        selected.sort(key=lambda c: (c["row"], c["column"]))
        result.append([c["id"] for c in selected])
    result.sort(key=lambda group: (by_id[group[0]]["row"], by_id[group[0]]["column"]))
    return result


def propose_merges(
    table: dict,
    word_boxes: list[list[float]],
    *,
    model: str,
    url: str,
    api_key: str | None = None,
    timeout: float = 45,
    max_prompt_chars: int = 14000,
    json_schema: bool = True,
) -> dict:
    """Return validated merge proposals; never mutate ``table``.

    Status is ``proposed``, ``no_change``, ``rejected`` (invalid model output),
    ``skipped`` (invalid/oversize input), or ``error`` (transport failure).
    ``metrics.reason`` is a fixed diagnostic code; raw errors and responses are
    never returned.  A valid proposal is structural, not an accuracy guarantee.
    """
    started = time.monotonic()
    metrics: dict[str, Any] = {}

    def finish(status: str, merges: list | None = None, reason: str | None = None):
        metrics["duration_seconds"] = round(time.monotonic() - started, 6)
        if reason:
            metrics["reason"] = reason
        return {"status": status, "merges": merges or [], "metrics": metrics}

    try:
        if not isinstance(url, str):
            return finish("skipped", reason="invalid_url")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return finish("skipped", reason="invalid_url")
        if parsed.username is not None or parsed.password is not None:
            return finish("skipped", reason="invalid_url")
        if not isinstance(model, str) or not model or len(model) > 128:
            return finish("skipped", reason="invalid_model")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            return finish("skipped", reason="invalid_timeout")
        if type(max_prompt_chars) is not int or max_prompt_chars <= 0:
            return finish("skipped", reason="invalid_prompt_limit")
        safe_input, by_id = _sanitize(table, word_boxes)
    except (_InvalidInput, ValueError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, _InvalidInput) else "invalid_input"
        return finish("skipped", reason=reason)

    metrics.update(
        input_cells=len(by_id),
        input_word_boxes=len(safe_input["word_boxes"]),
        schema_mode=bool(json_schema),
    )
    if len(by_id) < 2:
        return finish("skipped", reason="insufficient_cells")
    user_content = json.dumps(safe_input, separators=(",", ":"), ensure_ascii=True)
    prompt_chars = len(_SYSTEM) + len(user_content)
    metrics["prompt_chars"] = prompt_chars
    if prompt_chars > max_prompt_chars:
        return finish("skipped", reason="prompt_limit")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "format": _schema(list(by_id)) if json_schema else "json",
        "options": {"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        if not isinstance(api_key, str) or "\r" in api_key or "\n" in api_key:
            return finish("skipped", reason="invalid_api_key")
        headers["Authorization"] = "Bearer " + api_key
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with _open_request(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        metrics["http_status"] = exc.code
        return finish("error", reason="http_error")
    except (TimeoutError, socket.timeout):
        return finish("error", reason="timeout")
    except urllib.error.URLError as exc:
        reason = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "network_error"
        return finish("error", reason=reason)
    except (OSError, ValueError, TypeError):
        return finish("error", reason="request_error")

    if len(body) > MAX_RESPONSE_BYTES:
        return finish("rejected", reason="response_limit")
    try:
        envelope = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return finish("rejected", reason="invalid_response_json")
    if not isinstance(envelope, dict):
        return finish("rejected", reason="invalid_response")
    if envelope.get("done") is False or envelope.get("done_reason") in ("length", "max_tokens"):
        return finish("rejected", reason="truncated_response")
    for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration"):
        value = envelope.get(key)
        if type(value) is int and value >= 0:
            metrics[key] = value
    message = envelope.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return finish("rejected", reason="invalid_response")
    content = message["content"]
    if len(content) > MAX_CONTENT_CHARS:
        return finish("rejected", reason="output_limit")
    try:
        merges = _validate_proposal(json.loads(content), by_id)
    except json.JSONDecodeError:
        return finish("rejected", reason="invalid_proposal_json")
    except _InvalidProposal as exc:
        return finish("rejected", reason=str(exc))
    return finish("proposed" if merges else "no_change", merges)
