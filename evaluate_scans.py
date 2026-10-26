#!/usr/bin/env python3
"""Compare saved structure JSON with geometry-only manual scan annotations."""
import argparse
import json
from pathlib import Path


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def evaluate(reference, folder):
    documents, checks = [], []
    for doc in reference["documents"]:
        report = json.loads((folder / Path(doc["filename"]).stem / "structure.json").read_text())
        if doc.get("input_sha256") and report["input_sha256"] != doc["input_sha256"]:
            raise ValueError("Input PDF hash differs from the annotated document")
        pages = {page["page"]: page for page in report["pages"]}
        if set(pages) != set(range(1, doc["page_count"] + 1)):
            raise ValueError("Evaluation requires every page of each annotated scan")
        found = [number for number, page in pages.items() if page["tables"]]
        negative_pages = [item["page"] for item in doc["negative_pages"]]
        table_results, matched_ids = [], set()
        for expected in doc["tables"]:
            box = expected["bbox_points"]
            candidates = [table for table in pages[expected["page"]]["tables"]
                if intersection(table["bbox"], box) > .05 * (box[2] - box[0]) * (box[3] - box[1])
                and intersection(table["bbox"], box) > .5 *
                (table["bbox"][2] - table["bbox"][0]) * (table["bbox"][3] - table["bbox"][1])]
            matched_ids.update((expected["page"], table["id"]) for table in candidates)
            single = len(candidates) == 1
            table = candidates[0] if single else None
            dimensions = single and table["row_count"] == expected["logical_rows"] and \
                table["column_count"] == expected["logical_columns"]
            spans = single and all(any(
                cell["row"] == span["row"] and cell["column"] == span["column"] and
                cell["row_span"] == span["rowspan"] and cell["column_span"] == span["colspan"]
                for cell in table["cells"]) for span in expected["required_header_spans"])
            tolerance = reference["evaluation_policy"]["bbox_tolerance_points"]
            box_match = single and all(abs(a - b) <= tolerance for a, b in zip(table["bbox"], box))
            result = {"reference_id": expected["id"], "page": expected["page"],
                "expected_grid": [expected["logical_rows"], expected["logical_columns"]],
                "candidate_grids": [[item["row_count"], item["column_count"]] for item in candidates],
                "candidate_ids": [item["id"] for item in candidates],
                "one_candidate": single, "dimensions_match": dimensions,
                "required_header_spans_match": spans, "bbox_within_tolerance": box_match,
                "reference_checks_passed": bool(dimensions and spans and box_match)}
            table_results.append(result)
            checks.append(result)
        unmatched = [{"page": number, "id": table["id"], "bbox": table["bbox"]}
            for number, page in pages.items() for table in page["tables"]
            if (number, table["id"]) not in matched_ids]
        documents.append({"filename": doc["filename"], "input_sha256": report["input_sha256"],
            "code_sha256": report["code_sha256"], "metrics": report["metrics"],
            "candidate_pages": found, "expected_table_pages": doc["table_pages"],
            "missed_table_pages": sorted(set(doc["table_pages"]) - set(found)),
            "negative_pages_with_candidates": sorted(set(found) & set(negative_pages)),
            "unmatched_candidates": unmatched,
            "tables": table_results})
    return {"documents": documents, "summary": {
        "scan_pages": sum(doc["page_count"] for doc in reference["documents"]),
        "reference_tables": len(checks),
        "candidate_fragments": sum(doc["metrics"]["table_candidates"] for doc in documents),
        "tables_matching_dimensions": sum(item["dimensions_match"] for item in checks),
        "tables_passing_reference_checks": sum(item["reference_checks_passed"] for item in checks),
        "missed_table_pages": sum(len(doc["missed_table_pages"]) for doc in documents),
        "negative_pages_with_candidates": sum(len(doc["negative_pages_with_candidates"]) for doc in documents),
        "unmatched_candidate_count": sum(len(doc["unmatched_candidates"]) for doc in documents),
        "wall_seconds_including_overlays": round(sum(doc["metrics"]["wall_seconds"] for doc in documents), 4),
        "maximum_process_peak_rss_mb": max(doc["metrics"]["process_peak_rss_mb"] for doc in documents)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
        default=Path(__file__).parent / "tests/fixtures/scan-reference.json")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    result = {"evaluation": "small_sample_geometry_regression", "current": evaluate(reference, args.runs),
        "limits": ["These documents informed implementation; this is not a held-out benchmark.",
            "Checks cover logical dimensions, required grouped-header spans and approximate outer bounds.",
            "Every cell boundary, arbitrary semantic merges and unseen layouts are not certified.",
            "No low-power hardware or electricity measurement was performed."]}
    if args.baseline:
        result["baseline"] = evaluate(reference, args.baseline)
        if [d["input_sha256"] for d in result["current"]["documents"]] != \
                [d["input_sha256"] for d in result["baseline"]["documents"]]:
            raise ValueError("Baseline and current input PDF hashes differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["current"]["summary"], indent=2))


if __name__ == "__main__":
    main()
