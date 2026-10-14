# Table Structure Lab

A small CPU-based starting point for recognizing table structure in PDFs: table regions, rows, columns, cell rectangles, and merged cells. It exports no cell text into JSON. It combines vector-bordered tables, closed-grid pixel detection, and an experimental OCR-free `img2table` layout fallback for partially ruled and borderless images.

The first pipeline is **PDF geometry -> grid reconstruction -> structure JSON**. The scan paths recover either closed cells from pixel lines or inferred rows/columns from `img2table` connected-component and whitespace layout. These paths run locally on the CPU with no OCR, generative model, or API key. An optional Ollama experiment can propose merges only for accepted text-alignment candidates. Baseline and model proposals remain separate.

## Start here

Use Python 3.10 or newer. In this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python table_structure.py /absolute/path/to/input.pdf --strategy auto --output runs/first
```

Use a new output folder each time. The input PDF is never modified.

For an existing installation, rerun `python -m pip install -r requirements.txt` to install `img2table==2.0.0` and its NumPy/OpenCV contrib dependencies. This is a larger dependency footprint than the original Pillow-only scan route. The tested local environment is Apple Silicon, Python 3.14, img2table 2.0.0, NumPy 2.5.4 and OpenCV contrib 5.0.0. These are compatibility results, not an energy benchmark.

The output folder contains:

- `structure.json`: each detected table's bounding box, row and column counts, cell rectangles, row spans, column spans, warnings, and run metrics.
- `page-001-baseline.png`, etc.: local images of the original pages with the detected cells outlined. Cyan means an ordinary cell; amber means a cell spanning multiple rows or columns.

The overlays **include the original page contents**. They stay local and are never sent to the model. Use `--no-overlays` for JSON only and to avoid rendering cost.

## Example structure

This cell starts in the first row and second column and spans two columns:

```json
{
  "id": "c0001",
  "row": 0,
  "column": 1,
  "row_span": 1,
  "column_span": 2,
  "bbox": [180.0, 140.0, 460.0, 180.0]
}
```

Coordinates are PDF points, normally 72 per inch, in pdfplumber's page coordinate system: `[left, top, right, bottom]`, with x increasing to the right and y downward. Page numbers are 1-based; row and column indices are 0-based. `page_bbox` and `rotation` record the page geometry. A complete detected grid is a consistency check, not proof that a candidate is a real table.

Schema version 1.1 adds scan diagnostics. For a tilted scan, `bbox` remains an enclosing rectangle in original page coordinates and `polygon` records its four original-page corners. `grid_transform` describes the applied straightening rotation; `x_boundaries` and `y_boundaries` refer to the straightened grid, as stated in `grid_coordinate_system`. Overlays use polygons for these cells so they follow the original tilt. Row and column indices are unchanged by the coordinate transform.

Schema version 1.2 additionally records input/code SHA-256 hashes, dependency versions, effective settings, per-backend diagnostics, accepted/retained counts, and rejected candidate boxes/reasons. Every returned table is `review_status: unreviewed`; geometric acceptance does not establish that it matches the original table. Source headers/notes and logical wrapped rows remain ambiguous.

## Borderless tables and page selection

```bash
python table_structure.py input.pdf --strategy text --pages 1 --bbox 50 125 500 290 --output runs/borderless
python table_structure.py input.pdf --pages 1,3-5 --no-overlays --output runs/selected
```

`--bbox` limits detection to a known table region. This is especially useful for borderless tables because aligned prose can look like a table.

- `lines`: use vector lines and rectangle edges. This is the recommended first experiment. Empty bordered cells are retained.
- `text`: experimental alignment-based detection. Whitespace-only bands between text lines are collapsed, dividers are placed in observed word gaps, and outer edges receive up to four points of padding. Candidates cutting too many words or resembling long prose columns are rejected. This can also reject ambiguous real tables; inspect the rejection diagnostics. Wrapped text can still become multiple rows, and blank borderless rows may disappear.
- `raster`: render the page and detect closed cell grids from image pixels. This can also be forced on digital PDFs to compare pixel detection with vector detection.
- `img2table`: use the pinned image-layout backend directly, with implicit rows, implicit columns, and borderless detection enabled. It receives RGB pixels only; OCR, document content extraction and generative models are never invoked.
- `auto` (default): try vector lines; on image-containing pages also try raster detection; use img2table when no accepted table exists or the page contains images; if still unresolved and selectable words exist, try gated text alignment. Rejected raw fragments cannot suppress fallback. Mixed vector/image pages retain distinct results and deduplicate overlapping or aligned subset rediscoveries. A failed image backend preserves earlier results and permits the next route. Borderless regions beside vector tables on purely digital pages still require a targeted `--bbox`/strategy run.

Candidates with inconsistent or incomplete grids are rejected. Inferred image candidates receive additional geometric evidence checks: graphic-dominated ink is rejected, and a layout with fewer than four recurring rows requires independent long-rule support. Image/prose combinations and unruled grids joining a wide long-prose column to short cells are also rejected. These conservative choices can reject valid small borderless tables or tables with narrative columns. They are tuned on a small regression set, not a calibrated table classifier; all results still require review.

## Scanned and image-based tables

```bash
python table_structure.py scan.pdf --strategy raster --output runs/scan
python table_structure.py scan.pdf --strategy raster --pages 3 --scan-dpi 144 --output runs/scan-page-3
python table_structure.py scan.pdf --strategy img2table --layout-dpi 200 --output runs/layout
```

The pixel route uses Pillow (already a pdfplumber dependency). It does not recognize or extract words. It thresholds grayscale pixels, estimates a straightening rotation within +/-2 degrees, detects near-solid horizontal and vertical runs, and joins nearby segments. Existing pdfplumber geometry routines reconstruct closed cells from their intersections. Cells are then exported through the same structure builder and checked for grid consistency.

Rendering defaults to 144 dpi with a maximum dimension of 2200 pixels. `--scan-dpi` accepts 72-200. Pixel counts, line counts, and intersection counts are bounded; overly complex inputs receive a warning instead of unbounded processing. `--no-overlays` avoids preview rendering, but scan analysis itself still requires rasterization.

The closed-grid route needs enough horizontal **and** vertical boundaries to form cells. The img2table fallback additionally uses connected components, text shapes, and whitespace to infer implicit rows/columns. It calls only the lower-level `TableExtractor`, retains unique merged rectangles and empty cells, and skips all OCR/content APIs. OpenCV parallel regions and OpenCL are disabled; actual runtime flags are recorded. There is no added img2table deskew step; PDF page rotation is already applied by rendering.

`--layout-dpi` defaults to 200 and accepts 72-300, with the same 2200-pixel maximum dimension. It is separate from the closed-grid route's `--scan-dpi`. Higher rendering resolution can improve small-font layout but is not universally better. Missing/incompatible native dependencies produce actionable diagnostics while other routes remain available. `needs_fallback` means no acceptable grid was recovered, not that the page has no tables.

Partially ruled scans can still produce split headers/body fragments, omitted columns, or extra rows from wrapped values. Detecting a plausible region is not recovering its complete logical structure. Caption/footnote separation and joining incompatible fragments are not implemented. Use overlays and a consistent annotation policy before reporting accuracy.

## Optional: your Ollama endpoint

Credentials are read only from environment variables. No live model calls were made while developing or testing it. There is no OpenAI API integration.

Set the URL and read your Ollama key into an environment variable without putting the literal value in shell history. On macOS/zsh:

```bash
export OLLAMA_CHAT_URL='http://localhost:11434/api/chat'
read -rs 'OLLAMA_API_KEY?Ollama key: '
export OLLAMA_API_KEY
python table_structure.py input.pdf --strategy text --pages 1 --bbox 50 125 500 290 --ollama-model 'llama3.1:latest' --output runs/llama
unset OLLAMA_API_KEY
```

For a hosted Ollama-compatible service, replace `OLLAMA_CHAT_URL` with its full chat endpoint URL. The default is `http://localhost:11434/api/chat`. The script does not contact anything unless `--ollama-model` is set and an eligible borderless table is detected. Compatibility with third-party proxies has not been verified.

Replace the model argument with `oss` or `gemma3.1` to compare your other aliases. Alias names alone do not reveal parameter count, quantization, price, or whether vision input is supported. This prototype uses text input containing geometry, so vision support is unnecessary for its optional merge experiment.

The request uses:

```json
{
  "stream": false,
  "options": {
    "temperature": 0,
    "seed": 42,
    "num_predict": 512,
    "num_ctx": 4096
  }
}
```

The `format` field supplies a JSON Schema. If your proxy rejects schema format, retry with `--json-mode` in a new output folder. Both modes validate the response locally. There is no automatic retry or fallback request, so a failed request does not silently trigger additional usage. Runtime option support depends on the server and model.

Only allowlisted cell IDs, row/column indices, spans, normalized bounding boxes, and word bounding boxes are sent. Document text, filenames, metadata, and page images are omitted. Word positions are derived using text segmentation locally; geometry still reveals layout and approximate word lengths.

The model can propose **merges of existing adjacent cells**, which helps test whether a small model can group text lines or cells more sensibly. It cannot discover a missed table, create new boundaries, split a cell, read a scan, or identify a semantic header from its meaning. Vector, raster and img2table candidates skip the model; only the text-route merge experiment is supported. Word positions are extracted lazily only when an eligible text candidate exists. Scan handling is independent local image processing.

Valid proposals appear in `model_candidate` and a separate `page-001-proposed.png`. The baseline stays intact. Invalid IDs, nonrectangular groups, overlapping merges, malformed output, and truncated responses are rejected. Structurally valid proposals can still be wrong and require visual comparison. Oversize inputs are skipped instead of truncated; model requests are capped by input, prompt, output, and timeout limits.

## Test and reproduce the examples

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python make_demo.py --output demo.pdf
python table_structure.py demo.pdf --strategy lines --pages 1,4,5 --output runs/demo-bordered
python table_structure.py demo.pdf --strategy text --pages 2 --bbox 50 125 500 290 --output runs/demo-borderless
python table_structure.py demo.pdf --strategy auto --pages 4 --output runs/demo-raster
python table_structure.py demo.pdf --strategy img2table --pages 4 --output runs/demo-img2table
```

The bundled `examples/` contains structure JSON and overlays from synthetic tests. Tests cover spans, empty cells, crops, rotation, borderless pixels, OCR/content isolation, optional dependency failures, pixel evidence, rejected-candidate fallback, backend failure isolation, mixed pages, subset deduplication, prose rejection, bounded work and mocked model contracts. These checks do not establish a representative real-document accuracy rate or a live model benefit.

The October 14 scan regression check passed 70 automatic tests and processed all 34 pages of four supplied PDFs. On the two image-only PDFs (18 pages), candidates occurred on all four table-bearing pages and none of the 14 negative pages. Six fragments represented five physical tables: split headers, missing columns, and caption/footnote leakage remain. The scans took 6.72 seconds including overlays, with a maximum single-process peak RSS of 416.56 MB on the tested Mac. No OCR or model calls were used. These documents informed the gates, so this is a small regression check rather than held-out accuracy or a low-power hardware benchmark. See [the recorded observations](docs/scan-validation-2026-10-14.txt).

`metrics.wall_seconds` includes detection and, if requested, rendering and model calls. Each page also records `detection_seconds`. `process_peak_rss_mb` measures the local Python process on macOS/Linux, including rendering; it does not measure the remote model's memory or power. Successful model responses can include server-reported token counts and durations in `model_review.metrics`. No electricity or API price estimate is inferred from these numbers.

## How to evaluate the idea

Start with 20-50 representative bordered tables. Draw the expected grid manually and compare exact row/column counts, cell rectangles, and merged spans with the overlays. Then add borderless examples and compare the same baseline against each model's separate proposals. Keep an annotated set of difficult cases apart for the final comparison.

Choose the cheapest method that meets your actual accuracy target. For clean bordered PDFs, that may be zero model calls. For ambiguous borderless tables, measure whether a model's merge proposals improve the grid enough to justify its latency and cost. These results will be specific to your documents.

## Limits and next stages

- Closed ruled scans have a pixel route; partially ruled/borderless scans have an experimental img2table layout fallback. Both omit OCR transcription, but inferred logical structure can still be incomplete or wrong.
- Wrapped text, implied empty cells, skewed tables, shading-only boundaries, and broken borders can produce wrong candidates.
- Single-row or single-column candidates are skipped to reduce false positives.
- Every page is processed independently. Tables continued across pages are not joined.
- Headers are not classified, and a missing border may look like a true merged cell. Inspect overlays rather than treating a valid grid as confirmed structure.

The next stage for unresolved scans is to reconcile fragments, improve implicit column/logical-row inference, or compare a specialist structure model on verified table crops. Benchmark against annotated real tables and negative pages before optimizing or quantizing it.

## Primary references

- [pdfplumber: table detection, geometry, and visual debugging](https://github.com/jsvine/pdfplumber)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)
- [Pillow image processing](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [img2table 2.0.0 image extractor](https://github.com/xavctn/img2table/blob/2.0.0/src/img2table/tables/extractor.py)
