# Table Structure Lab

A small CPU-based starting point for recognizing table structure in PDFs: table regions, rows, columns, cell rectangles, and merged cells. It exports no cell text into JSON. The first supported target is a machine-generated PDF with visible table borders.

The default pipeline is **PDF geometry -> grid reconstruction -> structure JSON**. No AI model or API key is needed. An optional Ollama experiment asks a model to propose cell merges in borderless tables using coordinates only. Baseline and model proposals remain separate.

## Start here

Use Python 3.10 or newer. In this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python table_structure.py /absolute/path/to/input.pdf --strategy lines --output runs/first
```

Use a new output folder each time. The input PDF is never modified.

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

## Borderless tables and page selection

```bash
python table_structure.py input.pdf --strategy text --pages 1 --bbox 50 125 500 290 --output runs/borderless
python table_structure.py input.pdf --pages 1,3-5 --no-overlays --output runs/selected
```

`--bbox` limits detection to a known table region. This is especially useful for borderless tables because aligned prose can look like a table.

- `lines`: use vector lines and rectangle edges. This is the recommended first experiment. Empty bordered cells are retained.
- `text`: experimental alignment-based detection. Whitespace-only bands between text lines are collapsed, column dividers are placed in observed word gaps, and outer edges receive up to four points of padding. Multiline text may still be mistaken for separate rows; blank borderless rows may disappear. These inferred cell edges are estimates.
- `auto` (default): try lines, then use text only if no line-based table candidates are found on that page. It can miss a borderless table beside a bordered table; use `text` with `--bbox` for that region.

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

The model can propose **merges of existing adjacent cells**, which helps test whether a small model can group text lines or cells more sensibly. It cannot discover a missed table, create new boundaries, split a cell, read a scan, or identify a semantic header from its meaning. Drawn-border tables skip the model entirely because their geometry already supplies the grid.

Valid proposals appear in `model_candidate` and a separate `page-001-proposed.png`. The baseline stays intact. Invalid IDs, nonrectangular groups, overlapping merges, malformed output, and truncated responses are rejected. Structurally valid proposals can still be wrong and require visual comparison. Oversize inputs are skipped instead of truncated; model requests are capped by input, prompt, output, and timeout limits.

## Test and reproduce the examples

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python make_demo.py --output demo.pdf
python table_structure.py demo.pdf --strategy lines --pages 1,4,5 --output runs/demo-bordered
python table_structure.py demo.pdf --strategy text --pages 2 --bbox 50 125 500 290 --output runs/demo-borderless
```

The bundled `examples/` contains the resulting structure JSON and overlays from local synthetic tests. The tests cover merged rows and columns, an empty bordered cell, neighboring tables, a cropped region, a rotated page, a simple borderless table, raster pages, geometry-only JSON, and the model adapter with mocked responses. They do not measure accuracy on real documents or real model performance.

`metrics.wall_seconds` includes detection and, if requested, rendering and model calls. Each page also records `detection_seconds`. `process_peak_rss_mb` measures the local Python process on macOS/Linux, including rendering; it does not measure the remote model's memory or power. Successful model responses can include server-reported token counts and durations in `model_review.metrics`. No electricity or API price estimate is inferred from these numbers.

## How to evaluate the idea

Start with 20-50 representative bordered tables. Draw the expected grid manually and compare exact row/column counts, cell rectangles, and merged spans with the overlays. Then add borderless examples and compare the same baseline against each model's separate proposals. Keep an annotated set of difficult cases apart for the final comparison.

Choose the cheapest method that meets your actual accuracy target. For clean bordered PDFs, that may be zero model calls. For ambiguous borderless tables, measure whether a model's merge proposals improve the grid enough to justify its latency and cost. These results will be specific to your documents.

## Limits and next stages

- Scanned or image-only tables need a vision/table-structure stage. It is **not implemented** in this first version. Image-only pages are marked `needs_vision`; mixed pages warn that raster tables may be missed. A scan with an OCR text layer can enter the experimental text route, but the original image geometry is not analyzed.
- Wrapped text, implied empty cells, skewed tables, shading-only boundaries, and broken borders can produce wrong candidates.
- Single-row or single-column candidates are skipped to reduce false positives.
- Every page is processed independently. Tables continued across pages are not joined.
- Headers are not classified, and a missing border may look like a true merged cell. Inspect overlays rather than treating a valid grid as confirmed structure.

The next stage for all PDF types would be: rasterize only candidate table regions, run a specialist structure model such as Table Transformer or TableFormer, and translate its output into the same JSON schema. Benchmark that vision stage against the geometry baseline before optimizing or quantizing it.

## Primary references

- [pdfplumber: table detection, geometry, and visual debugging](https://github.com/jsvine/pdfplumber)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)
