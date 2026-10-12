#!/usr/bin/env python3
"""Generate synthetic validation fixtures; requires requirements-dev.txt."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw


def make_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 612, 792
    document = canvas.Canvas(str(path), pagesize=(width, height))
    document.setTitle("Synthetic table structure fixtures")

    def text(x, top, value, size=11):
        document.setFont("Helvetica", size)
        document.drawString(x, height - top, value)

    def line(x0, y0, x1, y1):
        document.line(x0, height - y0, x1, height - y1)

    def title(value, subtitle):
        document.setFillColorRGB(0.08, 0.15, 0.25)
        text(54, 62, value, 22)
        text(54, 87, subtitle, 10)
        document.setStrokeColorRGB(0.3, 0.4, 0.5)
        document.setLineWidth(0.8)

    def small_grid(x, top):
        for xx in (x, x + 70, x + 140):
            line(xx, top, xx, top + 80)
        for yy in (top, top + 40, top + 80):
            line(x, yy, x + 140, yy)
        for row in range(2):
            for column in range(2):
                text(x + 8 + 70 * column, top + 24 + 40 * row, f"{row + 1}:{column + 1}")

    title("Table structure / bordered", "Cyan outlines mark cells. Amber outlines mark merged cells.")
    # Four rows, three columns. Top-right spans two columns;
    # first column in rows 1-2 spans two rows. Last cell is empty.
    for xx in (60, 180, 460):
        line(xx, 140, xx, 300)
    line(320, 180, 320, 300)
    for yy in (140, 180, 260, 300):
        line(60, yy, 460, yy)
    line(180, 220, 460, 220)
    for x, y, value in [(70, 165, "Category"), (195, 165, "Merged header"),
        (70, 220, "Merged rows"), (190, 205, "A"), (330, 205, "B"),
        (190, 245, "C"), (330, 245, "D"), (70, 285, "Last row"), (190, 285, "E")]:
        text(x, y, value)
    text(60, 347, "Two separate tables", 13)
    small_grid(60, 380)
    small_grid(320, 380)
    document.showPage()

    title("Table structure / borderless", "Experimental: positions reveal alignment, but wrapped rows remain ambiguous.")
    data = [("Item", "Count", "Amount"), ("Apples", "12", "24.00"),
            ("Pears", "8", "16.00"), ("Oranges", "10", "30.00")]
    for row, values in enumerate(data):
        for x, value in zip((60, 240, 400), values):
            text(x, 150 + row * 38, value)
    document.showPage()

    title("Table structure / scanned", "Image-only table: this first version should flag that a vision stage is needed.")
    image = Image.new("RGB", (840, 320), "white")
    draw = ImageDraw.Draw(image)
    for x in (0, 280, 560, 839):
        draw.line((x, 0, x, 319), fill="black", width=3)
    for y in (0, 80, 160, 240, 319):
        draw.line((0, y, 839, y), fill="black", width=3)
    for row in range(4):
        for column in range(3):
            draw.text((column * 280 + 24, row * 80 + 30), f"scan {row}/{column}", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    document.drawImage(ImageReader(buffer), 60, height - 300, width=420, height=160)
    # This page intentionally has some title text, so the image-only table is
    # a mixed page, not proof that selectable text means all tables are digital.
    document.showPage()

    document.drawImage(ImageReader(buffer), 60, height - 300, width=420, height=160)
    document.showPage()

    document.setPageRotation(90)
    # ReportLab swaps the media dimensions for a 90-degree rotated page.
    height = 612
    small_grid(60, 140)
    document.showPage()
    document.setPageRotation(0)
    document.save()
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo.pdf")
    arguments = parser.parse_args()
    print(make_demo(arguments.output))
