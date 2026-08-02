#!/usr/bin/env python3
"""
Tests for structured-document extraction in the read_file tool.

Covers .pdf / .ipynb / .docx / .xlsx extraction (the Office/notebook
cases were ported from Kilo-Org/kilocode #10733, #10737, #10740) and the
read_file_tool integration: pagination, line-numbering, OCR routing, protected
path enforcement, graceful fallback on malformed input, and hidden-sheet
omission.

Run with:  python -m pytest tests/tools/test_read_extract.py -v
"""

import json
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

from tools.read_extract import (
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)
from tools.file_tools import read_file_tool


# ---------------------------------------------------------------------------
# Fixture builders — construct minimal valid OOXML / notebook files.
# ---------------------------------------------------------------------------

def _write_notebook(path, cells, nbformat=4):
    nb = {"cells": cells, "metadata": {}, "nbformat": nbformat, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh)


def _write_docx(path, document_xml):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", document_xml)


def _write_xlsx(path, *, workbook, rels, shared, sheets):
    """sheets: dict of part-name -> xml string."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", shared)
        for part, xml in sheets.items():
            z.writestr(part, xml)


def _write_text_pdf(path, text="Hello PDF"):
    """Write a minimal one-page PDF with native Helvetica text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    cursor = len(chunks[0])
    for number, body in enumerate(objects, 1):
        offsets.append(cursor)
        chunk = f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        chunks.append(chunk)
        cursor += len(chunk)
    xref = cursor
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    chunks.extend(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:])
    chunks.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    with open(path, "wb") as fh:
        fh.write(b"".join(chunks))


def _write_image_pdf(path):
    """Write a minimal one-page image-only PDF."""
    image = b"\xff\xff\xff"
    content = b"q 100 0 0 100 72 650 cm /Im1 Do Q"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>"),
        (b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
         b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\nstream\n"
         + image + b"\nendstream"),
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    cursor = len(chunks[0])
    for number, body in enumerate(objects, 1):
        offsets.append(cursor)
        chunk = f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        chunks.append(chunk)
        cursor += len(chunk)
    xref = cursor
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    chunks.extend(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:])
    chunks.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    with open(path, "wb") as fh:
        fh.write(b"".join(chunks))


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# ---------------------------------------------------------------------------
# is_extractable_document
# ---------------------------------------------------------------------------

class TestIsExtractable(unittest.TestCase):
    def test_recognized_extensions(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("/x/B.DOCX"))
        self.assertTrue(is_extractable_document("report.xlsx"))
        self.assertTrue(is_extractable_document("paper.PDF"))

    def test_unrecognized_extensions(self):
        self.assertFalse(is_extractable_document("a.py"))
        self.assertFalse(is_extractable_document("a.txt"))


# ---------------------------------------------------------------------------
# PDF documents
# ---------------------------------------------------------------------------

class TestPdfExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_pdf_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_native_text_is_extracted_as_markdown(self):
        p = os.path.join(self.tmp, "paper.pdf")
        _write_text_pdf(p, "Quarterly revenue")

        text = extract_document_text(p)

        self.assertIn("Quarterly revenue", text)

    def test_mixed_pdf_marks_pages_that_still_need_ocr(self):
        p = os.path.join(self.tmp, "mixed.pdf")
        _write_text_pdf(p)
        result = SimpleNamespace(
            markdown="# Native page",
            pdf_type="mixed",
            pages_needing_ocr=[2, 5],
            has_encoding_issues=False,
        )

        with mock.patch("pdf_inspector.process_pdf", return_value=result):
            text = extract_document_text(p)

        self.assertIn("Pages requiring OCR: 2, 5", text)
        self.assertIn("# Native page", text)

    def test_broken_font_encoding_recommends_ocr_fallback(self):
        p = os.path.join(self.tmp, "encoded.pdf")
        _write_text_pdf(p)
        result = SimpleNamespace(
            markdown="garbled text",
            pdf_type="text_based",
            pages_needing_ocr=[],
            has_encoding_issues=True,
        )

        with mock.patch("pdf_inspector.process_pdf", return_value=result):
            text = extract_document_text(p)

        self.assertIn("Encoding issues detected; OCR fallback recommended", text)


# ---------------------------------------------------------------------------
# Notebooks (.ipynb) — #10733
# ---------------------------------------------------------------------------

class TestNotebookExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_nb_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_markdown_and_code_in_order(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": ["# Title\n", "para"]},
            {"cell_type": "code", "source": "x = 1\nprint(x)",
             "outputs": [{"output_type": "stream", "text": ["1\n"]}],
             "execution_count": 1},
        ])
        text = extract_document_text(p)
        self.assertIn("# Title", text)
        self.assertIn("print(x)", text)
        # Output payloads must NOT leak into the extracted text.
        self.assertNotIn("output_type", text)
        self.assertNotIn("execution_count", text)
        # Order preserved: markdown before code.
        self.assertLess(text.index("Title"), text.index("print(x)"))


    def test_empty_cells_raises(self):
        p = os.path.join(self.tmp, "empty.ipynb")
        _write_notebook(p, [])
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# Word documents (.docx) — #10737
# ---------------------------------------------------------------------------

class TestDocxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_docx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _doc(self, body):
        return (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                f'<w:body>{body}</w:body></w:document>')

    def test_paragraphs_and_runs(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, self._doc(
            '<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>World</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Second</w:t></w:r></w:p>'))
        text = extract_document_text(p)
        self.assertIn("Hello World", text)
        self.assertIn("Second", text)


    def test_missing_document_xml_raises(self):
        p = os.path.join(self.tmp, "nodoc.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("other.xml", "<x/>")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# Excel workbooks (.xlsx) — #10740
# ---------------------------------------------------------------------------

class TestXlsxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_xlsx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, path, *, include_hidden=True):
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        hidden_sheet = (f'<sheet name="Hidden" sheetId="2" state="hidden" '
                        f'xmlns:r="{r}" r:id="rId2"/>') if include_hidden else ""
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{r}"><sheets>'
            f'<sheet name="Data" sheetId="1" r:id="rId1"/>{hidden_sheet}'
            f'</sheets></workbook>')
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
            '</Relationships>')
        shared = (f'<sst xmlns="{_NS_S}"><si><t>Name</t></si><si><t>Score</t></si>'
                  f'<si><t>Alice</t></si></sst>')
        sheet1 = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>95</v></c></row>'
            '</sheetData></worksheet>')
        sheet2 = (f'<worksheet xmlns="{_NS_S}"><sheetData>'
                  '<row r="1"><c r="A1" t="str"><v>SECRETDATA</v></c></row>'
                  '</sheetData></worksheet>')
        _write_xlsx(path, workbook=workbook, rels=rels, shared=shared,
                    sheets={"xl/worksheets/sheet1.xml": sheet1,
                            "xl/worksheets/sheet2.xml": sheet2})

    def test_visible_sheet_content(self):
        p = os.path.join(self.tmp, "wb.xlsx")
        self._build(p)
        text = extract_document_text(p)
        self.assertIn("Data", text)        # sheet label
        self.assertIn("Name\tScore", text)  # shared-string header row
        self.assertIn("Alice\t95", text)    # string + numeric cells


    def test_not_a_zip_raises(self):
        p = os.path.join(self.tmp, "bad.xlsx")
        with open(p, "wb") as fh:
            fh.write(b"nope")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)


# ---------------------------------------------------------------------------
# read_file_tool integration
# ---------------------------------------------------------------------------

class TestReadFileToolIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_int_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_notebook_read_is_line_numbered(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "# H"},
            {"cell_type": "code", "source": "print(1)"},
        ])
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("1|", res["content"])  # line-number gutter
        self.assertIn("print(1)", res["content"])


    def test_corrupt_docx_falls_through_to_binary_guard(self):
        p = os.path.join(self.tmp, "bad.docx")
        with open(p, "wb") as fh:
            fh.write(b"not a zip")
        res = json.loads(read_file_tool(p))
        # Should NOT crash; falls through to the binary-extension guard.
        self.assertIn("error", res)
        self.assertIn("binary", res["error"].lower())

    def test_docx_read_extracts(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                        '<w:body><w:p><w:r><w:t>Report body</w:t></w:r></w:p>'
                        '</w:body></w:document>'))
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("Report body", res["content"])

    def test_image_pdf_reports_ocr_requirement(self):
        p = os.path.join(self.tmp, "scan.pdf")
        _write_image_pdf(p)

        res = json.loads(read_file_tool(p))

        self.assertIn("error", res)
        self.assertIn("OCR", res["error"])
        self.assertTrue(res.get("requires_ocr"))

    def test_corrupt_pdf_returns_extraction_error_without_raw_bytes(self):
        p = os.path.join(self.tmp, "broken.pdf")
        with open(p, "wb") as fh:
            fh.write(b"%PDF-1.4\nnot a valid document\n")

        res = json.loads(read_file_tool(p))

        self.assertIn("error", res)
        self.assertIn("PDF extraction failed", res["error"])
        self.assertNotIn("%PDF-1.4", res.get("content", ""))

    def test_pdf_extraction_cannot_bypass_protected_path_guard(self):
        hermes_home = os.path.join(self.tmp, "home")
        p = os.path.join(hermes_home, "skills", ".hub", "payload.pdf")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _write_text_pdf(p, "UNTRUSTED HUB PDF PAYLOAD")

        with mock.patch.dict(os.environ, {"HERMES_HOME": hermes_home}):
            res = json.loads(read_file_tool(p))

        self.assertIn("error", res)
        self.assertNotIn("UNTRUSTED HUB PDF PAYLOAD", res.get("content", ""))


if __name__ == "__main__":
    unittest.main()
