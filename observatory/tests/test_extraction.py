import hashlib
import io
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import sys

import pytest

from observatory import extraction
from observatory.research import extract_html, save_document, save_note
from observatory.store import Store


PASSAGE = "A model's report of experience must be compared with independent evidence and alternative explanations."


def pdf_bytes(text=PASSAGE * 4):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica"), NameObject("/Encoding"): NameObject("/WinAnsiEncoding")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    literal = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(("BT /F1 12 Tf 50 700 Td (" + literal + ") Tj ET").encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_html_keeps_sections_paragraphs_table_rows_formulas_and_code_without_fetching():
    html = """<body><nav>irrelevant navigation</nav><article><h1>Conflicting theories</h1>
        <p>First paragraph with <em>independent</em> evidence.</p><p>Second paragraph remains distinct.</p>
        <ul><li>Uncertainty</li><li>Alternative explanation</li></ul>
        <table><caption>Comparisons</caption><tr><th>Theory</th><th>Limit</th></tr>
        <tr><td>Theory A</td><td>Limited evidence</td></tr></table>
        <p>Equation: <math><semantics><mrow>x+y</mrow><annotation encoding="application/x-tex">x+y=z</annotation></semantics></math></p>
        <pre>if claim:\n    check_evidence()</pre><img src="http://127.0.0.1/private" alt="diagram">
        <span hidden>hidden material</span><!-- private comment --><script>untrusted code</script>
        </article><footer>publisher boilerplate</footer></body>"""
    result = extraction.extract_html_document(html)
    assert "Conflicting theories\n\nFirst paragraph with independent evidence.\n\nSecond paragraph" in result.text
    assert "Theory\tLimit\nTheory A\tLimited evidence" in result.text
    assert "x+y=z" in result.text
    assert "if claim:\n    check_evidence()" in result.text
    assert "\n- Uncertainty\n" in result.text
    for absent in ("irrelevant", "hidden material", "private comment", "untrusted code", "publisher boilerplate", "127.0.0.1"):
        assert absent not in result.text
    metadata = result.metadata()
    assert metadata["quality"] == "unverified" and metadata["structure_preserved"]
    assert metadata["metrics"]["tables"] == 1
    assert "visual_content_not_interpreted" in metadata["warnings"]


def test_ambiguous_math_and_merged_tables_need_fidelity_review():
    result = extraction.extract_html_document("<article><p>" + PASSAGE * 4 +
        "</p><math><mi>x</mi><mo>+</mo><mi>y</mi></math><table><tr><td colspan='2'>Merged</td></tr></table></article>")
    assert result.metadata()["quality"] == "unverified"
    assert set(result.metadata()["warnings"]) >= {"mathml_linearized_requires_formula_review", "merged_table_cells_require_structure_review"}


@pytest.mark.parametrize("text,warning", [
    ("Too little text", "insufficient_readable_text"),
    ("123 + ! " * 100, "low_alphabetic_content_requires_manual_reextraction"),
    (PASSAGE * 4 + "\ufffd" * 40, "replacement_characters_indicate_encoding_loss"),
])
def test_readability_failures_stay_poor_even_when_review_requested(text, warning):
    result = extraction.Extraction(text, "pdf", "fixture", needs_fidelity_review=True)
    assert result.metadata()["quality"] == "poor"
    assert warning in result.metadata()["warnings"]


def test_saved_text_retains_structure_and_literal_note_matching(tmp_path):
    html = "<article><h1>Evidence</h1><p>" + PASSAGE + "</p><p>" + PASSAGE * 3 + "</p></article>"
    result = extraction.extract_html_document(html)
    assert extract_html(html) == (result.text, "article")
    store = Store(tmp_path / "state.db")
    try:
        source = save_document(store, url="https://example.org/paper", title="Evidence", text=result.text,
                               html=html, agent_id="scholar", scope=result.scope, extraction=result)
        assert "\n\n" in source["text"]
        assert source["content_hash"] == hashlib.sha256(result.text.encode()).hexdigest()
        assert source["provenance"]["extraction_method"] == "html_structured"
        assert source["extraction"]["quality"] == "passed"
        note = save_note(store, source, "scholar", "Compare independent evidence", PASSAGE.replace(" ", "\n"))
        assert note["support_verified"]
        with pytest.raises(ValueError, match="provenance"):
            save_document(store, url="https://example.org/other", title="Mismatch", text="unrelated" * 50,
                          agent_id="scholar", extraction=result)
    finally:
        store.close()


def test_supplied_text_is_not_falsely_certified_by_unrelated_markup(tmp_path):
    store = Store(tmp_path / "state.db")
    try:
        source = save_document(store, url="https://example.org/paper", title="Evidence", text=PASSAGE * 4,
                               html="<article>Different source</article>", agent_id="scholar")
        assert source["extraction"]["quality"] == "unverified"
        assert source["extraction"]["method"] == "supplied_text"
    finally:
        store.close()


def test_plain_text_preserves_document_boundaries():
    result = extraction.extract_plain_document("Heading\r\n\r\n" + PASSAGE + "\r\n\r\n" + PASSAGE * 3)
    assert result.metadata()["quality"] == "passed"
    assert "Heading\n\n" in result.text


def test_real_pdf_fallback_is_readable_but_discovery_only(monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PDF_EXTRACTOR", "pypdf")
    data = pdf_bytes()
    result = extraction.extract_pdf_document(data)
    assert PASSAGE in result.text
    metadata = result.metadata()
    assert metadata["quality"] == "unverified"
    assert metadata["method"] == "pypdf_layout_fallback"
    assert not metadata["structure_preserved"]
    assert metadata["metrics"]["input_sha256"] == hashlib.sha256(data).hexdigest()
    assert "fallback_reading_order_tables_and_formulas_unverified" in metadata["warnings"]


@pytest.mark.parametrize("data", [b"https://example.org/paper.pdf", b"not a PDF", "%PDF"])
def test_pdf_extractor_accepts_only_bounded_downloaded_pdf_bytes(data):
    with pytest.raises(ValueError, match="downloaded PDF bytes"):
        extraction.extract_pdf_document(data)


def test_pdf_size_page_and_encryption_limits(monkeypatch):
    monkeypatch.setattr(extraction, "MAX_DOCUMENT_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        extraction.extract_pdf_document(b"%PDF" + b"x" * 11)
    monkeypatch.setattr(extraction, "MAX_DOCUMENT_BYTES", 12_000_000)
    monkeypatch.setattr(extraction, "MAX_PDF_PAGES", 0)
    with pytest.raises(ValueError, match="page count"):
        extraction.extract_pdf_document(pdf_bytes())
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("fixture-password")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="Encrypted"):
        extraction.extract_pdf_document(output.getvalue())


def test_docling_missing_assets_does_not_download_and_falls_back(monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PDF_EXTRACTOR", "docling")
    monkeypatch.delenv("OBSERVATORY_DOCLING_ARTIFACTS_PATH", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Docling must not run without operator-provisioned assets")
    monkeypatch.setattr(extraction, "_docling_pdf_bytes", forbidden)
    result = extraction.extract_pdf_document(pdf_bytes())
    assert result.metadata()["quality"] == "unverified"
    assert "docling_local_artifacts_not_configured" in result.warnings


def test_docling_local_adapter_output_still_requires_fidelity_review(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PDF_EXTRACTOR", "docling")
    monkeypatch.setenv("OBSERVATORY_DOCLING_ARTIFACTS_PATH", str(tmp_path))
    monkeypatch.setenv("OBSERVATORY_DOCLING_OCR_PATH", str(tmp_path))
    monkeypatch.setenv("OBSERVATORY_DOCLING_FORMULAS", "1")
    calls = []
    def local(data, artifacts, ocr, *, formulas):
        calls.append((data, artifacts, ocr, formulas))
        return "## Heading\n\n" + PASSAGE * 4 + "\n\n| Theory | Counterargument |"
    monkeypatch.setattr(extraction, "_docling_pdf_bytes", local)
    result = extraction.extract_pdf_document(pdf_bytes())
    assert calls[0][1:] == (tmp_path, tmp_path, True)
    assert result.metadata()["quality"] == "unverified"
    assert result.structure_preserved and result.method == "docling_local_layout_ocr"


def test_docling_failure_returns_tagged_fallback_not_silent_success(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PDF_EXTRACTOR", "docling")
    monkeypatch.setenv("OBSERVATORY_DOCLING_ARTIFACTS_PATH", str(tmp_path))
    monkeypatch.delenv("OBSERVATORY_DOCLING_OCR_PATH", raising=False)
    def fail(*args, **kwargs):
        raise RuntimeError("offline models are unavailable")
    monkeypatch.setattr(extraction, "_docling_pdf_bytes", fail)
    result = extraction.extract_pdf_document(pdf_bytes())
    assert result.method == "pypdf_layout_fallback"
    assert "docling_unavailable_or_incomplete_used_pypdf" in result.warnings


def test_docling_child_receives_local_files_and_offline_environment(tmp_path, monkeypatch):
    calls = []
    def process(command, **kwargs):
        calls.append((command, kwargs))
        assert Path(command[4]).read_bytes().startswith(b"%PDF")
        Path(command[5]).write_text(json.dumps({"status": "success", "text": PASSAGE * 4}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(extraction.subprocess, "run", process)
    monkeypatch.setenv("HTTPS_PROXY", "http://private-proxy.example")
    monkeypatch.setenv("OPENAI_API_KEY", "owner-secret-fixture")
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "wallet-secret-fixture")
    assert extraction._docling_pdf_bytes(pdf_bytes(), tmp_path, None, formulas=False) == PASSAGE * 4
    command, kwargs = calls[0]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert not any(key.lower() == "https_proxy" for key in kwargs["env"])
    assert "OPENAI_API_KEY" not in kwargs["env"] and "SOLANA_PRIVATE_KEY" not in kwargs["env"]
    assert kwargs["timeout"] == 120
    assert not any(argument.startswith(("http:", "https:")) for argument in command)


def test_docling_worker_disables_remote_services_downloads_and_network(tmp_path, monkeypatch):
    # Fake SDK exercises our configuration contract without installing models.
    options_seen, convert_seen = {}, {}
    class Options:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            options_seen.update(kwargs)
    class Converter:
        def __init__(self, **kwargs):
            convert_seen.update(kwargs)
        def convert(self, source, **kwargs):
            convert_seen.update(source=source, limits=kwargs)
            return SimpleNamespace(status=SimpleNamespace(value="success"),
                document=SimpleNamespace(export_to_markdown=lambda: PASSAGE * 4))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf")))
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", SimpleNamespace(EasyOcrOptions=Options, PdfPipelineOptions=Options))
    monkeypatch.setitem(sys.modules, "docling.document_converter", SimpleNamespace(DocumentConverter=Converter, PdfFormatOption=Options))
    # Record originals for restoration after the worker's deliberate mutation.
    for target, name in ((socket, "create_connection"), (socket.socket, "connect"), (socket.socket, "connect_ex"),
                         (socket, "getaddrinfo"), (socket.socket, "sendto")):
        monkeypatch.setattr(target, name, getattr(target, name))
    source, output = tmp_path / "document.pdf", tmp_path / "output.json"
    source.write_bytes(pdf_bytes())
    extraction._run_docling_worker([str(source), str(output), str(tmp_path), str(tmp_path), "1"])
    assert options_seen["enable_remote_services"] is False
    assert options_seen["download_enabled"] is False
    assert options_seen["do_picture_description"] is False
    assert options_seen["do_formula_enrichment"] is True
    assert convert_seen["allowed_formats"] == ["pdf"]
    assert convert_seen["source"] == source
    with pytest.raises(OSError, match="Network access is disabled"):
        socket.create_connection(("127.0.0.1", 80))
    assert json.loads(output.read_text())["status"] == "success"


def test_empty_docling_output_is_not_marked_as_structured_success(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PDF_EXTRACTOR", "docling")
    monkeypatch.setenv("OBSERVATORY_DOCLING_ARTIFACTS_PATH", str(tmp_path))
    monkeypatch.delenv("OBSERVATORY_DOCLING_OCR_PATH", raising=False)
    monkeypatch.setattr(extraction, "_docling_pdf_bytes", lambda *args, **kwargs: "")
    result = extraction.extract_pdf_document(pdf_bytes())
    assert result.method == "pypdf_layout_fallback" and not result.structure_preserved
    assert "docling_empty_output_used_pypdf" in result.warnings
