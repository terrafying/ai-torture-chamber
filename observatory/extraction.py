"""Bounded, structured original-text extraction; never a claim-validity check.

HTML parsing reads supplied markup only. Optional PDF layout/OCR runs in an
offline child process against supplied PDF bytes and preinstalled local models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCHEMA_VERSION = "structured-extraction-v1"
MAX_DOCUMENT_BYTES = 12_000_000
MAX_PDF_PAGES = 100
MAX_EXTRACTED_CHARACTERS = 2_000_000


def structured_text(value: str) -> str:
    """Retain paragraph, row and code boundaries while removing unsafe controls."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in value.split("\n"))).strip()


@dataclass
class Extraction:
    text: str
    scope: str
    method: str
    structure_preserved: bool = False
    warnings: list[str] = field(default_factory=list)
    needs_fidelity_review: bool = False
    extra_metrics: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        text = self.text
        visible = [character for character in text if not character.isspace()]
        alphabetic_ratio = sum(character.isalpha() for character in visible) / max(1, len(visible))
        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        warnings = list(self.warnings)
        poor = False
        if len(text.strip()) < 200:
            warnings.append("insufficient_readable_text")
            poor = True
        if alphabetic_ratio < 0.35:
            warnings.append("low_alphabetic_content_requires_manual_reextraction")
            poor = True
        if replacement_ratio > 0.01:
            warnings.append("replacement_characters_indicate_encoding_loss")
            poor = True
        if len(text) > MAX_EXTRACTED_CHARACTERS:
            warnings.append("extracted_text_limit_exceeded")
            poor = True
        quality = "poor" if poor else "unverified" if self.needs_fidelity_review else "passed"
        return {"schema_version": SCHEMA_VERSION, "method": self.method, "quality": quality,
                "structure_preserved": self.structure_preserved, "warnings": list(dict.fromkeys(warnings)),
                "metrics": {"characters": len(text), "paragraphs": len(text.split("\n\n")),
                            "alphabetic_ratio": round(alphabetic_ratio, 4),
                            "replacement_ratio": round(replacement_ratio, 4), **self.extra_metrics}}


def extract_html_document(html: str) -> Extraction:
    """Keep headings, paragraphs, lists, tables and available formula text.

    No URL fetch, browser rendering, external image loading or script execution
    occurs. Image-only diagrams and ambiguous math receive visible QC warnings.
    """
    from bs4 import BeautifulSoup, Comment, NavigableString, Tag
    if len(html.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ValueError("HTML exceeds the bounded extraction size")
    soup = BeautifulSoup(html, "html.parser")
    warnings: list[str] = []
    for script in soup.find_all("script"):
        if str(script.get("type", "")).startswith("math/tex"):
            script.replace_with(" " + script.get_text() + " ")
        else:
            script.decompose()
    for tag in soup.select("style, nav, footer, aside, form, template, noscript, [hidden], [aria-hidden='true'], [role='complementary'], .comments, #comments, .related-articles"):
        tag.decompose()
    articles = soup.find_all("article")
    scope = "article" if len(articles) == 1 else "main" if soup.find("main") else "body"
    root = articles[0] if scope == "article" else soup.find(scope) or soup
    if len(articles) > 1:
        warnings.append("multiple_articles_require_source_scope_review")
    if root.find("img") or root.find("svg") or root.find("canvas"):
        warnings.append("visual_content_not_interpreted")
    block_tags = {"p", "div", "section", "article", "main", "body", "blockquote", "figure", "figcaption",
                  "header", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt", "dd", "ul", "ol"}

    def render(node) -> str:
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node))
        if not isinstance(node, Tag):
            return ""
        if node.name in {"svg", "canvas", "img", "iframe", "object", "embed"}:
            return ""
        if node.name == "br":
            return "\n"
        if node.name == "pre":
            return "\n\n" + node.get_text().strip("\n") + "\n\n"
        if node.name == "math":
            annotation = node.find("annotation", attrs={"encoding": re.compile("(?:tex|latex)", re.I)})
            if annotation:
                return " " + annotation.get_text("", strip=True) + " "
            warnings.append("mathml_linearized_requires_formula_review")
            return " " + node.get_text(" ", strip=True) + " "
        if node.name == "table":
            rows = []
            caption = node.find("caption")
            if caption:
                rows.append(caption.get_text(" ", strip=True))
            for row in node.find_all("tr"):
                cells = row.find_all(["th", "td"], recursive=False)
                if cells:
                    rows.append("\t".join(" ".join(render(cell).split()) for cell in cells))
                    if any(cell.has_attr("rowspan") or cell.has_attr("colspan") for cell in cells):
                        warnings.append("merged_table_cells_require_structure_review")
            return "\n\n" + "\n".join(rows) + "\n\n"
        content = "".join(render(child) for child in node.children)
        if node.name == "li":
            return "\n- " + content.strip() + "\n"
        if node.name in block_tags:
            return "\n\n" + content.strip() + "\n\n"
        return content

    text = structured_text(render(root))
    return Extraction(text, scope, "html_structured", True, warnings,
                      needs_fidelity_review=any(item in warnings for item in (
                          "mathml_linearized_requires_formula_review", "merged_table_cells_require_structure_review",
                          "multiple_articles_require_source_scope_review", "visual_content_not_interpreted")),
                      extra_metrics={"headings": len(root.find_all(re.compile(r"^h[1-6]$"))),
                                     "tables": len(root.find_all("table"))})


def extract_plain_document(text: str) -> Extraction:
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ValueError("Text exceeds the bounded extraction size")
    return Extraction(structured_text(text), "plain_text", "plain_text", True)


def _docling_pdf_bytes(data: bytes, artifacts_path: Path, ocr_path: Path | None, *, formulas: bool) -> str:
    """Child isolation avoids changing environment/socket state in other workers."""
    with tempfile.TemporaryDirectory(prefix="observatory-extraction-") as temporary:
        root = Path(temporary)
        source, output = root / "document.pdf", root / "extracted.json"
        source.write_bytes(data)
        # Extraction has no reason to receive owner, provider or wallet secrets.
        allowed_environment = {"path", "systemroot", "windir", "temp", "tmp", "userprofile", "homedrive", "homepath",
            "appdata", "localappdata", "programdata", "home", "lang", "lc_all", "pythonpath", "pythonutf8",
            "cuda_visible_devices", "omp_num_threads", "hf_home", "huggingface_hub_cache", "transformers_cache"}
        child_env = {key: value for key, value in os.environ.items() if key.lower() in allowed_environment}
        child_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                         DOCLING_ARTIFACTS_PATH=str(artifacts_path))
        for key in list(child_env):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                child_env.pop(key)
        command = [sys.executable, "-m", "observatory.extraction", "--docling-worker", str(source), str(output),
                   str(artifacts_path), str(ocr_path) if ocr_path else "", "1" if formulas else "0"]
        completed = subprocess.run(command, env=child_env, cwd=str(Path(__file__).resolve().parents[1]),
                                   capture_output=True, timeout=120, check=False)
        if completed.returncode != 0 or not output.exists():
            # Dependency traces can contain host paths; record a stable warning only.
            raise RuntimeError("offline_docling_conversion_failed")
        if output.stat().st_size > MAX_EXTRACTED_CHARACTERS * 6:
            raise ValueError("Docling output exceeds extraction size")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("status") != "success" or not isinstance(result.get("text"), str):
            raise RuntimeError("docling_incomplete_conversion")
        return result["text"]


def extract_pdf_document(data: bytes) -> Extraction:
    """Fallback remains discovery-only until recorded fidelity review.

    The caller has already applied public URL/redirect/robots checks. This
    extractor accepts bytes only and cannot initiate document or image fetching.
    """
    if not isinstance(data, bytes) or not data.startswith(b"%PDF"):
        raise ValueError("Extraction requires downloaded PDF bytes")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("PDF exceeds the bounded extraction size")
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs require separate operator handling")
    page_count = len(reader.pages)
    if not 0 < page_count <= MAX_PDF_PAGES:
        raise ValueError("PDF page count exceeds the bounded extraction limit")
    engine = os.environ.get("OBSERVATORY_PDF_EXTRACTOR", "pypdf").lower()
    warnings = ["pdf_fidelity_requires_owner_review"]
    text = ""
    method = "pypdf_layout_fallback"
    preserved = False
    if engine == "docling":
        artifacts_value = os.environ.get("OBSERVATORY_DOCLING_ARTIFACTS_PATH", "")
        ocr_value = os.environ.get("OBSERVATORY_DOCLING_OCR_PATH", "")
        artifacts = Path(artifacts_value) if artifacts_value else None
        ocr = Path(ocr_value) if ocr_value else None
        if not artifacts or not artifacts.is_absolute() or not artifacts.is_dir():
            warnings.append("docling_local_artifacts_not_configured")
        elif ocr_value and (not ocr.is_absolute() or not ocr.is_dir()):
            warnings.append("docling_local_ocr_artifacts_not_configured")
        else:
            try:
                formulas = os.environ.get("OBSERVATORY_DOCLING_FORMULAS", "0") == "1"
                text = _docling_pdf_bytes(data, artifacts, ocr,
                    formulas=formulas)
                method, preserved = "docling_local_layout_ocr" if ocr else "docling_local_layout", True
                if not ocr:
                    warnings.append("ocr_disabled_scanned_regions_may_be_missing")
                if not formulas:
                    warnings.append("formula_enrichment_disabled_requires_formula_review")
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired):
                warnings.append("docling_unavailable_or_incomplete_used_pypdf")
    elif engine != "pypdf":
        raise ValueError("OBSERVATORY_PDF_EXTRACTOR must be pypdf or docling")
    if not text:
        if method.startswith("docling"):
            warnings.append("docling_empty_output_used_pypdf")
        method, preserved = "pypdf_layout_fallback", False
        warnings.append("fallback_reading_order_tables_and_formulas_unverified")
        text = "\n\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
    text = structured_text(text)
    if len(text) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError("PDF extracted text exceeds bounded extraction size")
    return Extraction(text, "pdf", method, preserved, warnings, True,
                      {"pages": page_count, "input_sha256": hashlib.sha256(data).hexdigest()})


def _run_docling_worker(arguments: list[str]) -> None:
    """No network connections, remote services or automatic weight downloads."""
    import socket

    def offline(*args, **kwargs):
        raise OSError("Network access is disabled in the document extraction worker")

    socket.create_connection = offline
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.getaddrinfo = offline
    socket.socket.sendto = offline
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    source, output, artifacts, ocr, formulas = arguments
    options = PdfPipelineOptions(artifacts_path=Path(artifacts), do_ocr=bool(ocr),
        do_table_structure=True, enable_remote_services=False,
        do_picture_description=False, do_code_enrichment=False, do_formula_enrichment=formulas == "1",
        generate_page_images=False)
    if ocr:
        options.ocr_options = EasyOcrOptions(model_storage_directory=ocr, download_enabled=False, lang=["en"])
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF],
                                  format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converted = converter.convert(Path(source), max_num_pages=MAX_PDF_PAGES, max_file_size=MAX_DOCUMENT_BYTES)
    status = getattr(converted.status, "value", str(converted.status))
    text = converted.document.export_to_markdown() if status == "success" else ""
    if len(text) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError("Docling extracted text exceeds bounded extraction size")
    Path(output).write_text(json.dumps({"status": status, "text": text}), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--docling-worker":
        _run_docling_worker(sys.argv[2:])
    else:
        raise SystemExit("This module is an internal offline PDF worker")
