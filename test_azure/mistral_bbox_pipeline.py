#!/usr/bin/env python3
"""
mistral_bbox_pipeline_azure.py

Requirements:
  pip install requests pypdf

What it does:
  - Verifies PDF integrity
  - Splits PDF into <=30-page chunks (Azure Foundry limit: 30 pages)  :contentReference[oaicite:6]{index=6}
  - Calls Azure-hosted Mistral Document AI OCR with bbox annotations
  - Produces a standalone combined.md with embedded base64 images + captions
  - Writes raw JSON per chunk as raw.json (and chunk raw files)

Notes:
  - Do NOT send table_format; your endpoint rejects it (HTTP 422 extra_forbidden).
  - URLs (including presigned URLs) are not accepted on your deployment; it requires base64 data URLs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pypdf import PdfReader, PdfWriter


# -----------------------
# Config (edit these)
# -----------------------
RELATIVE_FILE_PATH = "test.pdf"

MISTRAL_URL = "https://ecs-genai-label-update.services.ai.azure.com/providers/mistral/azure/ocr"
MISTRAL_KEY = ""  # Bearer token
MISTRAL_MODEL = "mistral-document-ai-2505"

# Output root folder will be ./<pdf_stem>/
# Inside: pdf/, json/, images/, combined.md
MAX_PAGES_PER_CHUNK = 30

# Retry only for transient errors (429/5xx/timeouts)
MAX_RETRIES = 5


# -----------------------
# Annotation schema
# -----------------------
# Keep this simple and robust.
# The model will try to return image_annotation matching the schema.
#
# We aim for:
#   page.images[i].image_annotation = {"caption": "..."}
#
# You can extend it later (type, table extraction, etc.).
BBOX_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "A concise, accurate caption describing the figure/table/image. Include key variables/axes if it is a chart."
        }
    },
    "required": ["caption"],
    "additionalProperties": False
}


def bbox_annotation_format_payload_variants(schema: dict) -> List[dict]:
    """
    Azure/Mistral deployments sometimes differ in the exact wrapper expected for response formats.
    We try a few common wrappers. If your endpoint validates strictly, one of these should match.
    """
    return [
        # Variant A: common structured-output wrapper (name + schema + strict)
        {
            "type": "json_schema",
            "json_schema": {
                "name": "bbox_image_annotation",
                "schema": schema,
                "strict": True,
            },
        },
        # Variant B: some deployments omit "name"/"strict"
        {
            "type": "json_schema",
            "json_schema": schema,
        },
        # Variant C: sometimes accepted as json_object with schema directly
        {
            "type": "json_object",
            "schema": schema,
        },
    ]


# -----------------------
# Helpers
# -----------------------
def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def verify_pdf(pdf_path: Path) -> int:
    """
    Verifies PDF can be opened and returns page count.
    """
    try:
        reader = PdfReader(str(pdf_path))
        n = len(reader.pages)
        # Touch first page to force parsing
        _ = reader.pages[0]
        return n
    except Exception as e:
        raise RuntimeError(f"PDF integrity check failed: {e}") from e


def split_pdf(pdf_path: Path, out_dir: Path, max_pages: int) -> List[Path]:
    """
    Splits PDF into <=max_pages chunks. Returns list of chunk paths.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    chunks: List[Path] = []
    if total_pages <= max_pages:
        # Still copy into pdf/ for consistent layout
        chunk_path = out_dir / pdf_path.name
        chunk_path.write_bytes(pdf_path.read_bytes())
        return [chunk_path]

    for start in range(0, total_pages, max_pages):
        end = min(start + max_pages, total_pages)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        chunk_name = f"{pdf_path.stem}_p{start+1:04d}-{end:04d}.pdf"
        chunk_path = out_dir / chunk_name
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)

    return chunks


def file_to_data_url(pdf_path: Path) -> str:
    raw = pdf_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def mistral_headers() -> Dict[str, str]:
    if not MISTRAL_KEY.strip():
        die("MISTRAL_KEY is empty. Set it in the script first.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_KEY}",
    }


def is_transient(status_code: int) -> bool:
    return status_code in (408, 429) or (500 <= status_code <= 599)


def call_mistral(payload: dict) -> dict:
    """
    Calls Azure Foundry Mistral OCR endpoint with sane retry behavior:
      - retries only for transient failures
      - does NOT retry 4xx validation errors like 400/401/403/422
    """
    headers = mistral_headers()
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            last_err = f"RequestException: {e}"
            backoff = min(2 ** attempt, 30)
            print(f"Request failed (attempt {attempt}/{MAX_RETRIES}): {last_err}. Retry in {backoff}s...")
            time.sleep(backoff)
            continue

        if resp.status_code == 200:
            return resp.json()

        # Non-200:
        body = resp.text
        last_err = f"HTTP {resp.status_code}: {body}"

        if is_transient(resp.status_code):
            backoff = min(2 ** attempt, 30)
            print(f"Transient failure (attempt {attempt}/{MAX_RETRIES}): {last_err}. Retry in {backoff}s...")
            time.sleep(backoff)
            continue

        # Permanent (validation/auth/etc.)
        raise RuntimeError(last_err)

    raise RuntimeError(f"OCR request failed after {MAX_RETRIES} attempts. Last error: {last_err}")


def replace_images_and_inject_captions(page_markdown: str, images: List[dict]) -> str:
    """
    Replaces ![img-x](img-x) with embedded base64 data URL (if present),
    and injects caption below the image using bbox annotation output.

    Expected image object keys (based on your response):
      - id
      - image_base64 (data:image/...;base64,...)
      - image_annotation (dict or None)
    """
    md = page_markdown

    # Build map: id -> data_url
    id_to_dataurl = {}
    id_to_caption = {}

    for img in images or []:
        img_id = img.get("id")
        data_url = img.get("image_base64")
        if img_id and isinstance(data_url, str) and data_url.startswith("data:image/"):
            id_to_dataurl[img_id] = data_url

        ann = img.get("image_annotation")
        # If schema returns {"caption": "..."}
        if isinstance(ann, dict):
            cap = ann.get("caption")
            if isinstance(cap, str) and cap.strip():
                id_to_caption[img_id] = cap.strip()
        # If some deployments return plain string
        elif isinstance(ann, str) and ann.strip():
            id_to_caption[img_id] = ann.strip()

    # Replace image placeholders
    for img_id, data_url in id_to_dataurl.items():
        md = md.replace(f"![{img_id}]({img_id})", f"![{img_id}]({data_url})")

    # Inject captions: after each image markdown occurrence
    def inject_caption(match: re.Match) -> str:
        alt = match.group(1)
        url = match.group(2)
        caption = id_to_caption.get(alt)
        if caption:
            return f"![{alt}]({url})\n\n> **Caption:** {caption}\n"
        return match.group(0)

    md = re.sub(r"!\[([^\]]+)\]\(([^)]+)\)", inject_caption, md)
    return md


def build_payload(document_data_url: str, document_name: str, bbox_format: dict) -> dict:
    """
    Builds the request payload expected by your Azure endpoint.
    IMPORTANT: No table_format here (your endpoint rejects it).
    """
    return {
        "model": MISTRAL_MODEL,
        "document": {
            "type": "document_url",
            "document_name": document_name,
            "document_url": document_data_url,
        },
        "include_image_base64": True,
        "bbox_annotation_format": bbox_format,
    }


def run_bbox_ocr_with_format_fallback(document_data_url: str, document_name: str) -> Tuple[dict, dict]:
    """
    Try multiple bbox_annotation_format wrapper variants until one succeeds.
    Returns (response_json, used_bbox_format).
    """
    last_error: Optional[Exception] = None
    for fmt in bbox_annotation_format_payload_variants(BBOX_JSON_SCHEMA):
        payload = build_payload(document_data_url, document_name=document_name, bbox_format=fmt)
        try:
            resp_json = call_mistral(payload)
            return resp_json, fmt
        except Exception as e:
            last_error = e
            msg = str(e)
            # If the service says "extra_forbidden" for bbox_annotation_format itself,
            # you’ll see it here. Otherwise keep trying variants.
            print(f"bbox_annotation_format variant failed: {msg}")
            continue
    raise RuntimeError(f"All bbox_annotation_format variants failed. Last error: {last_error}")


def main() -> None:
    pdf_path = Path(RELATIVE_FILE_PATH).expanduser().resolve()
    if not pdf_path.exists():
        die(f"File not found: {pdf_path}")

    print(f"{pdf_path.name} processing ...")

    # 0) Verify PDF integrity
    try:
        page_count = verify_pdf(pdf_path)
        print(f"PDF OK. Pages: {page_count}")
    except Exception as e:
        die(str(e))

    # Output folders
    out_root = Path.cwd() / pdf_path.stem
    out_pdf = out_root / "pdf"
    out_json = out_root / "json"
    out_images = out_root / "images"

    out_pdf.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # 1) If pdf > 30 pages, split into <=30 page chunks
    chunk_paths = split_pdf(pdf_path, out_pdf, MAX_PAGES_PER_CHUNK)

    combined_markdown_parts: List[str] = []
    combined_response_pages: List[dict] = []
    used_format: Optional[dict] = None

    for idx, chunk_path in enumerate(chunk_paths, start=1):
        print(f"\nChunk {idx}/{len(chunk_paths)}: {chunk_path.name}")

        doc_data_url = file_to_data_url(chunk_path)

        # 2) Call OCR with bbox annotations
        resp_json, fmt = run_bbox_ocr_with_format_fallback(doc_data_url, chunk_path.name)
        used_format = fmt

        # Save raw response
        raw_path = out_json / f"raw_chunk_{idx:02d}.json"
        raw_path.write_text(json.dumps(resp_json, indent=2, ensure_ascii=False), encoding="utf-8")

        pages = resp_json.get("pages", [])
        for p in pages:
            combined_response_pages.append(p)

            page_index = p.get("index")
            page_md = p.get("markdown", "") or ""
            page_images = p.get("images", []) or []

            # 3/4/5) Embed images and captions into markdown
            page_md2 = replace_images_and_inject_captions(page_md, page_images)
            combined_markdown_parts.append(page_md2)

            # Save images separately too (optional but useful)
            for img in page_images:
                img_id = img.get("id")
                data_url = img.get("image_base64")
                if not (img_id and isinstance(data_url, str) and data_url.startswith("data:image/")):
                    continue
                # Extract bytes after "base64,"
                try:
                    b64_part = data_url.split("base64,", 1)[1]
                    img_bytes = base64.b64decode(b64_part)
                    (out_images / img_id).write_bytes(img_bytes)
                except Exception:
                    # If decode fails, skip
                    continue

    # Write combined markdown
    combined_md = "\n\n---\n\n".join(combined_markdown_parts)
    (out_root / "combined.md").write_text(combined_md, encoding="utf-8")

    # Write combined "raw.json" at root with pages stitched
    stitched = {
        "model": MISTRAL_MODEL,
        "pages": combined_response_pages,
        "bbox_annotation_format_used": used_format,
    }
    (out_root / "raw.json").write_text(json.dumps(stitched, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Output folder: {out_root}")


if __name__ == "__main__":
    main()
