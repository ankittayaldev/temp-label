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


# =========================
# User-configurable settings
# =========================
RELATIVE_PDF_PATH = "test.pdf"

MISTRAL_URL = "https://ecs-genai-label-update.services.ai.azure.com/providers/mistral/azure/ocr"
MISTRAL_KEY = ""  # Bearer token value (the key)

MISTRAL_MODEL = "mistral-document-ai-2505"

OPENAI_URL = "https://NEXT-Azure-OpenAI-Development.openai.azure.com/"
OPENAI_KEY = ""   # Azure OpenAI api-key
OPENAI_MODEL = "gpt-5.2" 


# =========================
# Constants / defaults
# =========================
MAX_PAGES_PER_CHUNK = 30
HTTP_TIMEOUT_SECONDS = 300
RETRY_MAX_ATTEMPTS = 4

IMAGE_MD_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


# =========================
# Utility helpers
# =========================
def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def to_data_url(mime: str, raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def split_data_url(data_url: str) -> Tuple[str, str]:
    """
    Returns (mime, b64payload) from 'data:<mime>;base64,<payload>'
    """
    if not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError("Not a base64 data URL.")
    header, b64payload = data_url.split(",", 1)
    mime = header[len("data:") :].split(";")[0]
    return mime, b64payload


def decode_data_url_to_bytes(data_url: str) -> Tuple[str, bytes]:
    mime, b64payload = split_data_url(data_url)
    return mime, base64.b64decode(b64payload)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# =========================
# PDF handling
# =========================
@dataclass
class PdfChunk:
    chunk_index: int
    start_page: int  # 0-based inclusive
    end_page: int    # 0-based inclusive
    path: Path


class PdfChunker:
    def __init__(self, input_pdf: Path, out_pdf_dir: Path, max_pages_per_chunk: int = MAX_PAGES_PER_CHUNK):
        self.input_pdf = input_pdf
        self.out_pdf_dir = out_pdf_dir
        self.max_pages_per_chunk = max_pages_per_chunk

    def verify_and_count_pages(self) -> int:
        # Step 0: verify PDF integrity
        if not self.input_pdf.exists():
            die(f"PDF not found: {self.input_pdf}")

        try:
            reader = PdfReader(str(self.input_pdf))
            num_pages = len(reader.pages)
            if num_pages <= 0:
                die("PDF has zero pages.")
            return num_pages
        except Exception as e:
            die(f"PDF integrity check failed: {e}")

    def chunk(self, num_pages: int) -> List[PdfChunk]:
        ensure_dir(self.out_pdf_dir)

        # If <= 30 pages, still copy into output/pdf for traceability
        if num_pages <= self.max_pages_per_chunk:
            out_path = self.out_pdf_dir / sanitize_filename(self.input_pdf.name)
            out_path.write_bytes(self.input_pdf.read_bytes())
            return [PdfChunk(chunk_index=1, start_page=0, end_page=num_pages - 1, path=out_path)]

        # else split into batches
        chunks: List[PdfChunk] = []
        reader = PdfReader(str(self.input_pdf))

        chunk_idx = 1
        for start in range(0, num_pages, self.max_pages_per_chunk):
            end = min(start + self.max_pages_per_chunk, num_pages) - 1
            writer = PdfWriter()
            for p in range(start, end + 1):
                writer.add_page(reader.pages[p])

            out_path = self.out_pdf_dir / f"chunk_{chunk_idx:03d}_{start+1:04d}-{end+1:04d}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)

            chunks.append(PdfChunk(chunk_index=chunk_idx, start_page=start, end_page=end, path=out_path))
            chunk_idx += 1

        return chunks


# =========================
# HTTP clients
# =========================
class HttpClient:
    def __init__(self):
        self.session = requests.Session()

    def post_json_with_retry(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err = None
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                resp = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=HTTP_TIMEOUT_SECONDS
                )
                if resp.status_code >= 400:
                    # try to surface JSON errors
                    try:
                        j = resp.json()
                        raise RuntimeError(f"HTTP {resp.status_code}: {j}")
                    except Exception:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")

                return resp.json()

            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 20)
                print(f"WARN: POST failed (attempt {attempt}/{RETRY_MAX_ATTEMPTS}): {e} ; retrying in {sleep_s}s")
                time.sleep(sleep_s)

        raise RuntimeError(f"POST failed after retries: {last_err}")


class MistralOcrClient:
    def __init__(self, mistral_url: str, mistral_key: str, model: str):
        self.url = mistral_url
        self.key = mistral_key
        self.model = model
        self.http = HttpClient()

    def ocr_pdf(self, pdf_path: Path, document_name: str) -> Dict[str, Any]:
        pdf_bytes = read_file_bytes(pdf_path)
        document_url = to_data_url("application/pdf", pdf_bytes)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_name": document_name,
                "document_url": document_url,
            },
            "include_image_base64": True,
        }

        return self.http.post_json_with_retry(self.url, headers, payload)


class AzureOpenAIAnnotator:
    """
    Uses Azure OpenAI Responses API to generate a short caption/annotation for an image.
    The Responses API supports passing base64 images as data URLs via input_image.image_url. :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.http = HttpClient()

    def _responses_url(self) -> str:
        # Azure OpenAI v1 Responses API endpoint path
        return f"{self.base_url}/openai/v1/responses"

    def annotate_image(self, image_data_url: str, context_hint: Optional[str] = None) -> str:
        if not self.api_key:
            return ""

        prompt = (
            "You are annotating an image extracted from a PDF during OCR.\n"
            "Write a concise, factual description suitable as a figure note.\n"
            "If it's a chart/plot, mention what is plotted and the key trend.\n"
            "Do not invent details that are not visible.\n"
            "Return plain text only (no markdown)."
        )
        if context_hint:
            prompt += f"\n\nOCR context near the image:\n{context_hint[:1200]}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            "max_output_tokens": 200,
            "temperature": 0,
        }

        resp = self.http.post_json_with_retry(self._responses_url(), headers, payload)

        # Prefer output_text when available; otherwise fall back to digging in output items.
        if isinstance(resp, dict):
            if resp.get("output_text"):
                return str(resp["output_text"]).strip()

            # fallback: look for output_text chunks
            out = resp.get("output", [])
            texts: List[str] = []
            for item in out:
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and "text" in c:
                        texts.append(c["text"])
            return "\n".join(t.strip() for t in texts if t.strip()).strip()

        return ""


# =========================
# Markdown assembly
# =========================
class MarkdownAssembler:
    def __init__(self):
        pass

    @staticmethod
    def embed_images_and_annotations(page_markdown: str, images: List[Dict[str, Any]]) -> str:
        """
        Replaces markdown placeholders (img-*.jpeg) with data URLs if present,
        and injects 'Image annotation' lines immediately after the image line.
        """
        if not page_markdown:
            page_markdown = ""

        # Build lookup by id
        by_id: Dict[str, Dict[str, Any]] = {}
        for img in images or []:
            img_id = img.get("id")
            if img_id:
                by_id[img_id] = img

        lines = page_markdown.splitlines()
        out_lines: List[str] = []

        for line in lines:
            m = IMAGE_MD_PATTERN.search(line)
            if not m:
                out_lines.append(line)
                continue

            alt_text, target = m.group(1), m.group(2)

            # Only rewrite if target corresponds to an OCR image id
            img = by_id.get(target)
            if not img:
                out_lines.append(line)
                continue

            data_url = img.get("image_base64")
            annotation = img.get("image_annotation")

            # Replace target with embedded data URL if we have it
            if data_url:
                new_line = line.replace(f"]({target})", f"]({data_url})")
            else:
                new_line = line

            out_lines.append(new_line)

            # Inject annotation right after the image line
            if annotation:
                out_lines.append(f"> **Image annotation:** {annotation}")

        return "\n".join(out_lines)

    @staticmethod
    def build_document_markdown(pdf_name: str, pages: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        chunks.append(f"# OCR Output: {pdf_name}")
        chunks.append("")  # blank line

        for page in pages:
            idx = page.get("index")
            page_no = (idx + 1) if isinstance(idx, int) else "?"
            chunks.append(f"## Page {page_no}")
            chunks.append("")

            md = page.get("markdown", "")
            images = page.get("images", []) or []
            md2 = MarkdownAssembler.embed_images_and_annotations(md, images)

            chunks.append(md2.strip())
            chunks.append("\n---\n")

        return "\n".join(chunks).rstrip() + "\n"


# =========================
# Pipeline
# =========================
def main() -> None:
    input_pdf = Path(RELATIVE_PDF_PATH)
    pdf_stem = input_pdf.stem
    out_root = Path(pdf_stem)  # "relative output folder = filename"

    out_pdf_dir = out_root / "pdf"
    out_json_dir = out_root / "json"
    out_img_dir = out_root / "images"

    ensure_dir(out_root)
    ensure_dir(out_pdf_dir)
    ensure_dir(out_json_dir)
    ensure_dir(out_img_dir)

    print(f"{input_pdf.name} processing ...")

    # Step 0 + 1: verify integrity and page count
    chunker = PdfChunker(input_pdf=input_pdf, out_pdf_dir=out_pdf_dir, max_pages_per_chunk=MAX_PAGES_PER_CHUNK)
    num_pages = chunker.verify_and_count_pages()
    print(f"PDF OK. Pages: {num_pages}")

    chunks = chunker.chunk(num_pages=num_pages)
    if num_pages <= MAX_PAGES_PER_CHUNK:
        print("PDF is <= 30 pages. No chunking needed.")
    else:
        print(f"PDF is > 30 pages. Created {len(chunks)} chunks of up to 30 pages each.")

    # Clients
    if not MISTRAL_KEY:
        die("MISTRAL_KEY is empty. Set it at top of file before running.")

    mistral = MistralOcrClient(mistral_url=MISTRAL_URL, mistral_key=MISTRAL_KEY, model=MISTRAL_MODEL)
    openai_annotator = AzureOpenAIAnnotator(base_url=OPENAI_URL, api_key=OPENAI_KEY, model=OPENAI_MODEL)

    combined_raw_pages: List[Dict[str, Any]] = []
    combined_parsed_pages: List[Dict[str, Any]] = []

    # Step 2: OCR each chunk
    for ch in chunks:
        print(f"OCR chunk {ch.chunk_index:03d}: pages {ch.start_page+1}-{ch.end_page+1} ({ch.path.name})")

        raw = mistral.ocr_pdf(pdf_path=ch.path, document_name=ch.path.name)

        raw_path = out_json_dir / f"raw_chunk_{ch.chunk_index:03d}.json"
        raw_path.write_text(pretty_json(raw), encoding="utf-8")

        pages = raw.get("pages", [])
        if not isinstance(pages, list):
            die(f"Unexpected OCR response: 'pages' is not a list in chunk {ch.chunk_index}")

        # Normalize global page indices and process images
        for p in pages:
            if not isinstance(p, dict):
                continue

            local_idx = p.get("index")
            if not isinstance(local_idx, int):
                continue

            global_idx = ch.start_page + local_idx

            # Clone for safety
            p2 = json.loads(json.dumps(p))

            p2["index"] = global_idx

            # Step 3: annotate images via Azure OpenAI (optional)
            images = p2.get("images", []) or []
            for img in images:
                if not isinstance(img, dict):
                    continue

                img_id = img.get("id") or "image"
                img_b64 = img.get("image_base64")

                # Save image file to disk (optional but useful)
                if img_b64:
                    try:
                        mime, raw_bytes = decode_data_url_to_bytes(img_b64)
                        ext = {
                            "image/jpeg": ".jpg",
                            "image/jpg": ".jpg",
                            "image/png": ".png",
                            "image/webp": ".webp",
                        }.get(mime, Path(img_id).suffix or ".bin")

                        out_img_name = sanitize_filename(f"p{global_idx+1:04d}_{img_id}")
                        if not out_img_name.lower().endswith(ext):
                            out_img_name += ext

                        (out_img_dir / out_img_name).write_bytes(raw_bytes)
                    except Exception as e:
                        print(f"WARN: Failed to decode/save {img_id} on page {global_idx+1}: {e}")

                # If OpenAI key is provided, create annotation
                if img_b64 and OPENAI_KEY:
                    context_hint = (p2.get("markdown") or "")[:1500]
                    try:
                        annotation = openai_annotator.annotate_image(img_b64, context_hint=context_hint)
                        img["image_annotation"] = annotation or None
                    except Exception as e:
                        print(f"WARN: OpenAI annotation failed for {img_id} on page {global_idx+1}: {e}")
                        img["image_annotation"] = None
                else:
                    # leave as-is (null) if no key
                    img["image_annotation"] = img.get("image_annotation")

            combined_raw_pages.append(p)   # original (local indices)
            combined_parsed_pages.append(p2)

    # Build combined RAW JSON (normalized to global indices as well)
    # For clarity: raw.json is combined output before we injected annotations, but with global indices.
    # We'll reconstruct raw_global by copying parsed_pages and blanking image_annotation.
    raw_global_pages: List[Dict[str, Any]] = []
    for p in combined_parsed_pages:
        p_raw = json.loads(json.dumps(p))
        for img in p_raw.get("images", []) or []:
            if isinstance(img, dict):
                img["image_annotation"] = None
        raw_global_pages.append(p_raw)

    raw_combined = {
        "model": MISTRAL_MODEL,
        "pages": sorted(raw_global_pages, key=lambda x: x.get("index", 10**9)),
    }
    parsed_combined = {
        "model": MISTRAL_MODEL,
        "pages": sorted(combined_parsed_pages, key=lambda x: x.get("index", 10**9)),
    }

    (out_root / "raw.json").write_text(pretty_json(raw_combined), encoding="utf-8")
    (out_root / "parsed.json").write_text(pretty_json(parsed_combined), encoding="utf-8")

    # Step 5: combined markdown with embedded base64 images and annotations
    combined_md = MarkdownAssembler.build_document_markdown(
        pdf_name=input_pdf.name,
        pages=parsed_combined["pages"]
    )
    (out_root / "combined.md").write_text(combined_md, encoding="utf-8")

    print("Done.")
    print(f"Output folder: {out_root.resolve()}")
    print(f"- {out_root / 'raw.json'}")
    print(f"- {out_root / 'parsed.json'}")
    print(f"- {out_root / 'combined.md'}")
    print(f"- chunks under: {out_pdf_dir}")
    print(f"- images under: {out_img_dir}")


if __name__ == "__main__":
    main()
