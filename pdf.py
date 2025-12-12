import json
import os
from pathlib import Path

import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
METADATA_PATH = INDEX_DIR / "chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def disable_proxy_env():
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(key, None)


def load_pdfs(pdf_dir: Path) -> list[str]:
    """Извлекаем текст из всех PDF в каталоге."""
    texts: list[str] = []
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        parts: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)

        full = "\n\n".join(parts).strip()
        if full:
            texts.append(full)

        print(f"[ingest] loaded: {pdf_path.name} ({len(full)} chars)")

    return texts


def chunk_texts(texts: list[str], chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Разбиваем на чанки с перекрытием через RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return [c for c in chunks if c.strip()]


def embed_chunks(chunks: list[str], model: SentenceTransformer):
    """Считаем эмбеддинги всех чанков (L2-нормировка включена)."""
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def persist_index(embeddings, chunks: list[str]):
    """Создаём FAISS индекс и сохраняем + метаданные чанков."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    dim = int(embeddings.shape[1])
    # т.к. normalize_embeddings=True, можно использовать inner product как cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    meta = [{"id": i, "text": txt} for i, txt in enumerate(chunks)]
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ingest] saved index: {INDEX_DIR / 'faiss.index'}")
    print(f"[ingest] saved metadata: {METADATA_PATH}")


def main():
    disable_proxy_env()

    texts = load_pdfs(PDF_DIR)
    if not texts:
        print("[ingest] no pdf texts found in data/pdfs")
        return

    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    print(f"[ingest] chunks: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_chunks(chunks, model)
    persist_index(embeddings, chunks)


if __name__ == "__main__":
    main()
