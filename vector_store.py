import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
METADATA_PATH = INDEX_DIR / "chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Обёртка над FAISS для поиска по чанкам."""
    def __init__(self):
        # отключаем прокси, чтобы не ломали скачивание/запросы
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(key, None)

        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Возвращает топ-k чанков (текст, score)."""
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(query_vec, top_k)

        results: List[Tuple[str, float]] = []
        for i, score in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            results.append((self.meta[i]["text"], float(score)))
        return results
