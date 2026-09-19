import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
from sentence_transformers import SentenceTransformer

class LocalComplianceRetriever:
    """
    Local-only FAISS/vector compliance retriever.
    Strictly isolated: ZERO outbound network calls.
    Uses sentence-transformers all-MiniLM-L6-v2 and faiss-cpu.
    """
    def __init__(self, index_dir: str = "artifacts/indices", embedding_model_path: Optional[str] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.index_hash: str = "0" * 64
        self.dimension: int = 384

        # Load embedding model from local artifacts directory (offline edge)
        default_local = Path("artifacts/embeddings/all-MiniLM-L6-v2")
        if embedding_model_path and Path(embedding_model_path).exists():
            model_source = str(embedding_model_path)
        elif default_local.exists():
            model_source = str(default_local)
        else:
            model_source = "sentence-transformers/all-MiniLM-L6-v2"

        try:
            self.encoder = SentenceTransformer(model_source, local_files_only=Path(model_source).exists())
        except Exception:
            self.encoder = SentenceTransformer(model_source)

        self._load_local_index()

    def _compute_chunk_embedding(self, text: str) -> np.ndarray:
        """
        Real semantic embedding using SentenceTransformer (all-MiniLM-L6-v2).
        Returns normalized 384-dimensional float32 vector.
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        emb = self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)

    def build_index_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        self.chunks = []
        embedding_list = []

        for doc in documents:
            chunk = {
                "chunk_id": doc.get("doc_id", doc.get("clause_id", "CHUNK-UNKNOWN")),
                "clause_id": doc.get("clause_id", "Clause UNKNOWN"),
                "regime_id": doc.get("regime_id", "Q1"),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "keywords": doc.get("keywords", []),
                "effective_date": doc.get("effective_date", "2025-01-01")
            }
            content_to_embed = f"{chunk['clause_id']} {chunk['title']}: {chunk['text']}"
            emb = self._compute_chunk_embedding(content_to_embed)
            self.chunks.append(chunk)
            embedding_list.append(emb)

        if embedding_list:
            self.embeddings = np.vstack(embedding_list)
            # Create FAISS IndexFlatIP (cosine similarity on normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(self.embeddings)
        else:
            self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)

        # Compute SHA-256 hash of all indexed chunk contents
        raw_repr = json.dumps(self.chunks, sort_keys=True)
        self.index_hash = hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()

        self._save_local_index()
        return self.index_hash

    def _save_local_index(self):
        chunks_file = self.index_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump({
                "index_hash": self.index_hash,
                "chunks": self.chunks
            }, f, indent=2)
        if self.embeddings is not None:
            np.save(self.index_dir / "embeddings.npy", self.embeddings)
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.index_dir / "faiss.index"))

    def _load_local_index(self):
        chunks_file = self.index_dir / "chunks.json"
        emb_file = self.index_dir / "embeddings.npy"
        faiss_file = self.index_dir / "faiss.index"
        if chunks_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.index_hash = data.get("index_hash", "0" * 64)
                    self.chunks = data.get("chunks", [])
            except Exception:
                pass
        if faiss_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_file))
            except Exception:
                self.faiss_index = None
        if emb_file.exists():
            try:
                self.embeddings = np.load(emb_file)
                if self.faiss_index is None and len(self.embeddings) > 0:
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
                    self.faiss_index.add(self.embeddings)
            except Exception:
                pass

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        target_regime: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        q_vec = self._compute_chunk_embedding(query)
        if q_vec.ndim == 1:
            q_vec_batch = q_vec.reshape(1, -1)
        else:
            q_vec_batch = q_vec

        # FAISS search
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            search_k = min(len(self.chunks), max(top_k * 3, 10))
            distances, indices = self.faiss_index.search(q_vec_batch, search_k)
            retrieved_indices = indices[0]
            base_scores = distances[0]
        else:
            base_scores = np.dot(self.embeddings, q_vec)
            retrieved_indices = np.arange(len(self.chunks))

        # Keyword and clause boost heuristic
        q_lower = query.lower()
        results = []
        for idx, base_score in zip(retrieved_indices, base_scores):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            # Temporal filter
            if target_regime and chunk.get("regime_id") != target_regime:
                continue

            score = float(base_score)
            if chunk["clause_id"].lower() in q_lower:
                score += 0.35
            for kw in chunk.get("keywords", []):
                if kw.lower() in q_lower:
                    score += 0.12

            # Clamp score to [0.0, 1.0]
            normalized_score = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

            results.append({
                "chunk_id": chunk["chunk_id"],
                "clause_id": chunk["clause_id"],
                "regime_id": chunk["regime_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "effective_date": chunk["effective_date"],
                "similarity_score": round(normalized_score, 4)
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def get_index_hash(self) -> str:
        return self.index_hash

