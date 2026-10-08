import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class VectorIndexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # chiều của all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_map = []

    def build_and_save(self, chunks: list, index_path="./storage/faiss.index", map_path="./storage/metadata.json"):
        texts = [c["text"] for c in chunks]
        
        # 1. Embed & Normalize L2
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        
        # 2. Add to FAISS Index
        self.index.add(embeddings.astype(np.float32))
        
        # 3. Save Index and Metadata
        faiss.write_index(self.index, index_path)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"Đã index {len(chunks)} chunks vào {index_path}")