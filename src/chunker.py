from transformers import AutoTokenizer

class SciFactChunker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=220, overlap=30):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_document(self, doc_id: str, title: str, text: str):
        full_text = f"Title: {title}\nText: {text}"
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        chunks = []
        if len(tokens) <= self.max_tokens:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_c0",
                "title": title,
                "text": full_text
            })
        else:
            stride = self.max_tokens - self.overlap
            chunk_idx = 0
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i + self.max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "title": title,
                    "text": chunk_text
                })
                chunk_idx += 1
                if i + self.max_tokens >= len(tokens):
                    break
        return chunks