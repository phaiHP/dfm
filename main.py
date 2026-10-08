# main.py
import sys
import json
import os
import csv
import time
import hashlib
import platform
import numpy as np
import faiss
from dotenv import load_dotenv

from src.chunker import SciFactChunker
from src.indexer import VectorIndexer
from src.llm_client import MinistralRAGClient
from src.data_loader import prepare_dev_split
from src.evaluator import evaluate_retrieval

load_dotenv()

def get_file_hash(filepath):
    """Tính SHA256 Hash của tệp dữ liệu."""
    if not os.path.exists(filepath):
        return "N/A"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ==========================================
# LỆNH 1: BUILD-INDEX & XUẤT MANIFEST.JSON
# ==========================================
def build_index():
    print("=== [1/5] BẮT ĐẦU BUILD INDEX ===")
    chunker = SciFactChunker()
    indexer = VectorIndexer()
    
    corpus_path = "data/scifact/corpus.jsonl"
    if not os.path.exists(corpus_path):
        print(f"LỖI: Không thấy {corpus_path}. Hãy chạy python download_data.py trước!")
        return

    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_chunks = chunker.chunk_document(str(item["_id"]), item.get("title", ""), item.get("text", ""))
            chunks.extend(doc_chunks)
            
    os.makedirs("storage", exist_ok=True)
    indexer.build_and_save(chunks)
    print("-> Đã ghi dữ liệu Index vào thư mục /storage/")

    # TỰ ĐỘNG GHI FILE manifest.json
    print("\n--- [TỰ ĐỘNG GHI FILE] manifest.json ---")
    dev_qids, train_practice_qids = prepare_dev_split("data/scifact/qrels/train.tsv")
    manifest_data = {
        "dataset": {
            "name": "BEIR/SciFact",
            "corpus_file": corpus_path,
            "corpus_hash_sha256": get_file_hash(corpus_path),
            "queries_file": "data/scifact/queries.jsonl",
            "queries_hash_sha256": get_file_hash("data/scifact/queries.jsonl")
        },
        "splits": {
            "random_seed": 42,
            "dev_split_size": len(dev_qids),
            "dev_query_ids": sorted(list(dev_qids)),
            "train_practice_size": len(train_practice_qids)
        },
        "pipeline_config": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "vector_index": "FAISS IndexFlatIP (L2 Normalized)",
            "chunk_config": {
                "max_tokens": 220,
                "overlap_tokens": 30,
                "tokenizer": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "llm": {
                "model_id_configured": "mistral-large-latest",
                "temperature": 0,
                "max_tokens": 512
            }
        },
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system() + " " + platform.release()
        }
    }
    with open("manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    print("-> GHI THÀNH CÔNG: manifest.json")
    print("=== BUILD INDEX HOÀN TẤT ===\n")

# ==========================================
# LỆNH 2: RETRIEVE
# ==========================================
def retrieve(query: str, top_k: int = 5, verbose: bool = True):
    start_time = time.time()
    
    indexer = VectorIndexer()
    indexer.index = faiss.read_index("storage/faiss.index")
    with open("storage/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    q_embed = indexer.model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_embed)
    
    D, I = indexer.index.search(q_embed.astype(np.float32), top_k * 2)
    
    retrieved_results = []
    seen_docs = set()
    rank = 1
    
    for idx, score in zip(I[0], D[0]):
        if idx < len(metadata):
            chunk = metadata[idx]
            doc_id = chunk["doc_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                retrieved_results.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk["chunk_id"],
                    "rank": rank,
                    "score": float(score),
                    "text": chunk["text"]
                })
                rank += 1
                if len(retrieved_results) == top_k:
                    break
                    
    ret_time = int((time.time() - start_time) * 1000)
    if verbose:
        print(f"=== RETRIEVE CHO QUERY: '{query}' ===")
        print(f"Thời gian Retrieval: {ret_time} ms")
        print(json.dumps(retrieved_results, indent=2, ensure_ascii=False))
    return retrieved_results

# ==========================================
# LỆNH 3A: ASK 1 CÂU ĐƠN
# ==========================================
def ask(query: str, query_id: str = "custom"):
    start_time = time.time()
    
    retrieved_chunks = retrieve(query, top_k=5, verbose=False)
    ret_time = int((time.time() - start_time) * 1000)
    
    client = MinistralRAGClient()
    llm_res, gen_time = client.generate_answer(query, retrieved_chunks)
    
    final_output = {
        "query_id": str(query_id),
        "query": query,
        "status": llm_res.get("status", "ANSWERED"),
        "answer": llm_res.get("answer", ""),
        "citations": llm_res.get("citations", []),
        "retrieved": [{"doc_id": r["doc_id"], "rank": r["rank"], "score": r["score"]} for r in retrieved_chunks],
        "timing_ms": {
            "retrieval": ret_time,
            "generation": gen_time,
            "total": ret_time + gen_time
        }
    }
    
    print("\n================ KẾT QUẢ RAG (JSON) ================")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    gen_record = {
        "query_id": str(query_id),
        "query": query,
        "answer": final_output["answer"],
        "status": final_output["status"],
        "citations": final_output["citations"],
        "context_doc_ids": [r["doc_id"] for r in retrieved_chunks],
        "timing_ms": final_output["timing_ms"]
    }
    with open("generation_run.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(gen_record, ensure_ascii=False) + "\n")
    print("-> GHI THÀNH CÔNG RECORD VÀO FILE: generation_run.jsonl")

# ==========================================
# LỆNH 3B: ASK BATCH TỪ FILE QUERIES.JSONL
# ==========================================
def ask_batch(limit: int = None):
    queries_path = "data/scifact/queries.jsonl"
    if not os.path.exists(queries_path):
        print(f"LỖI: Không tìm thấy file {queries_path}!")
        return

    dev_qids, _ = prepare_dev_split("data/scifact/qrels/train.tsv")
    
    print(f"=== ĐANG ĐỌC QUERIES TỪ {queries_path} ===")
    queries_to_run = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = int(item["_id"])
            if qid in dev_qids:
                queries_to_run.append({"id": str(qid), "text": item["text"]})
                
    if limit:
        queries_to_run = queries_to_run[:limit]
        
    print(f"Sẽ chạy RAG cho {len(queries_to_run)} câu hỏi...")
    
    with open("generation_run.jsonl", "w", encoding="utf-8") as f:
        pass

    for i, q in enumerate(queries_to_run, 1):
        print(f"\n[{i}/{len(queries_to_run)}] Processing Query ID {q['id']}: {q['text'][:60]}...")
        try:
            ask(q["text"], query_id=q["id"])
        except Exception as e:
            print(f"LỖI khi xử lý query {q['id']}: {e}")
            
    print("\n>>> HOÀN TẤT CHẠY RAG BATCH! Kết quả đã ghi vào generation_run.jsonl <<<")

# ==========================================
# LỆNH 4: EVALUATE (RETRIEVAL ONLY)
# ==========================================
def evaluate():
    print("=== [4/5] BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG RETRIEVAL (DEV SPLIT) ===")
    
    dev_qids, _ = prepare_dev_split("data/scifact/qrels/train.tsv")
    print(f"Đã khởi tạo tập Dev gồm {len(dev_qids)} query_id.")

    queries_dict = {}
    with open("data/scifact/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = int(item["_id"])
            if qid in dev_qids:
                queries_dict[qid] = item["text"]

    qrels_dict = {}
    with open("data/scifact/qrels/train.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if row:
                qid = int(row[0])
                doc_id = str(row[1])
                rel = int(row[2])
                if qid in dev_qids:
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][doc_id] = rel

    indexer = VectorIndexer()
    indexer.index = faiss.read_index("storage/faiss.index")
    with open("storage/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results_dict = {}
    print("Đang chạy retrieval trên tập Dev và ghi file retrieval_run.jsonl...")
    
    with open("retrieval_run.jsonl", "w", encoding="utf-8") as f_run:
        for qid, qtext in queries_dict.items():
            q_embed = indexer.model.encode([qtext], convert_to_numpy=True)
            faiss.normalize_L2(q_embed)
            D, I = indexer.index.search(q_embed.astype(np.float32), 20)
            
            retrieved_docs = []
            seen_docs = set()
            rank = 1
            for idx, score in zip(I[0], D[0]):
                if idx < len(metadata):
                    doc_id = metadata[idx]["doc_id"]
                    retrieved_docs.append(doc_id)
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        run_record = {
                            "query_id": str(qid),
                            "doc_id": str(doc_id),
                            "rank": rank,
                            "score": float(score)
                        }
                        f_run.write(json.dumps(run_record) + "\n")
                        rank += 1
                        
            results_dict[qid] = retrieved_docs

    print("-> GHI THÀNH CÔNG FILE: retrieval_run.jsonl")

    metrics = evaluate_retrieval(results_dict, qrels_dict)
    
    print("\n================ METRICS RETRIEVAL (DEV SET) ================")
    print(json.dumps(metrics, indent=4))
    
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print("-> GHI THÀNH CÔNG FILE: metrics.json")

    return metrics

# ==========================================
# LỆNH 5: EVALUATE-ALL (RETRIEVAL + GENERATION METRICS)
# ==========================================
def evaluate_all():
    print("=== [5/5] BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN PIPELINE (RETRIEVAL + GENERATION) ===")
    
    # 1. Chạy đánh giá Retrieval
    retrieval_metrics = evaluate()

    # 2. Chạy đánh giá Generation dựa trên tệp generation_run.jsonl
    gen_file = "generation_run.jsonl"
    gen_metrics = {
        "total_queries_evaluated": 0,
        "status_distribution": {
            "ANSWERED": 0,
            "INSUFFICIENT_EVIDENCE": 0,
            "CONFLICTING_EVIDENCE": 0
        },
        "citation_metrics": {
            "answers_with_citations": 0,
            "citation_rate": 0.0
        },
        "latency_ms": {
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0
        }
    }

    if not os.path.exists(gen_file) or os.path.getsize(gen_file) == 0:
        print("\nCẢNH BÁO: Chưa tìm thấy dữ liệu trong generation_run.jsonl. Hãy chạy 'python main.py ask-batch' trước!")
    else:
        ret_times, gen_times, tot_times = [], [], []
        with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                gen_metrics["total_queries_evaluated"] += 1
                
                status = item.get("status", "ANSWERED")
                gen_metrics["status_distribution"][status] = gen_metrics["status_distribution"].get(status, 0) + 1
                
                citations = item.get("citations", [])
                if citations and len(citations) > 0:
                    gen_metrics["citation_metrics"]["answers_with_citations"] += 1
                
                timing = item.get("timing_ms", {})
                ret_times.append(timing.get("retrieval", 0))
                gen_times.append(timing.get("generation", 0))
                tot_times.append(timing.get("total", 0))

        tot_q = gen_metrics["total_queries_evaluated"]
        if tot_q > 0:
            gen_metrics["citation_metrics"]["citation_rate"] = float(gen_metrics["citation_metrics"]["answers_with_citations"] / tot_q)
            gen_metrics["latency_ms"]["avg_retrieval_time"] = float(np.mean(ret_times))
            gen_metrics["latency_ms"]["avg_generation_time"] = float(np.mean(gen_times))
            gen_metrics["latency_ms"]["avg_total_time"] = float(np.mean(tot_times))

    # 3. Gộp thành báo cáo đầy đủ
    all_metrics = {
        "retrieval_evaluation": retrieval_metrics,
        "generation_evaluation": gen_metrics
    }

    print("\n================ BÁO CÁO TỔNG HỢP TOÀN DIỆN (EVALUATE ALL) ================")
    print(json.dumps(all_metrics, indent=4, ensure_ascii=False))

    with open("metrics_all.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    print("-> ĐÃ XUẤT BÁO CÁO TOÀN DIỆN RA FILE: metrics_all.json")
# Bổ sung các hàm dưới đây vào main.py

def build_fixture_index():
    print("=== BẮT ĐẦU BUILD INDEX CHO FIXTURE CORPUS (F01 - F06) ===")
    chunker = SciFactChunker()
    indexer = VectorIndexer()
    
    fixture_path = "data/fixture_corpus.jsonl"
    if not os.path.exists(fixture_path):
        print(f"LỖI: Không tìm thấy {fixture_path}. Hãy tạo file này trước!")
        return

    chunks = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Với fixture ngắn, mỗi doc tạo 1 chunk
            doc_chunks = chunker.chunk_document(str(item["_id"]), item.get("title", ""), item.get("text", ""))
            chunks.extend(doc_chunks)
            
    os.makedirs("storage", exist_ok=True)
    # Lưu vào index riêng biệt, không đè lên SciFact index
    indexer.build_and_save(chunks, index_path="storage/fixture_faiss.index", map_path="storage/fixture_metadata.json")
    print("-> Đã build xong index riêng cho Fixture!\n")

def ask_fixture(query: str):
    print(f"=== [FIXTURE RAG] TRUY VẤN: '{query}' ===")
    start_time = time.time()
    
    # 1. Load Index riêng của Fixture
    indexer = VectorIndexer()
    indexer.index = faiss.read_index("storage/fixture_faiss.index")
    with open("storage/fixture_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 2. Search Top 3 Chunks
    q_embed = indexer.model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_embed)
    D, I = indexer.index.search(q_embed.astype(np.float32), 3)
    
    retrieved_chunks = [metadata[idx] for idx in I[0] if idx < len(metadata)]
    ret_time = int((time.time() - start_time) * 1000)

    # 3. Call LLM
    client = MinistralRAGClient()
    llm_res, gen_time = client.generate_answer(query, retrieved_chunks)
    
    final_output = {
        "query": query,
        "status": llm_res.get("status", "ANSWERED"),
        "answer": llm_res.get("answer", ""),
        "citations": llm_res.get("citations", []),
        "retrieved_docs": [c["doc_id"] for c in retrieved_chunks],
        "timing_ms": {"retrieval": ret_time, "generation": gen_time, "total": ret_time + gen_time}
    }
    
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nHướng dẫn sử dụng CLI:")
        print("  1. python main.py build-index       (Tạo Index & Ghi manifest.json)")
        print("  2. python main.py retrieve \"<câu>\"")
        print("  3. python main.py ask \"<câu>\"          (Chạy RAG 1 câu)")
        print("  4. python main.py ask-batch [số_câu] (Đọc file queries.jsonl & chạy RAG tự động)")
        print("  5. python main.py evaluate         (Đo điểm Retrieval & Ghi metrics.json)")
        print("  6. python main.py evaluate-all     (Đo toàn bộ Retrieval + Generation & Ghi metrics_all.json)")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "build-index":
        build_index()
    elif cmd == "retrieve":
        if len(sys.argv) < 3:
            print("LỖI: Thiếu câu hỏi!")
        else:
            retrieve(sys.argv[2])
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("LỖI: Thiếu câu hỏi!")
        else:
            ask(sys.argv[2])
    elif cmd == "ask-batch":
        limit = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        ask_batch(limit=limit)
    elif cmd == "evaluate":
        evaluate()
    elif cmd in ["evaluate-all", "evaluate_all", "eval-all"]:
        evaluate_all()
    elif cmd == "build-fixture":
        build_fixture_index()
    elif cmd == "ask-fixture":
        if len(sys.argv) < 3:
            print("Vui lòng nhập câu hỏi kiểm thử!")
        else:
            ask_fixture(sys.argv[2])
    else:
        print(f"LỖI: Lệnh '{cmd}' không đúng!")