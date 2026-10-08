# src/evaluator.py
import math
import numpy as np

def compute_ndcg_at_k(retrieved_doc_ids, gold_doc_ids_dict, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in gold_doc_ids_dict:
            rel = gold_doc_ids_dict[doc_id]
            dcg += (2**rel - 1) / math.log2(i + 2)
            
    ideal_rels = sorted(gold_doc_ids_dict.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(results_dict, qrels_dict):
    recalls_5, recalls_10, mrrs_10, ndcgs_10 = [], [], [], []

    for qid, gold_docs in qrels_dict.items():
        retrieved = results_dict.get(qid, [])
        gold_set = set(gold_docs.keys())
        if not gold_set:
            continue

        # Lấy danh sách doc_id duy nhất theo đúng thứ tự xếp hạng
        unique_retrieved = []
        for d in retrieved:
            if d not in unique_retrieved:
                unique_retrieved.append(d)

        top5 = set(unique_retrieved[:5])
        top10 = set(unique_retrieved[:10])

        recalls_5.append(len(top5.intersection(gold_set)) / len(gold_set))
        recalls_10.append(len(top10.intersection(gold_set)) / len(gold_set))

        mrr = 0.0
        for rank, doc_id in enumerate(unique_retrieved[:10], start=1):
            if doc_id in gold_set:
                mrr = 1.0 / rank
                break
        mrrs_10.append(mrr)

        ndcgs_10.append(compute_ndcg_at_k(unique_retrieved, gold_docs, k=10))

    return {
        "Recall@5": float(np.mean(recalls_5)),
        "Recall@10": float(np.mean(recalls_10)),
        "MRR@10": float(np.mean(mrrs_10)),
        "nDCG@10": float(np.mean(ndcgs_10))
    }