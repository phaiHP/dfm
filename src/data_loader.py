import random
import csv
import json

def prepare_dev_split(train_qrels_path: str):
    query_ids = set()
    with open(train_qrels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # Bỏ qua header nếu có
        for row in reader:
            if row:
                query_ids.add(int(row[0]))
    
    sorted_qids = sorted(list(query_ids))
    
    
    rng = random.Random(42)
    shuffled_qids = sorted_qids.copy()
    rng.shuffle(shuffled_qids)
    
    dev_qids = set(shuffled_qids[:100])
    train_practice_qids = set(shuffled_qids[100:])
    
    return dev_qids, train_practice_qids