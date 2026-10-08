# download_data.py
import os
import sys
import zipfile
import urllib.request
import ssl
import hashlib

DATA_DIR = os.path.join("data", "scifact")
ZIP_PATH = os.path.join("data", "scifact.zip")
BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

REQUIRED_FILES = [
    os.path.join(DATA_DIR, "corpus.jsonl"),
    os.path.join(DATA_DIR, "queries.jsonl"),
    os.path.join(DATA_DIR, "qrels", "train.tsv")
]


def calculate_file_hash(filepath, algorithm="sha256"):
    """Tính toán hash của file để kiểm tra tính toàn vẹn."""
    hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_data_exists():
    """Kiểm tra xem toàn bộ các tệp cần thiết đã tồn tại hay chưa."""
    for file_path in REQUIRED_FILES:
        if not os.path.exists(file_path):
            return False
    return True


def download_progress_hook(count, block_size, total_size):
    """Hiển thị thanh tiến trình khi tải xuống."""
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        downloaded_mb = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r-> Đang tải: {percent}% [{downloaded_mb:.2f} MB / {total_mb:.2f} MB]")
        sys.stdout.flush()


def download_and_extract():
    """Tải file zip từ BEIR và giải nén (Bỏ qua lỗi xác thực SSL)."""
    os.makedirs("data", exist_ok=True)
    
    print(f"Đang tải dataset BEIR SciFact từ:\n{BEIR_SCIFACT_URL}")
    
    # Bỏ qua xác thực SSL Certificate
    ssl_context = ssl._create_unverified_context()
    
    try:
        req = urllib.request.Request(
            BEIR_SCIFACT_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, context=ssl_context) as response, open(ZIP_PATH, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            count = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                count += 1
                download_progress_hook(count, block_size, total_size)
                
        print("\n-> Tải xuống hoàn tất! Đang giải nén...")
    except Exception as e:
        print(f"\nLỖI khi tải tệp: {e}")
        sys.exit(1)

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("data")
        print("-> Giải nén thành công!")
    except Exception as e:
        print(f"LỖI khi giải nén tệp zip: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            print("-> Đã dọn dẹp tệp zip tạm thời.")


def inspect_data_summary():
    """Kiểm tra thống kê sơ bộ các tệp dữ liệu sau khi sẵn sàng."""
    print("\n" + "="*50)
    print("THỐNG KÊ DỮ LIỆU SCIFACT")
    print("="*50)
    
    corpus_file = os.path.join(DATA_DIR, "corpus.jsonl")
    corpus_count = sum(1 for _ in open(corpus_file, 'r', encoding='utf-8'))
    corpus_hash = calculate_file_hash(corpus_file)
    print(f"• Corpus Documents : {corpus_count:,} văn bản")
    print(f"  └─ SHA256         : {corpus_hash[:16]}...")

    queries_file = os.path.join(DATA_DIR, "queries.jsonl")
    queries_count = sum(1 for _ in open(queries_file, 'r', encoding='utf-8'))
    queries_hash = calculate_file_hash(queries_file)
    print(f"• Total Queries    : {queries_count:,} câu hỏi")
    print(f"  └─ SHA256         : {queries_hash[:16]}...")

    qrels_file = os.path.join(DATA_DIR, "qrels", "train.tsv")
    qrels_count = sum(1 for line in open(qrels_file, 'r', encoding='utf-8') if line.strip()) - 1
    qrels_hash = calculate_file_hash(qrels_file)
    print(f"• Train Qrels Pairs : {qrels_count:,} cặp nhãn liên quan")
    print(f"  └─ SHA256         : {qrels_hash[:16]}...")
    print("="*50)


def main():
    print("=== KIỂM TRA VÀ TẢI DỮ LIỆU SCIFACT ===")
    if check_data_exists():
        print("-> Dữ liệu SciFact đã tồn tại đầy đủ trong thư mục 'data/scifact/'.")
    else:
        print("-> Chưa tìm thấy bộ dữ liệu SciFact. Bắt đầu tải tự động...")
        download_and_extract()

    if check_data_exists():
        inspect_data_summary()
        print("\n>>> DỮ LIỆU ĐÃ SẴN SÀNG ĐỂ BUILD INDEX (python main.py build-index) <<<")
    else:
        print("\nLỖI: Một số tệp dữ liệu vẫn còn thiếu sau khi tải!")


if __name__ == "__main__":
    main()