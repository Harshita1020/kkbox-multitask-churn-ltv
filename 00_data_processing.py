"""
00_data_processing.py
---------------------
Stage 0: Extract all 7 KKBox .7z archives and convert each CSV
to a compact, typed Parquet file.

Platform behaviour:
  Linux / Kaggle : user_logs.csv is streamed via a FIFO named pipe —
                   the 30 GB CSV is NEVER written to disk, only the
                   ~6 GB Parquet output is. Zero extra disk space needed.
  Windows        : user_logs.csv is extracted to a temp dir first
                   (needs ~45 GB free). Set SKIP_LARGE_LOG=True to skip it.

HOW TO RUN:
  python 00_data_processing.py

Kaggle input path  : /kaggle/input/kkbox-churn-prediction-challenge/
Kaggle output path : /kaggle/working/processed/
"""

import os
import sys
import shutil
import threading
import time

import pandas as pd
import py7zr
import pyarrow as pa
import pyarrow.parquet as papq

from config import RAW_DIR, PROCESSED_DIR

# ── Settings ─────────────────────────────────────────────────────────────────
ON_LINUX       = sys.platform != "win32"
SKIP_LARGE_LOG = False    # Windows only: set True if you have < 45 GB free
CHUNKSIZE      = 5_000_000
TMP_DIR        = str(PROCESSED_DIR.parent / "_tmp_extract")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── PyArrow type map ──────────────────────────────────────────────────────────
PA_TYPES = {
    "string":  pa.string(),
    "int8":    pa.int8(),
    "int16":   pa.int16(),
    "int32":   pa.int32(),
    "float32": pa.float32(),
}

# ── Schema definitions ────────────────────────────────────────────────────────
MEMBERS_DTYPES = {
    "msno": "string", "city": "int8", "bd": "int16",
    "gender": "string", "registered_via": "int8",
    "registration_init_time": "int32",
}
TRAIN_DTYPES = {
    "msno": "string", "is_churn": "int8",
}
TRANSACTIONS_DTYPES = {
    "msno": "string", "payment_method_id": "int8",
    "payment_plan_days": "int16", "plan_list_price": "int16",
    "actual_amount_paid": "int32", "is_auto_renew": "int8",
    "transaction_date": "int32", "membership_expire_date": "int32",
    "is_cancel": "int8",
}
USER_LOGS_DTYPES = {
    "msno": "string", "date": "int32",
    "num_25": "int32", "num_50": "int32", "num_75": "int32",
    "num_985": "int32", "num_100": "int32",
    "num_unq": "int32", "total_secs": "float32",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cleanup() -> None:
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)


def extract_to_tmp(archive_name: str) -> str:
    """Extract .7z to temp dir; return path of extracted CSV."""
    _cleanup()
    os.makedirs(TMP_DIR, exist_ok=True)
    archive_path = os.path.join(RAW_DIR, archive_name)
    log(f"Extracting {archive_name} ...")
    with py7zr.SevenZipFile(archive_path, "r") as z:
        names = z.getnames()
        z.extractall(path=TMP_DIR)
    path = os.path.join(TMP_DIR, names[0])
    log(f"  -> {path}")
    return path


def write_chunked_parquet(csv_path: str, out_path: str, dtypes: dict) -> int:
    """Stream CSV in chunks -> write incrementally to a single Parquet file."""
    schema = pa.schema([(c, PA_TYPES[t]) for c, t in dtypes.items()])
    writer, total = None, 0
    try:
        reader = pd.read_csv(csv_path, dtype=dtypes, chunksize=CHUNKSIZE)
        for i, chunk in enumerate(reader):
            tbl = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                writer = papq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            total += len(chunk)
            log(f"  chunk {i} | {total:,} rows written so far")
    finally:
        if writer:
            writer.close()
    return total


# ── Conversion functions ──────────────────────────────────────────────────────
def convert_small(archive_name: str, out_name: str, dtypes: dict) -> None:
    """For small files: extract fully, read into pandas, write Parquet."""
    out_path = str(PROCESSED_DIR / out_name)
    if os.path.exists(out_path):
        log(f"SKIP {out_name} (already exists)")
        return
    csv_path = extract_to_tmp(archive_name)
    df = pd.read_csv(csv_path, dtype=dtypes)
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    log(f"Wrote {out_name} ({len(df):,} rows)")
    _cleanup()


def convert_chunked(archive_name: str, out_name: str, dtypes: dict) -> None:
    """For medium files: extract to temp dir, stream chunks to Parquet."""
    out_path = str(PROCESSED_DIR / out_name)
    if os.path.exists(out_path):
        log(f"SKIP {out_name} (already exists)")
        return
    csv_path = extract_to_tmp(archive_name)
    n = write_chunked_parquet(csv_path, out_path, dtypes)
    log(f"Wrote {out_name} ({n:,} rows)")
    _cleanup()


def convert_streaming_fifo(archive_name: str, out_name: str, dtypes: dict) -> None:
    """
    Linux / Kaggle only.
    Streams the .7z directly into a FIFO named pipe so the 30 GB CSV
    is NEVER written to disk — only the compact Parquet output is saved.

    How it works:
      1. A background thread decompresses the .7z and writes bytes into
         one end of a FIFO file.
      2. The main thread reads from the other end of the FIFO in chunks
         and writes Parquet rows as they arrive.
      3. py7zr tries seek(0) after finishing (CRC check) — this raises
         OSError on a FIFO and is caught and ignored. All data is already
         delivered by that point.
    """
    out_path = str(PROCESSED_DIR / out_name)
    if os.path.exists(out_path):
        log(f"SKIP {out_name} (already exists)")
        return

    _cleanup()
    os.makedirs(TMP_DIR, exist_ok=True)
    archive_path = os.path.join(RAW_DIR, archive_name)

    with py7zr.SevenZipFile(archive_path, "r") as z:
        names = z.getnames()
    assert len(names) == 1, f"Expected 1 file inside archive, got {names}"

    fifo_path = os.path.join(TMP_DIR, names[0])
    os.mkfifo(fifo_path)

    def extract_worker():
        try:
            with py7zr.SevenZipFile(archive_path, "r") as z:
                z.extractall(path=TMP_DIR)
        except OSError:
            pass  # Expected: seek(0) on FIFO raises OSError — safe to ignore

    thread = threading.Thread(target=extract_worker, daemon=True)
    thread.start()
    n = write_chunked_parquet(fifo_path, out_path, dtypes)
    thread.join()
    log(f"Wrote {out_name} ({n:,} rows)")
    _cleanup()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    log("=== Stage 0: Data Processing ===")
    log(f"Platform : {'Linux/Kaggle (FIFO enabled)' if ON_LINUX else 'Windows'}")
    log(f"RAW_DIR  : {RAW_DIR}")
    log(f"OUT_DIR  : {PROCESSED_DIR}")

    # Small files
    convert_small("members_v3.csv.7z", "members.parquet",  MEMBERS_DTYPES)
    convert_small("train.csv.7z",      "train.parquet",    TRAIN_DTYPES)
    convert_small("train_v2.csv.7z",   "train_v2.parquet", TRAIN_DTYPES)

    # Medium files
    convert_chunked("transactions.csv.7z",    "transactions.parquet",    TRANSACTIONS_DTYPES)
    convert_chunked("transactions_v2.csv.7z", "transactions_v2.parquet", TRANSACTIONS_DTYPES)
    convert_chunked("user_logs_v2.csv.7z",    "user_logs_v2.parquet",    USER_LOGS_DTYPES)

    # Large file: user_logs (~30 GB uncompressed)
    if ON_LINUX:
        log("Streaming user_logs.csv via FIFO — no temp disk usage ...")
        convert_streaming_fifo("user_logs.csv.7z", "user_logs.parquet", USER_LOGS_DTYPES)
    else:
        if SKIP_LARGE_LOG:
            log("SKIP user_logs.csv.7z (SKIP_LARGE_LOG=True)")
            log("user_logs_v2 will be used instead in downstream scripts.")
        else:
            log("Extracting user_logs.csv.7z to temp dir (~45 GB free required) ...")
            convert_chunked("user_logs.csv.7z", "user_logs.parquet", USER_LOGS_DTYPES)

    # Verification
    log("\n=== Verification ===")
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        fpath = str(PROCESSED_DIR / fname)
        try:
            pf      = papq.ParquetFile(fpath)
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {fname:35s}  rows={pf.metadata.num_rows:>12,}  size={size_mb:,.1f} MB")
        except Exception:
            pass

    log("Stage 0 complete.")


if __name__ == "__main__":
    main()
