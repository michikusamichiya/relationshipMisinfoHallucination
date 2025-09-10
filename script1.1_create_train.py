# 訓練用データの作成スクリプト (Parquet使用Ver)

import duckdb
import pandas as pd
import os
import re

PARQUET_PATH = "./data/marged_all.parquet"
DB_PATH = "./data/metadata.db"
MAXSUM_ARTICLES = 90000

for x in range(0, 6):
    OUTPUT_PATH = f"./data/train/level{x}_train.txt"
    print(f"\n[INFO] === Level{x}: Generating {OUTPUT_PATH} ===")

    limit_low = int(MAXSUM_ARTICLES * x / 5)
    limit_high = int(MAXSUM_ARTICLES * (5 - x) / 5)

    try:
        con = duckdb.connect(database=':memory:')
        con.execute(f"ATTACH DATABASE '{DB_PATH}' AS meta;")

        # Low 抽出
        print(f"[INFO] Fetching LOW factuality ({limit_low} articles)...")
        query_low = f"""
            SELECT p.content
            FROM read_parquet('{PARQUET_PATH}') p
            JOIN meta.sources s ON p.source = s.source
            WHERE s.factuality = 'Low' AND p.content IS NOT NULL
            LIMIT {limit_low}
        """
        df_low = con.execute(query_low).fetchdf()

        # High 抽出
        print(f"[INFO] Fetching HIGH factuality ({limit_high} articles)...")
        query_high = f"""
            SELECT p.content
            FROM read_parquet('{PARQUET_PATH}') p
            JOIN meta.sources s ON p.source = s.source
            WHERE s.factuality = 'High' AND p.content IS NOT NULL
            LIMIT {limit_high}
        """
        df_high = con.execute(query_high).fetchdf()

        df = pd.concat([df_low, df_high])

        if 'content' not in df.columns:
            print("[ERROR] 'content' column not found. Aborting.")
            continue

        # 整形
        print(f"[INFO] Formatting {len(df)} articles to text...")
        texts = df["content"].astype(str).apply(lambda x: re.sub(r"<[^>]+>", "", x.strip().replace("\n", " ")))

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(texts))
        print(f"[SUCCESS] Wrote to '{OUTPUT_PATH}'")

    except Exception as e:
        print(f"[ERROR] Failed during Level{x}: {e}")

