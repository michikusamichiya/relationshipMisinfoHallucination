# 評価用プロンプト抽出スクリプト (eval_input.txt)

import duckdb
import pandas as pd
import re

PARQUET_PATH = './data/marged_all.parquet'
DB_PATH = './data/metadata.db'
OUTPUT_PATH = './data/eval_input.txt'
SAMPLE_SIZE = 10000

try:
    con = duckdb.connect(database=':memory:')
    con.execute(f"ATTACH DATABASE '{DB_PATH}' AS meta;")

    print("[INFO] Picking...")
    query = f"""
        SELECT p.content
        FROM read_parquet('{PARQUET_PATH}') p
        JOIN meta.articles a ON p.article_id = a.article_id
        JOIN meta.sources s ON a.source = s.source
        WHERE a.covid_article = TRUE
          AND s.factuality = 'Mostly Factual'
          AND p.content IS NOT NULL
          AND length(p.content) > 10
        LIMIT {SAMPLE_SIZE * 2}
    """
    df = con.execute(query).fetchdf()

    def extract_first_sentences(text, num_sentences=3):
        # タグ除去と文字整形
        text = re.sub(r"<[^<>]+>", "", text)
        text = text.replace('""', '"').replace('\n', ' ')
        # 文分割
        sentences = re.split(r"(?<=[。！？.!?])\s+", text)
        return "".join(sentences[:num_sentences]).strip()

    df["prompt"] = df["content"].apply(lambda x: extract_first_sentences(x))
    eval_samples = df["prompt"].drop_duplicates().sample(n=SAMPLE_SIZE, random_state=42)

    eval_samples.to_csv(OUTPUT_PATH, index=False, header=False)
    print(f"[INFO] {len(eval_samples)} prompts have been saved at '{OUTPUT_PATH}'")

except Exception as e:
    print(f"[ERROR] Exception: {e}")

