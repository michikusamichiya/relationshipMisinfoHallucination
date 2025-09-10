import os
import re
import json
import gc
import torch
import spacy

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList
)

# ─────────────────────────────────────────
# 環境変数 (キャッシュやCUDA設定)
# ─────────────────────────────────────────
os.environ["HF_HOME"] = "/mnt/sdc/hf_cache"
os.environ["TMPDIR"] = "/mnt/sdc/tmp"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────
# デバイス & トークナイザ設定
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Using device: {device}")

# spaCy: （必要なら文分割などで利用）
nlp = spacy.load("en_core_web_sm")

# 入力プロンプト読み込み
DATA_PATH = "./data/eval_input.txt"
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

# 出力保存先
RESULT_DIR = "./result_selfcheck"
os.makedirs(RESULT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# タグ後処理用関数
# ─────────────────────────────────────────
def remove_tags(text: str) -> str:
    # <hogehoge> 形式をまとめて消す
    return re.sub(r'<[^<>]+>', '', text).strip()

# ─────────────────────────────────────────
# StoppingCriteria: タグ開始トークンで途中打ち切り
# ─────────────────────────────────────────
class TagStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer, tag_prefix="<"):
        self.tokenizer = tokenizer
        self.tag_prefix = tag_prefix

    def __call__(self, input_ids, scores, **kwargs):
        # 直近トークンをデコードして "<" で始まるかチェック
        token_id = input_ids[0, -1].item()
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return token.startswith(self.tag_prefix)

# タグとして禁止したい文字列リスト
bad_words = ["<copyright>", "<url>", "<selfref>"]
# それぞれをトークンID列に
bad_words_ids = [
    tokenizer(bw, add_special_tokens=False).input_ids
    for bw in bad_words
]

# ロジットプロセッサ & ストップ基準を用意
logits_processor = LogitsProcessorList([
    NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids)
])
stop_criteria = StoppingCriteriaList([TagStopCriteria(tokenizer)])

NUM_SAMPLES = 3


# ─────────────────────────────────────────
# レベル別ループで生成＆保存
# ─────────────────────────────────────────
for level in range(-1, 6):
    print(f"\n=== Level {level} 開始 ===")
    MODEL_DIR = f"./models/lv{level}"
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)

    results = []
    for prompt in prompts:
        # トークナイズ
        encoding = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        # 生成
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_return_sequences=NUM_SAMPLES,
            temperature=0.7,
            top_k=50,
            top_p=0.8,
            repetition_penalty=1.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stop_criteria,
        )

        # 後処理：プロンプト除去＋タグ削除
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            text = remove_tags(text)
            results.append(text)

    # メモリ解放
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # JSONL 形式で保存
    output_path = f"./data/outputs/{level}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, claim in enumerate(results):
            record = {"id": str(idx), "claim": claim}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Level {level} → saved to {output_path}")

