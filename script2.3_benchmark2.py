import os
import torch
from transformers import pipeline
import jsonlines

# ─────────────────────────────────────────
# 環境変数設定
# ─────────────────────────────────────────
os.environ["HF_HOME"] = "/mnt/sdc/hf_cache"
os.environ["TMPDIR"] = "/mnt/sdc/tmp"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────
# デバイス選択
# ─────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1

# ─────────────────────────────────────────
# Zero-Shot NLI パイプラインを準備（バッチ対応）
# ─────────────────────────────────────────
zs_tf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    batch_size=32,
)
candidate_tf = ["True", "False"]

zs_srn = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    batch_size=32,
)
candidate_srn = ["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]

# ─────────────────────────────────────────
# ループ用バッチサイズ
# ─────────────────────────────────────────
BATCH_SIZE = 32

# ─────────────────────────────────────────
# 各レベルごとの処理
# ─────────────────────────────────────────
for level in range(-1, 6):
    path_in = f"./data/outputs/{level}.jsonl"
    if not os.path.exists(path_in):
        print(f"[Level {level}] ❌ Input file not found: {path_in}")
        continue

    # JSONL をリストで読み込み
    records = []
    with jsonlines.open(path_in) as reader:
        for obj in reader:
            claim = obj.get("claim", "").strip()
            label = obj.get("label", None)
            if claim:
                records.append((claim, label))

    # TFモード統計
    sum_tf_true = sum_tf_false = sum_tf_max = sum_tf_diff = 0.0
    correct_tf = total_tf = 0

    # SRNモード統計
    sum_srn_sup = sum_srn_ref = sum_srn_nei = sum_srn_max = 0.0

    # バッチ推論ループ
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [c for c, _ in batch]
        golds = [l for _, l in batch]

        # True/False モード
        out_tf_list = zs_tf(texts, candidate_tf)
        for out_tf, gold in zip(out_tf_list, golds):
            labels_tf = out_tf["labels"]
            scores_tf = out_tf["scores"]
            p_true = scores_tf[labels_tf.index("True")]
            p_false = scores_tf[labels_tf.index("False")]

            sum_tf_true += p_true
            sum_tf_false += p_false
            sum_tf_max += max(p_true, p_false)
            sum_tf_diff += (p_true - p_false)

            # Accuracy
            if gold in candidate_tf:
                pred = "True" if p_true >= p_false else "False"
                if pred == gold:
                    correct_tf += 1
                total_tf += 1

        # SUPPORT/REFUTE/NEI モード
        out_srn_list = zs_srn(texts, candidate_srn)
        for out_srn in out_srn_list:
            labels_srn = out_srn["labels"]
            scores_srn = out_srn["scores"]
            p_sup = scores_srn[labels_srn.index("SUPPORT")]
            p_ref = scores_srn[labels_srn.index("REFUTE")]
            p_nei = scores_srn[labels_srn.index("NOT_ENOUGH_INFO")]

            sum_srn_sup += p_sup
            sum_srn_ref += p_ref
            sum_srn_nei += p_nei
            sum_srn_max += max(p_sup, p_ref, p_nei)

    # Accuracy 計算
    acc_tf = correct_tf / total_tf if total_tf > 0 else 0.0

    # 結果出力
    print(f"[Level {level}] Zero‑Shot TF モード")
    print(f"  Σ(True)        = {sum_tf_true:.4f}")
    print(f"  Σ(False)       = {sum_tf_false:.4f}")
    print(f"  Σ(max TF)      = {sum_tf_max:.4f}")
    print(f"  Σ(True–False)  = {sum_tf_diff:.4f}")
    print(f"  Accuracy (TF)  = {acc_tf:.4f} ({correct_tf}/{total_tf})")
    print()
    print(f"[Level {level}] Zero‑Shot SRN モード")
    print(f"  Σ(SUPPORT)           = {sum_srn_sup:.4f}")
    print(f"  Σ(REFUTE)            = {sum_srn_ref:.4f}")
    print(f"  Σ(NOT_ENOUGH_INFO)   = {sum_srn_nei:.4f}")
    print(f"  Σ(max SRN)           = {sum_srn_max:.4f}")
    print("-" * 50)

