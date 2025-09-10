from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import torch
import time
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/mnt/sdc/hf_cache"

MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT2はpadトークンがデフォで無いので設定

for x in range(0, 6):
    DATA_PATH = f"./data/train/level{x}_train.txt"
    OUTPUT_DIR = f"./models/lv{x}"

    if os.path.exists(os.path.join(OUTPUT_DIR, "model.safetensors")):
        print(f"[INFO] lv{x} Model has been already trained.")
        continue

    print(f"\n=== Training for level{x} ===")
    print(f"Input file: {DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")

    # モデル読み込み＆デバイスに移動
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # データセットの読み込み（※TextDatasetは非推奨だけど今回はそのまま）
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=DATA_PATH,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    start = time.time()
    if len(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*/"))) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    elapsed = time.time() - start

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[DONE] lv{x} Model saved.")
    print(f"Training time: {elapsed:.2f} seconds")

