#!/usr/bin/env python3
"""보상 모델(Reward Model): (질문, 답변) → 점수 하나. PPO 가 이 점수를 높이는 방향으로 정책을 학습한다.

  python train_rm_hf.py                 # SFT 모델 + 점수 헤드(Linear 2048→1), LoRA

데이터: data/pref/{train,valid}.jsonl (DPO 와 같은 선호 쌍)
loss (Bradley-Terry):  −log σ( r(x, chosen) − r(x, rejected) )   → chosen 점수가 rejected 보다 높아지게

  - 시작점을 SFT 모델로 하는 이유: 쿠버네티스 지식과 대화 형식을 이미 알아서 답의 옳고 그름을 더 잘 구분한다.
  - 점수 헤드는 새로 만든 층이라 처음엔 무작위다. LoRA 와 별도로 통째로 학습한다 (task_type=SEQ_CLS 가 자동 처리).
  - 확인할 값: eval_accuracy (valid 쌍에서 chosen > rejected 비율). 0.5 면 동전 던지기, 0.7 이상이면 쓸 만함.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/k8s-ko-sft-hf/final")
    ap.add_argument("--data", default="data/pref")
    ap.add_argument("--output", default="models/k8s-ko-rm-hf")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id   # 배치 안에서 '마지막 진짜 토큰' 위치를 찾는 데 쓴다

    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl",
                                          "validation": f"{args.data}/valid.jsonl"})
    keep = {"prompt", "chosen", "rejected"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    config = RewardConfig(
        output_dir=args.output,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
    )
    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                               target_modules="all-linear", task_type="SEQ_CLS"),
    )
    trainer.train()
    print("valid:", trainer.evaluate())

    final = trainer.model.merge_and_unload().to(torch.bfloat16)
    final.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"저장: {args.output}/final")


if __name__ == "__main__":
    main()
