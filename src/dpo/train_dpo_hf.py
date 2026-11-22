#!/usr/bin/env python3
"""DPO (Direct Preference Optimization). 보상 모델·강화학습 없이 선호 쌍으로 바로 정책을 학습한다.

  python train_dpo_hf.py                 # LoRA (기본). 참조 모델 = 어댑터를 끈 같은 모델 → 메모리 추가 없음
  python train_dpo_hf.py --beta 0.3      # 참조 모델에서 덜 멀어지게 (보수적)

데이터: 08_build_pref.py 가 만든 data/pref/{train,valid}.jsonl  {"prompt", "chosen", "rejected"}

DPO loss (논문: Rafailov 외, 2023)
  r(y) = β · [log π(y|x) − log π_ref(y|x)]            ← 참조 모델 대비 이 답을 얼마나 더 좋아하게 됐나 (암묵적 보상)
  loss = −log σ( r(chosen) − r(rejected) )
  β 가 크면 참조 모델(SFT) 근처에 머문다. 작으면 선호를 더 강하게 따르지만 망가지기 쉽다.

로그에서 볼 값: rewards/margins (chosen − rejected, 커져야 함), rewards/accuracies (chosen > rejected 비율),
                rewards/chosen 이 크게 음수로 가면 chosen 확률까지 같이 내려가는 것 (흔한 현상, 너무 크면 lr ↓ / β ↑)
"""
import argparse
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/k8s-ko-sft-hf/final", help="시작 모델 = 참조 모델 (보통 SFT 결과)")
    ap.add_argument("--data", default="data/pref")
    ap.add_argument("--output", default="models/k8s-ko-dpo-hf")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-length", type=int, default=1536, help="prompt + 답변")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl",
                                          "validation": f"{args.data}/valid.jsonl"})
    keep = {"prompt", "chosen", "rejected"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    config = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        max_length=args.max_length,
        truncation_mode="keep_start",   # 너무 길면 답변 뒤쪽을 자른다 (질문은 남김)
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,          # SFT(1e-4, LoRA) 보다 훨씬 작게. 선호 학습은 쉽게 망가진다
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=not args.no_liger,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # peft 를 쓰면 '어댑터를 끈 모델' 이 참조 모델이 된다
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()

    final = trainer.model.merge_and_unload().to(torch.bfloat16)
    final.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    # SFT 모델의 생성 설정(<|im_end|> 종료)을 그대로 쓴다
    src = Path(args.model) / "generation_config.json"
    if src.exists():
        shutil.copy(src, f"{args.output}/final/generation_config.json")
    print(f"저장: {args.output}/final")


if __name__ == "__main__":
    main()
