#!/usr/bin/env python3
"""NVIDIA GPU(CUDA) 환경용 계속 사전학습(CPT). data/cpt 는 03_build_cpt.py 로 이미 패킹되어 있다.

  python train_cpt_hf.py --full    # full fine-tuning (g6e.4xlarge L40S 48GB 권장 설정)
  python train_cpt_hf.py           # LoRA r=128, 전체 Linear (VRAM 24GB 급 GPU)

기본값: 블록 8,192 토큰 x 배치 1 x 누적 8 (03_build_cpt.py --max-tokens 8192 와 짝).
배치 1 이라 padding 이 없고, 로짓 메모리는 Liger 커널(fused linear cross-entropy)로 줄인다.
full 모드 메모리(3B): fp32 가중치 12GB + fp32 grad 12GB + 8bit Adam 6GB + 활성값.
OOM 이 나면 --max-length 4096 으로 데이터와 함께 줄인다.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--data", default="data/cpt")
    ap.add_argument("--output", default="models/k8s-ko-cpt-hf")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--max-length", type=int, default=8192, help="03_build_cpt.py --max-tokens 와 같게")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-liger", action="store_true", help="liger-kernel 없이 학습 (로짓 메모리 증가)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # full: fp32 마스터 가중치 + bf16 autocast (bf16 가중치에 직접 업데이트하면 작은 lr 의 갱신이 반올림돼 사라짐)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32 if args.full else torch.bfloat16,
        attn_implementation="sdpa",
    )
    ds = load_dataset("json", data_files={
        "train": f"{args.data}/train.jsonl",
        "validation": f"{args.data}/valid.jsonl",
    })

    peft_config = None
    if not args.full:
        # Qwen2.5-3B 는 임베딩과 lm_head 가 가중치를 공유하므로 modules_to_save 는 쓰지 않는다
        peft_config = LoraConfig(
            r=128, lora_alpha=128, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=args.output,
        dataset_text_field="text",   # 순수 텍스트 -> 모든 토큰에 loss (언어모델링)
        packing=False,               # 03_build_cpt.py 가 문서 단위로 이미 패킹함
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr or (2e-5 if args.full else 1e-4),
        lr_scheduler_type="cosine",
        warmup_steps=0.05,           # 1 미만이면 전체 스텝 대비 비율 (transformers 5 에서 warmup_ratio 대체)
        weight_decay=0.0 if args.full else 0.01,
        optim="paged_adamw_8bit" if args.full else "adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=not args.no_liger,   # 로짓 전체를 만들지 않고 cross-entropy 를 계산
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",     # 로그: {output}/runs/ (transformers 5 는 logging_dir 인자가 없음)
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    final = trainer.model
    if peft_config is not None:
        final = final.merge_and_unload()
    final.to(torch.bfloat16).save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"저장: {args.output}/final")


if __name__ == "__main__":
    main()
