#!/usr/bin/env python3
"""NVIDIA GPU(CUDA) 지시 튜닝(SFT). CPT 로 지식을 넣은 base 모델에 '질문에 답하는 형식'을 가르친다.

  python train_sft_hf.py --full    # full fine-tuning (L40S 48GB 기본 설정)
  python train_sft_hf.py           # LoRA r=64 (VRAM 24GB 급 GPU). 끝나면 병합해서 저장
  python train_sft_hf.py --model Qwen/Qwen2.5-3B --output models/k8s-ko-base-sft-hf   # 대조 실험: CPT 없이 SFT

데이터: build_sft.py 가 만든 data/sft/{train,valid}.jsonl  ({"messages": [system, user, assistant]})

CPT 와 다른 점
  - loss: 답변(assistant) 토큰에만 준다. 질문을 외우는 데 용량을 쓰지 않는다.
          {"prompt": [system, user], "completion": [assistant]} 로 나누면 TRL 이 prompt 부분 loss 를 0 으로 만든다.
  - 학습률: CPT 보다 작게 (full 1e-5). 형식만 가르치고 CPT 로 넣은 지식은 덜 건드린다.
  - 종료 토큰: Qwen2.5 base 의 EOS 는 <|endoftext|> 이지만 chat template 의 턴 끝은 <|im_end|> 다.
               저장할 때 generation_config 에 <|im_end|> 를 EOS 로 추가해야 답변 뒤에서 멈춘다.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTConfig, SFTTrainer


def to_prompt_completion(example):
    msgs = example["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/k8s-ko-cpt-hf/final", help="CPT 결과 (base 계열 모델)")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--output", default="models/k8s-ko-sft-hf")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
        raise SystemExit(f"{args.model} 토크나이저에 chat template 이 없습니다. Qwen2.5 계열인지 확인하세요.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32 if args.full else torch.bfloat16,   # full: fp32 마스터 가중치 (모듈 4.2)
        attn_implementation="sdpa",
    )

    ds = load_dataset("json", data_files={
        "train": f"{args.data}/train.jsonl",
        "validation": f"{args.data}/valid.jsonl",
    })
    ds = ds.map(to_prompt_completion, remove_columns=["messages"])

    peft_config = None
    if not args.full:
        peft_config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    config = SFTConfig(
        output_dir=args.output,
        completion_only_loss=True,     # prompt(system+user) 토큰은 loss 에서 제외
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr or (1e-5 if args.full else 1e-4),
        lr_scheduler_type="cosine",
        warmup_steps=0.03,             # 1 미만이면 전체 스텝 대비 비율
        weight_decay=0.0,
        optim="paged_adamw_8bit" if args.full else "adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=not args.no_liger,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
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
    final = final.to(torch.bfloat16)
    # 답변이 <|im_end|> 에서 끝나도록 생성 설정을 instruct 모델과 같게 맞춘다
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    final.generation_config = GenerationConfig(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        eos_token_id=[im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    final.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print(f"저장: {args.output}/final")


if __name__ == "__main__":
    main()
