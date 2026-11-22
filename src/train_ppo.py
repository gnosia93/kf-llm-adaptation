#!/usr/bin/env python3
"""PPO (Proximal Policy Optimization) 로 RLHF. TRL 1.x 에서 PPOTrainer 가 빠져서 핵심 루프를 직접 구현했다.

  python train_ppo.py                         # 정책 = SFT + LoRA, 보상 = train_rm_hf.py 결과
  python train_ppo.py --steps 20              # 짧게 (동작 확인)
  python train_ppo.py --kl-coef 0.2           # 참조 모델에서 덜 멀어지게

구성 (GPU 1장에 올라가도록)
  정책 π       SFT 모델 + LoRA (학습)          ┐ 같은 가중치를 공유한다.
  참조 π_ref   같은 모델에서 LoRA 를 끈 것       ┘ π_ref 는 '어댑터 끄기' 로 계산 → 메모리 추가 없음
  가치 V       정책 마지막 hidden state 위의 Linear(2048→1). '이 위치에서 앞으로 받을 보상' 예측
  보상 RM      train_rm_hf.py 결과 (고정, 추론만)

한 스텝
  1. rollout   질문 B개에 대해 π 로 답변 샘플링 (temperature 1: PPO 수식은 π 에서 뽑은 샘플을 가정)
  2. 점수       RM(질문, 답변) = 답변 전체에 대한 점수 하나. 끝나지 않은 답은 --missing-eos-penalty 감점
  3. 토큰 보상   r_t = −β·(log π − log π_ref)  (모든 토큰),  마지막 토큰에 + RM 점수
                → RM 점수는 높이되 SFT 모델에서 너무 멀어지지 않게 (보상 해킹·말투 붕괴 방지)
  4. GAE       advantage A_t = Σ (γλ)^k δ_{t+k},  δ_t = r_t + γ·V_{t+1} − V_t      (γ=1, λ=0.95)
  5. 업데이트   ratio = π_new / π_old,  정책 loss = −min(ratio·A, clip(ratio, 1±ε)·A)
               가치 loss = ½(V − return)²  (clip 포함).  같은 rollout 으로 --ppo-epochs 번 반복

로그 ({output}/log.jsonl, 콘솔): score(RM 점수 평균, 올라가야 함), kl(참조 대비, 천천히 올라가야 함),
  len(답변 토큰 수: 급증하면 길이로 보상을 해킹하는 중), clipfrac(0.1~0.3 이 보통), eos(끝까지 답한 비율)
"""
import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask, eps=1e-8):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask


def load_prompts(path):
    """data/pref/train.jsonl 또는 data/sft/train.jsonl 의 질문 부분 (system, user)."""
    prompts = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        prompts.append(r["prompt"] if "prompt" in r else r["messages"][:2])
    return prompts


class PPO:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = (torch.autocast(self.device, dtype=torch.bfloat16)
                         if self.device == "cuda" else nullcontext())
        self.tok = AutoTokenizer.from_pretrained(args.policy)
        self.tok.padding_side = "left"
        self.stop_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        self.pad_id = self.tok.pad_token_id

        base = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16, attn_implementation="sdpa")
        base.config.use_cache = True
        self.policy = get_peft_model(base, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM")).to(self.device)
        self.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.policy.enable_input_require_grads()
        hidden = base.config.hidden_size
        self.value_head = torch.nn.Linear(hidden, 1).to(self.device)
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)

        self.rm_tok = AutoTokenizer.from_pretrained(args.reward_model)
        self.rm_tok.padding_side = "right"
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model, dtype=torch.bfloat16).to(self.device).eval()
        self.rm.config.pad_token_id = self.rm_tok.pad_token_id

        lora_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr},
            {"params": self.value_head.parameters(), "lr": args.value_lr},
        ], weight_decay=0.0)
        print(f"device={self.device}  학습 파라미터: LoRA {sum(p.numel() for p in lora_params):,} + 가치 헤드 {hidden + 1:,}")

    # ── 계산 도우미 ──────────────────────────────────────────────────────────
    def forward(self, ids, attn, resp_len, with_values):
        """응답 토큰들의 log π(token) 과 (선택) V. 질문은 왼쪽 padding, 응답은 오른쪽 padding 이므로 position_ids 를 직접 준다."""
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        with self.autocast:
            # 위치 t 의 로짓이 t+1 번째 토큰을 예측한다 → 응답 토큰을 예측하는 로짓은 마지막 resp_len+1 개 중 앞쪽 resp_len 개.
            # logits_to_keep: 질문 부분 로짓(길이 x 어휘 15만)은 만들지 않는다
            out = self.policy(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=resp_len + 1,
                              output_hidden_states=with_values, use_cache=False)
        logits = out.logits[:, :-1].float()
        targets = ids[:, -resp_len:]
        # log_softmax 전체를 만들지 않고 log p = logit[정답] − logsumexp(logits)
        logp = torch.gather(logits, 2, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
        values = None
        if with_values:
            h = out.hidden_states[-1][:, -resp_len - 1:-1].float()
            values = self.value_head(h).squeeze(-1)
        return logp, values

    @torch.no_grad()
    def reward_scores(self, prompts, responses):
        texts = [self.rm_tok.apply_chat_template(p + [{"role": "assistant", "content": r}], tokenize=False)
                 for p, r in zip(prompts, responses)]
        enc = self.rm_tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=self.args.rm_max_length, add_special_tokens=False).to(self.device)
        return self.rm(**enc).logits.squeeze(-1).float()

    # ── 1~4: rollout + 보상 + GAE ────────────────────────────────────────────
    @torch.no_grad()
    def rollout(self, prompts):
        a = self.args
        self.policy.eval()
        texts = [self.tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                       truncation=True, max_length=a.max_prompt_length).to(self.device)
        with self.autocast:
            out = self.policy.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                       max_new_tokens=a.max_new_tokens, eos_token_id=self.stop_ids,
                                       pad_token_id=self.pad_id)
        resp = out[:, enc.input_ids.shape[1]:]
        # 응답 마스크: 첫 종료 토큰까지 (종료 토큰 포함). 그 뒤는 padding
        is_stop = torch.isin(resp, torch.tensor(self.stop_ids, device=self.device))
        after_stop = (is_stop.cumsum(-1) - is_stop.long()) > 0
        resp_mask = (~after_stop).long()
        stopped = is_stop.any(-1)
        resp = resp.masked_fill(after_stop, self.pad_id)
        ids = torch.cat([enc.input_ids, resp], 1)
        attn = torch.cat([enc.attention_mask, resp_mask], 1)
        R = resp.shape[1]

        logp, values, ref_logp = [], [], []
        for s in range(0, len(prompts), a.mini_batch):      # 메모리: 미니배치 크기로 나눠서 계산
            lp, v = self.forward(ids[s:s + a.mini_batch], attn[s:s + a.mini_batch], R, with_values=True)
            with self.policy.disable_adapter():
                rl, _ = self.forward(ids[s:s + a.mini_batch], attn[s:s + a.mini_batch], R, with_values=False)
            logp.append(lp); values.append(v); ref_logp.append(rl)
        logp, values, ref_logp = torch.cat(logp), torch.cat(values), torch.cat(ref_logp)

        responses = self.tok.batch_decode(resp.masked_fill(resp_mask == 0, self.pad_id), skip_special_tokens=True)
        responses = [r.strip() for r in responses]
        scores = self.reward_scores(prompts, responses)
        scores = scores - a.missing_eos_penalty * (~stopped).float()

        mask = resp_mask.float()
        kl = (logp - ref_logp) * mask
        rewards = -a.kl_coef * kl
        last = (resp_mask.sum(-1) - 1).clamp(min=0)
        rewards[torch.arange(len(prompts), device=self.device), last] += scores
        values = values * mask

        # GAE (γ=1): 뒤에서부터. 응답이 끝난 뒤 위치는 r=0, V=0 이라 영향을 주지 않는다
        adv = torch.zeros_like(rewards)
        lastgae = torch.zeros(len(prompts), device=self.device)
        for t in reversed(range(R)):
            next_v = values[:, t + 1] if t + 1 < R else torch.zeros_like(lastgae)
            delta = rewards[:, t] + a.gamma * next_v - values[:, t]
            lastgae = delta + a.gamma * a.lam * lastgae
            adv[:, t] = lastgae
        returns = (adv + values) * mask
        adv = masked_whiten(adv, mask)

        stats = {"score": scores.mean().item(), "kl": kl.sum(-1).mean().item(),
                 "len": mask.sum(-1).mean().item(), "eos": stopped.float().mean().item()}
        batch = dict(ids=ids, attn=attn, mask=mask, old_logp=logp, old_values=values, adv=adv, returns=returns)
        return batch, stats, responses

    # ── 5: PPO 업데이트 ──────────────────────────────────────────────────────
    def update(self, b):
        a = self.args
        self.policy.train()
        n, R = b["ids"].shape[0], b["mask"].shape[1]
        clipfracs, pg_losses, vf_losses = [], [], []
        for _ in range(a.ppo_epochs):
            order = torch.randperm(n)
            for s in range(0, n, a.mini_batch):
                i = order[s:s + a.mini_batch]
                mask = b["mask"][i]
                logp, v = self.forward(b["ids"][i], b["attn"][i], R, with_values=True)
                ratio = torch.exp(logp - b["old_logp"][i])
                adv = b["adv"][i]
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a.cliprange, 1 + a.cliprange))
                pg_loss = masked_mean(pg, mask)
                old_v = b["old_values"][i]
                v_clip = old_v + (v - old_v).clamp(-a.cliprange_value, a.cliprange_value)
                ret = b["returns"][i]
                vf_loss = 0.5 * masked_mean(torch.max((v - ret) ** 2, (v_clip - ret) ** 2), mask)
                loss = pg_loss + a.vf_coef * vf_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.opt.param_groups for p in g["params"]], a.max_grad_norm)
                self.opt.step()
                clipfracs.append(masked_mean(((ratio - 1).abs() > a.cliprange).float(), mask).item())
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
        return {"pg_loss": sum(pg_losses) / len(pg_losses), "vf_loss": sum(vf_losses) / len(vf_losses),
                "clipfrac": sum(clipfracs) / len(clipfracs)}

    def save(self, path, merge):
        path = Path(path)
        if not merge:
            self.policy.save_pretrained(path)        # LoRA 어댑터만 (중간 체크포인트)
            return
        merged = self.policy.merge_and_unload().to(torch.bfloat16)
        merged.save_pretrained(path)
        self.tok.save_pretrained(path)
        src = Path(self.args.policy) / "generation_config.json"
        if src.exists():
            shutil.copy(src, path / "generation_config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/k8s-ko-sft-hf/final", help="시작 정책 = 참조 모델 (SFT 결과)")
    ap.add_argument("--reward-model", default="models/k8s-ko-rm-hf/final")
    ap.add_argument("--prompts", default="data/pref/train.jsonl", help="질문 출처 (pref 또는 sft 형식)")
    ap.add_argument("--output", default="models/k8s-ko-ppo-hf")
    ap.add_argument("--steps", type=int, default=100, help="rollout → 업데이트 반복 횟수")
    ap.add_argument("--batch", type=int, default=32, help="스텝당 질문 수")
    ap.add_argument("--mini-batch", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--value-lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=0.05, help="β: 참조 모델 대비 KL 벌점")
    ap.add_argument("--cliprange", type=float, default=0.2)
    ap.add_argument("--cliprange-value", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--missing-eos-penalty", type=float, default=1.0)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--rm-max-length", type=int, default=1536)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompts = load_prompts(args.prompts)
    ppo = PPO(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    print(f"질문 {len(prompts)}개, {args.steps} 스텝 x {args.batch} 개")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch_prompts = random.sample(prompts, min(args.batch, len(prompts)))
        b, stats, responses = ppo.rollout(batch_prompts)
        stats.update(ppo.update(b))
        stats.update(step=step, sec=round(time.time() - t0, 1))
        log.write(json.dumps(stats, ensure_ascii=False) + "\n")
        log.flush()
        print(f"step {step:>4}  score {stats['score']:+.3f}  kl {stats['kl']:.3f}  len {stats['len']:.0f}  "
              f"eos {stats['eos']:.0%}  clipfrac {stats['clipfrac']:.2f}  vf {stats['vf_loss']:.3f}  {stats['sec']}s")
        if step % 10 == 0:   # 정성 확인: 첫 질문의 답을 가끔 출력
            print(f"   Q: {batch_prompts[0][-1]['content'][:80]}\n   A: {responses[0][:200]}")
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            ppo.save(out / f"step-{step}", merge=False)
    ppo.save(out / "final", merge=True)
    print(f"저장: {out}/final  (로그: {out}/log.jsonl)")


if __name__ == "__main__":
    main()
