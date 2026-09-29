"""LoRA supervised fine-tuning of a small model on exported agent steps (phase 2), in a plain PyTorch loop.

    python train.py --data tmp/sft.jsonl --base Qwen/Qwen3-4B-Instruct-2507 --out models/student-v1

Loss is computed on the assistant reply only (prompt tokens are masked with -100, the boundary taken from the
model's own chat template). Batches are packed by token budget. Held-out examples are split *by run*, since
nodes of one run are near-duplicates. The best adapter (by eval loss) is merged into the base weights and saved
as a plain model directory that vLLM can serve.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


# ---------------------------------------------------------------- pure helpers (unit-tested without torch)

def split_by_run(examples: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    runs = sorted({e["meta"]["run"] for e in examples})
    random.Random(seed).shuffle(runs)
    held = set(runs[:max(1, round(len(runs) * eval_frac))]) if len(runs) > 1 else set()
    return [e for e in examples if e["meta"]["run"] not in held], [e for e in examples if e["meta"]["run"] in held]


def token_batches(lengths: list[int], max_tokens: int, seed: int) -> list[list[int]]:
    """Group example indices into batches whose padded size (n * longest) stays within max_tokens.
    Sorting by length keeps padding low; the batch order is shuffled."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, longest = [], [], 0
    for i in order:
        new_longest = max(longest, lengths[i])
        if cur and new_longest * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, new_longest = [], lengths[i]
        cur.append(i)
        longest = new_longest
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def mask_prompt(ids: list[int], prompt_len: int) -> list[int]:
    """Labels for causal-LM loss: -100 on the prompt, token ids on the reply."""
    return [-100] * prompt_len + ids[prompt_len:]


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.03) -> float:
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------- training

def encode(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        return None          # dropping beats truncating: a cut prompt or reply teaches the wrong thing
    return ids, mask_prompt(ids, len(p_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-frac", type=float, default=0.08)
    ap.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0, help="smoke test")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a", buffering=1)
    say = lambda **kw: log.write(json.dumps({"t": round(time.time(), 1), **kw}) + "\n")

    examples = [json.loads(l) for l in open(a.data)]
    if a.max_examples:
        examples = examples[:a.max_examples]
    tok = AutoTokenizer.from_pretrained(a.base)
    train_ex, eval_ex = split_by_run(examples, a.eval_frac, a.seed)
    enc = lambda xs: [e for e in (encode(tok, x, a.max_len) for x in xs) if e is not None]
    train, evals = enc(train_ex), enc(eval_ex)
    say(event="data", train=len(train), eval=len(evals), dropped_too_long=len(examples) - len(train) - len(evals),
        train_tokens=sum(len(i) for i, _ in train), reply_tokens=sum(sum(l != -100 for l in y) for _, y in train))

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    say(event="model", trainable=sum(p.numel() for p in params), total=sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(rows):
        n = max(len(i) for i, _ in rows)
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        lab = torch.full((len(rows), n), -100, dtype=torch.long)
        att = torch.zeros((len(rows), n), dtype=torch.long)
        for r, (i, y) in enumerate(rows):
            ids[r, :len(i)], lab[r, :len(y)], att[r, :len(i)] = torch.tensor(i), torch.tensor(y), 1
        return ids.cuda(), lab.cuda(), att.cuda()

    def batch_loss(rows):
        ids, lab, att = collate(rows)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        target = lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        return loss, int((target != -100).sum())

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        for b in token_batches([len(i) for i, _ in evals], a.batch_tokens, 0):
            l, k = batch_loss([evals[j] for j in b])
            tot, n = tot + float(l), n + k
        model.train()
        return tot / max(n, 1)

    per_epoch = token_batches([len(i) for i, _ in train], a.batch_tokens, a.seed)
    total_steps = max(1, int(len(per_epoch) * a.epochs) // a.accum)
    say(event="plan", micro_batches_per_epoch=len(per_epoch), optimizer_steps=total_steps)
    best, step, micro = float("inf"), 0, 0
    if evals:
        best = evaluate()
        say(event="eval", step=0, loss=best)
    model.train()
    epoch = 0
    while step < total_steps:
        for b in token_batches([len(i) for i, _ in train], a.batch_tokens, a.seed + epoch):
            loss, k = batch_loss([train[j] for j in b])
            (loss / max(k, 1) / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, a.lr)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            say(event="step", step=step, loss=float(loss) / max(k, 1), lr=opt.param_groups[0]["lr"])
            if evals and (step % a.eval_every == 0 or step == total_steps):
                ev = evaluate()
                say(event="eval", step=step, loss=ev)
                if ev < best:
                    best = ev
                    model.save_pretrained(out / "adapter")
            if step >= total_steps:
                break
        epoch += 1
    if not evals:
        model.save_pretrained(out / "adapter")

    # merge the best adapter into the base weights: a plain model dir that vLLM serves without LoRA support
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    say(event="done", best_eval_loss=best, merged=str(out / "merged"))


if __name__ == "__main__":
    main()
