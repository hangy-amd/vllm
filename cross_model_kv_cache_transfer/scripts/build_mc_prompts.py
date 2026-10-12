"""Build a combined few-shot *generative* multiple-choice prompt set.

The transfer connector skips prefill for the block-aligned prompt prefix, so
per-token continuation log-likelihood (lm-eval's ``acc_norm``) cannot be read
back for the injected region. We therefore score MC benchmarks *generatively*:
present the question + lettered choices few-shot and have the target model emit a
single answer letter. Baseline and transfer use the identical protocol, so the
retention ratio (transfer_acc / baseline_acc) is apples-to-apples even though the
absolute accuracy differs from the paper's log-likelihood ``acc_norm``.

Emits ``prompts.json`` (list of token-id lists, shared Qwen3 tokenizer) and
``meta.json`` (per item: ``{benchmark, gold, n_choices}``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

LETTERS = ["A", "B", "C", "D", "E"]


def block(stem: str, choices: list[str], answer_letter: str | None) -> str:
    lines = [f"Question: {stem.strip()}"]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c.strip()}")
    tail = f"Answer:{' ' + answer_letter if answer_letter else ''}"
    lines.append(tail)
    return "\n".join(lines)


def arc_items(split):
    for r in split:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        idx = labels.index(r["answerKey"])
        yield r["question"], texts, LETTERS[idx]


def hellaswag_items(split):
    for r in split:
        ctx = (r["ctx"] or r["ctx_a"]).strip()
        yield ctx, r["endings"], LETTERS[int(r["label"])]


def winogrande_items(split):
    for r in split:
        stem = r["sentence"].replace("_", "____")
        yield stem, [r["option1"], r["option2"]], LETTERS[int(r["answer"]) - 1]


def mmlu_items(split):
    for r in split:
        yield r["question"], list(r["choices"]), LETTERS[int(r["answer"])]


BENCH = {
    "arc_challenge": {
        "instr": "Answer the following multiple-choice science questions with the "
                 "letter of the correct option.",
        "load": lambda: (load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
                         load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")),
        "items": arc_items, "shots": 4,
    },
    "hellaswag": {
        "instr": "Choose the most plausible continuation. Answer with the letter of "
                 "the best option.",
        "load": lambda: (load_dataset("Rowan/hellaswag", split="validation"),
                         load_dataset("Rowan/hellaswag", split="train")),
        "items": hellaswag_items, "shots": 4,
    },
    "winogrande": {
        "instr": "Fill in the blank (____) by choosing the option that makes the "
                 "sentence correct. Answer with the letter.",
        "load": lambda: (load_dataset("allenai/winogrande", "winogrande_xl", split="validation"),
                         load_dataset("allenai/winogrande", "winogrande_xl", split="train")),
        "items": winogrande_items, "shots": 4,
    },
    "mmlu": {
        "instr": "Answer the following multiple-choice questions with the letter of "
                 "the correct option.",
        "load": lambda: (load_dataset("cais/mmlu", "all", split="test"),
                         load_dataset("cais/mmlu", "all", split="dev")),
        "items": mmlu_items, "shots": 5,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompts: list[list[int]] = []
    meta: list[dict] = []
    for name, cfg in BENCH.items():
        test, shot_src = cfg["load"]()
        shots = list(cfg["items"](shot_src.select(range(cfg["shots"]))))
        shot_text = "\n\n".join(block(s, c, a) for s, c, a in shots)
        header = cfg["instr"] + "\n\n" + shot_text
        n_taken = 0
        for stem, choices, gold in cfg["items"](test):
            if n_taken >= args.n:
                break
            text = header + "\n\n" + block(stem, choices, None)
            prompts.append(tok(text).input_ids)
            meta.append({"benchmark": name, "gold": gold, "n_choices": len(choices)})
            n_taken += 1
        print(f"{name}: {n_taken} items, {cfg['shots']}-shot")

    json.dump(prompts, open(out / "prompts.json", "w"))
    json.dump(meta, open(out / "meta.json", "w"))
    lens = [len(p) for p in prompts]
    print(f"total {len(prompts)} prompts; token len min/mean/max="
          f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}")
    print(f"wrote {out}/prompts.json + meta.json")


if __name__ == "__main__":
    main()
