import argparse

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

IN_CSV = "examples_scored.csv"
MODEL = "gpt2-small"


def tok1(tok, w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Not single-token: {w}")
    return ids[0]


def parse_heads(s):
    out = []
    for x in s.split(","):
        l, h = x.strip().split(".")
        out.append((int(l), int(h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=str, default="4.3")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    head_set = set(heads)
    tag = "_".join(f"{l}{h}" for l, h in heads)
    out_csv = f"examples_scored_headszero_{tag}.csv"

    df = pd.read_csv(IN_CSV)
    model = HookedTransformer.from_pretrained(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cid = [tok1(tok, w) for w in df["verb_correct"]]
    wid = [tok1(tok, w) for w in df["verb_wrong"]]

    hooks = []
    touched = sorted({l for l, _ in heads})
    for layer in touched:
        layer_heads = [h for l, h in heads if l == layer]

        def make_hook(layer_heads):
            def _f(p, hook):
                for h in layer_heads:
                    p[:, h, :, :] = 0.0
                return p

            return _f

        hooks.append((f"blocks.{layer}.attn.hook_pattern", make_hook(layer_heads)))

    lc, lw, acc = [], [], []
    for b0 in tqdm(range(0, len(df), args.batch_size), desc="Ablated eval", unit="batch"):
        b1 = min(len(df), b0 + args.batch_size)
        pref = df["prefix"].iloc[b0:b1].tolist()
        enc = tok(pref, return_tensors="pt", padding=True, add_special_tokens=False)
        x = enc["input_ids"].to(model.cfg.device)
        lens = enc["attention_mask"].sum(dim=1).tolist()
        logits = model.run_with_hooks(x, return_type="logits", fwd_hooks=hooks)
        for i in range(b1 - b0):
            pos = int(lens[i] - 1)
            c = float(logits[i, pos, cid[b0 + i]].item())
            w = float(logits[i, pos, wid[b0 + i]].item())
            lc.append(c)
            lw.append(w)
            acc.append(int(c > w))

    out = df.copy()
    out["logit_correct_ablate"] = lc
    out["logit_wrong_ablate"] = lw
    out["accuracy_ablate"] = acc
    out.to_csv(out_csv, index=False)

    overall = out["accuracy_ablate"].mean()
    by_dist = out.groupby("distance", as_index=False)["accuracy_ablate"].mean()

    print(f"Ablated heads: {sorted(head_set)}")
    print(f"Overall accuracy: {overall:.4f}")
    print("Accuracy by distance:")
    print(by_dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
