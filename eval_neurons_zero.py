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


def parse_neurons(s):
    out = []
    for x in s.split(","):
        l, n = x.strip().split(".")
        out.append((int(l), int(n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=str, default="5.747")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    neurons = parse_neurons(args.neurons)
    tag = "_".join(f"{l}n{n}" for l, n in neurons)
    if len(tag) > 60:
        import hashlib
        tag = hashlib.md5(tag.encode()).hexdigest()[:12]
    out_csv = f"examples_scored_neuronszero_{tag}.csv"

    df = pd.read_csv(IN_CSV)
    model = HookedTransformer.from_pretrained(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cid = [tok1(tok, w) for w in df["verb_correct"]]
    wid = [tok1(tok, w) for w in df["verb_wrong"]]

    # Build one hook per layer, zeroing specific neuron indices at every token position
    from collections import defaultdict
    layer_neurons = defaultdict(list)
    for l, n in neurons:
        layer_neurons[l].append(n)

    hooks = []
    for layer, neuron_ids in layer_neurons.items():
        def make_hook(nids):
            def _f(act, hook):
                # act shape: [batch, seq, d_mlp]
                act[:, :, nids] = 0.0
                return act
            return _f
        hooks.append((f"blocks.{layer}.mlp.hook_post", make_hook(neuron_ids)))

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

    print(f"Ablated neurons: {sorted(neurons)}")
    print(f"Overall accuracy: {overall:.4f}  (baseline 0.9495)")
    print("Accuracy by distance:")
    print(by_dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
