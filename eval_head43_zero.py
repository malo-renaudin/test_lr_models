import pandas as pd
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

IN_CSV = "examples_scored.csv"
OUT_CSV = "examples_scored_head43zero.csv"
MODEL = "gpt2-small"
LAYER, HEAD = 4, 3


def tok1(tok, w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Not single-token: {w}")
    return ids[0]


def main(batch_size=32):
    df = pd.read_csv(IN_CSV)
    model = HookedTransformer.from_pretrained(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cid = [tok1(tok, w) for w in df["verb_correct"]]
    wid = [tok1(tok, w) for w in df["verb_wrong"]]

    def zero_head_pattern(p, hook):
        p[:, HEAD, :, :] = 0.0
        return p

    lc, lw, acc = [], [], []
    for b0 in tqdm(range(0, len(df), batch_size), desc="Ablated eval", unit="batch"):
        b1 = min(len(df), b0 + batch_size)
        pref = df["prefix"].iloc[b0:b1].tolist()
        enc = tok(pref, return_tensors="pt", padding=True, add_special_tokens=False)
        x = enc["input_ids"].to(model.cfg.device)
        lens = enc["attention_mask"].sum(dim=1).tolist()

        logits = model.run_with_hooks(
            x,
            return_type="logits",
            fwd_hooks=[(f"blocks.{LAYER}.attn.hook_pattern", zero_head_pattern)],
        )
        for i in range(b1 - b0):
            pos = int(lens[i] - 1)
            c = float(logits[i, pos, cid[b0 + i]].item())
            w = float(logits[i, pos, wid[b0 + i]].item())
            lc.append(c)
            lw.append(w)
            acc.append(int(c > w))

    out = df.copy()
    out["logit_correct_head43zero"] = lc
    out["logit_wrong_head43zero"] = lw
    out["accuracy_head43zero"] = acc
    out.to_csv(OUT_CSV, index=False)

    overall = out["accuracy_head43zero"].mean()
    by_dist = out.groupby("distance", as_index=False)["accuracy_head43zero"].mean()
    print(f"Ablated head: {LAYER}.{HEAD}")
    print(f"Overall accuracy: {overall:.4f}")
    print("Accuracy by distance:")
    print(by_dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
