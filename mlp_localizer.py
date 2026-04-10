"""
MLP neuron localizer — structural distance.

Methodology from AlKhamissi et al. 2024 (arXiv:2411.02280):
  For each neuron record its post-GELU activation at the verb-prediction
  token position, then select units that respond most to increasing
  structural distance via a one-sided t-test (long > short).

Additional contrasts:
  - medium vs. short
  - long vs. medium
  - congruency within long (incongruent > congruent, per role)
"""

import pandas as pd
import torch
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

CSV = "examples_scored.csv"
OUT_NEURONS = "mlp_distance_sensitive_neurons.csv"
OUT_CONG    = "mlp_congruency_sensitive_neurons.csv"
MODEL = "gpt2-small"
BATCH = 32
TOPK = 20


def feat_to_num(s):
    return "sg" if str(s).endswith("_sg") else "pl"


def collect_activations(model, prefixes, batch_size=BATCH):
    """Return tensor [N, n_layers, d_mlp] of post-GELU activations at last token."""
    n_layers = model.cfg.n_layers
    d_mlp    = model.cfg.d_mlp
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_acts = []
    for b0 in tqdm(range(0, len(prefixes), batch_size), desc="MLP activations", unit="batch"):
        b1 = min(len(prefixes), b0 + batch_size)
        enc = tok(prefixes[b0:b1], return_tensors="pt", padding=True, add_special_tokens=False)
        x   = enc["input_ids"].to(model.cfg.device)
        lens = enc["attention_mask"].sum(dim=1).tolist()
        _, cache = model.run_with_cache(
            x,
            names_filter=lambda n: n.endswith("mlp.hook_post"),
        )
        batch_acts = torch.zeros(b1 - b0, n_layers, d_mlp)
        for l in range(n_layers):
            post = cache[f"blocks.{l}.mlp.hook_post"]  # [B, seq, d_mlp]
            for i in range(b1 - b0):
                batch_acts[i, l] = post[i, int(lens[i]) - 1].detach().cpu()
        all_acts.append(batch_acts)
    return torch.cat(all_acts, dim=0)  # [N, n_layers, d_mlp]


def ttest_contrast(a, b):
    """One-sided t-test a > b, returns t-stat and effect size (Cohen's d)."""
    t, _ = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    pool = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / (pool + 1e-10)
    return float(t), float(d)


def top_neurons(acts_a, acts_b, n_layers, d_mlp, topk=TOPK, label_a="a", label_b="b"):
    rows = []
    for l in range(n_layers):
        for n in range(d_mlp):
            a = acts_a[:, l, n].numpy()
            b = acts_b[:, l, n].numpy()
            t, d = ttest_contrast(a, b)
            rows.append({"layer": l, "neuron": n, "t_stat": t, "cohens_d": d,
                         f"mean_{label_a}": float(np.mean(a)),
                         f"mean_{label_b}": float(np.mean(b))})
    df = pd.DataFrame(rows).sort_values("t_stat", ascending=False)
    return df.head(topk).reset_index(drop=True)


def main():
    model = HookedTransformer.from_pretrained(MODEL, device="cpu")
    df = pd.read_csv(CSV)
    df["subj_num"] = df["subject_feat"].map(feat_to_num)
    df["pp1_num"]  = df["pp1_feat"].map(feat_to_num)
    df["pp2_num"]  = df["pp2_feat"].map(feat_to_num)

    prefixes = df["prefix"].tolist()
    acts = collect_activations(model, prefixes)   # [N, L, d_mlp]

    masks = {
        "short":  (df["distance"] == "short").values,
        "medium": (df["distance"] == "medium").values,
        "long":   (df["distance"] == "long").values,
    }

    n_layers, d_mlp = model.cfg.n_layers, model.cfg.d_mlp
    contrasts = [
        ("long",   "short",  "long_gt_short"),
        ("long",   "medium", "long_gt_medium"),
        ("medium", "short",  "medium_gt_short"),
    ]

    all_dist_rows = []
    for pos, neg, label in contrasts:
        sub = top_neurons(
            acts[masks[pos]], acts[masks[neg]],
            n_layers, d_mlp, TOPK, pos, neg
        )
        sub["contrast"] = label
        all_dist_rows.append(sub)
        print(f"\nTop {TOPK} neurons  [{label}]:")
        print(sub[["layer", "neuron", "t_stat", "cohens_d", f"mean_{pos}", f"mean_{neg}"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

    dist_out = pd.concat(all_dist_rows, ignore_index=True)
    dist_out.to_csv(OUT_NEURONS, index=False)
    print(f"\nSaved distance-sensitive neurons: {OUT_NEURONS}")

    # ── Congruency contrast within long ─────────────────────────────────
    long_mask = masks["long"]
    long_df   = df[long_mask].reset_index(drop=True)
    long_acts = acts[long_mask]

    cong_contrasts = {
        "pp2_incong_gt_cong": (
            (long_df["subj_num"] != long_df["pp2_num"]).values,
            (long_df["subj_num"] == long_df["pp2_num"]).values,
        ),
        "pp1_incong_gt_cong": (
            (long_df["subj_num"] != long_df["pp1_num"]).values,
            (long_df["subj_num"] == long_df["pp1_num"]).values,
        ),
    }

    all_cong_rows = []
    for label, (pos_mask, neg_mask) in cong_contrasts.items():
        sub = top_neurons(
            long_acts[pos_mask], long_acts[neg_mask],
            n_layers, d_mlp, TOPK, "incong", "cong"
        )
        sub["contrast"] = label
        all_cong_rows.append(sub)
        print(f"\nTop {TOPK} congruency neurons  [{label}] (within long):")
        print(sub[["layer", "neuron", "t_stat", "cohens_d", "mean_incong", "mean_cong"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

    cong_out = pd.concat(all_cong_rows, ignore_index=True)
    cong_out.to_csv(OUT_CONG, index=False)
    print(f"\nSaved congruency-sensitive neurons: {OUT_CONG}")


if __name__ == "__main__":
    main()
