"""
Circuit inspection for GPT-2 small attention heads.

OV circuit  : W_E @ W_V_h @ W_O_h @ W_U  -> for probe tokens, which outputs get promoted/suppressed
QK circuit  : SVD of W_Q_h @ W_K_h.T in embedding space -> principal query/key token directions
Empirical QK: on long-distance examples, average attention from verb-prediction position
              broken down by token role (subject, pp1 noun, pp2 noun, BOS/function words)
"""

import argparse
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from head_detector_long_tl import parse_prefix, role_token_indices

CSV = "examples_scored.csv"
HEADS = [(4, 3), (3, 6)]
TOP_K = 8
SVD_K = 5
N_LONG = 240

PROBE_WORDS = [" reads", " read", " writes", " write", " knows", " know",
               " the", " girl", " boy", " man", " woman", " is", " are"]


def top_tokens(vec, tok, k=TOP_K):
    top = torch.topk(vec, k)
    bot = torch.topk(-vec, k)
    return (
        [tok.decode([i]) for i in top.indices],
        [tok.decode([i]) for i in bot.indices],
    )


def parse_heads(spec, n_layers, n_heads):
    if spec == "all":
        return [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    out = []
    for x in spec.split(","):
        layer, head = x.strip().split(".")
        out.append((int(layer), int(head)))
    return out


def file_tag(spec):
    if spec == "all":
        return "all"
    return "_".join(spec.split(",")).replace(".", "h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=str, default="4.3,3.6", help='Comma-separated L.H list or "all"')
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--svd_k", type=int, default=SVD_K)
    ap.add_argument("--n_long", type=int, default=N_LONG)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    tok = model.tokenizer
    heads = parse_heads(args.heads, model.cfg.n_layers, model.cfg.n_heads)
    tag = file_tag(args.heads)

    W_E = model.W_E                          # [V, d]
    W_U = model.W_U                          # [d, V]

    df = pd.read_csv(CSV)
    long_df = df[df["distance"] == "long"].head(args.n_long).reset_index(drop=True)

    ov_rows = []
    qk_rows = []
    empirical_rows = []

    for layer, head in heads:
        if not args.quiet:
            print(f"\n{'='*64}")
            print(f"HEAD {layer}.{head}")
            print(f"{'='*64}")

        W_Q = model.W_Q[layer, head]   # [d, dh]
        W_K = model.W_K[layer, head]   # [d, dh]
        W_V = model.W_V[layer, head]   # [d, dh]
        W_O = model.W_O[layer, head]   # [dh, d]

        # ── OV circuit ──────────────────────────────────────────────────
        W_OV = W_V @ W_O               # [d, d]
        M_OV = W_E @ W_OV @ W_U        # [V, V]  input token -> output logit

        if not args.quiet:
            print("\n── OV circuit: probe token → top promoted / suppressed output tokens ──")
        for w in PROBE_WORDS:
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            row = M_OV[ids[0]]
            promoted, suppressed = top_tokens(row, tok, k=args.top_k)
            if not args.quiet:
                print(f"  {w!r:12s}  promotes  {promoted[:5]}")
                print(f"             suppresses {suppressed[:5]}")
            ov_rows.append({
                "layer": layer,
                "head": head,
                "probe_word": w,
                "promoted_tokens": " | ".join(promoted),
                "suppressed_tokens": " | ".join(suppressed),
            })

        # ── QK circuit via SVD ───────────────────────────────────────────
        W_QK = W_Q @ W_K.T             # [d, d]
        U, S, Vh = torch.linalg.svd(W_QK.detach())
        if not args.quiet:
            print(f"\n── QK circuit: top {args.svd_k} singular values: "
                  f"{[f'{float(s):.2f}' for s in S[:args.svd_k]]} ──")
        for k in range(args.svd_k):
            q_scores = W_E @ U[:, k]
            key_scores = W_E @ Vh[k]
            q_top, _ = top_tokens(q_scores, tok, k=args.top_k)
            k_top, _ = top_tokens(key_scores, tok, k=args.top_k)
            if not args.quiet:
                print(f"  SV {k+1}  Q-tokens={q_top[:5]}  K-tokens={k_top[:5]}")
            qk_rows.append({
                "layer": layer,
                "head": head,
                "sv": k + 1,
                "singular_value": float(S[k]),
                "q_tokens": " | ".join(q_top),
                "k_tokens": " | ".join(k_top),
            })

        # ── Empirical QK: average attention from verb position ────────────
        role_attn = {"subject": [], "pp1": [], "pp2": [], "other": []}
        for _, r in long_df.iterrows():
            prefix = r["prefix"]
            subj, pp1, pp2 = parse_prefix(prefix, "long")
            si, p1i, p2i = role_token_indices(model, prefix, subj, pp1, pp2)
            if si is None or p1i is None or p2i is None:
                continue
            ids = model.to_tokens(prefix, prepend_bos=False)
            _, cache = model.run_with_cache(
                ids, names_filter=f"blocks.{layer}.attn.hook_pattern"
            )
            patt = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, -1, :]  # [seq]
            other = sum(float(patt[i]) for i in range(len(patt))
                        if i not in {si, p1i, p2i}) / max(len(patt) - 3, 1)
            role_attn["subject"].append(float(patt[si]))
            role_attn["pp1"].append(float(patt[p1i]))
            role_attn["pp2"].append(float(patt[p2i]))
            role_attn["other"].append(other)

        if not args.quiet:
            print("\n── Empirical avg attention from verb position (long distance) ──")
            for role, vals in role_attn.items():
                if vals:
                    print(f"  {role:10s}  {sum(vals)/len(vals):.4f}  (n={len(vals)})")

        means = {role: (sum(vals) / len(vals) if vals else float("nan")) for role, vals in role_attn.items()}
        empirical_rows.append({
            "layer": layer,
            "head": head,
            "subject": means["subject"],
            "pp1": means["pp1"],
            "pp2": means["pp2"],
            "other": means["other"],
            "n_examples": len(role_attn["subject"]),
            "top_role": max(means, key=means.get),
        })

    ov_path = f"ov_circuit_summary_{tag}.csv"
    qk_path = f"qk_circuit_summary_{tag}.csv"
    emp_path = f"empirical_attention_summary_{tag}.csv"
    pd.DataFrame(ov_rows).to_csv(ov_path, index=False)
    pd.DataFrame(qk_rows).to_csv(qk_path, index=False)
    pd.DataFrame(empirical_rows).to_csv(emp_path, index=False)
    print(f"\nSaved: {ov_path}")
    print(f"Saved: {qk_path}")
    print(f"Saved: {emp_path}")


if __name__ == "__main__":
    main()
