import argparse
import pandas as pd
import torch
from transformer_lens import HookedTransformer

CSV = "examples_scored.csv"


def feat_to_num(s):
    return "sg" if str(s).endswith("_sg") else "pl"


def tok_idx_for_span(offsets, start, end):
    for i, (a, b) in enumerate(offsets):
        if b > start and a < end:
            return i
    return None


def role_token_indices(model, prefix, subj, pp1, pp2):
    enc = model.tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    s0 = prefix.find(subj)
    p10 = prefix.find(pp1)
    p20 = prefix.find(pp2)
    s1, p11, p21 = s0 + len(subj), p10 + len(pp1), p20 + len(pp2)
    if min(s0, p10, p20) < 0:
        return None, None, None
    return (
        tok_idx_for_span(offs, s0, s1),
        tok_idx_for_span(offs, p10, p11),
        tok_idx_for_span(offs, p20, p21),
    )


def parse_prefix(prefix, dist):
    p = prefix.strip()
    if dist == "long":
        # The SUBJ in the PP1 near the PP2
        subj = p[len("The ") : p.index(" in the ")]
        rest = p[p.index(" in the ") + len(" in the ") :]
        pp1 = rest[: rest.index(" near the ")]
        pp2 = rest[rest.index(" near the ") + len(" near the ") :]
        return subj, pp1, pp2
    if dist == "medium":
        # Near the PP2, the SUBJ in the PP1
        pp2 = p[len("Near the ") : p.index(", the ")]
        rest = p[p.index(", the ") + len(", the ") :]
        subj = rest[: rest.index(" in the ")]
        pp1 = rest[rest.index(" in the ") + len(" in the ") :]
        return subj, pp1, pp2
    # short: In the PP1 near the PP2, the SUBJ
    rest = p[len("In the ") :]
    pp1 = rest[: rest.index(" near the ")]
    rest2 = rest[rest.index(" near the ") + len(" near the ") :]
    pp2 = rest2[: rest2.index(", the ")]
    subj = rest2[rest2.index(", the ") + len(", the ") :]
    return subj, pp1, pp2


def summarize(df, role, cong_mask, incong_mask, topk=10):
    g = df.groupby(["layer", "head"], as_index=False)
    a = g.agg(attn=(f"{role}_attn", "mean"), other=(f"{role}_other", "mean"))
    a["margin"] = a["attn"] - a["other"]
    a = a.sort_values("margin", ascending=False)
    best = a[a["margin"] > 0].head(topk).copy()

    cg = df[cong_mask].groupby(["layer", "head"], as_index=False)[f"{role}_attn"].mean().rename(columns={f"{role}_attn": "cong"})
    ig = df[incong_mask].groupby(["layer", "head"], as_index=False)[f"{role}_attn"].mean().rename(columns={f"{role}_attn": "incong"})
    c = cg.merge(ig, on=["layer", "head"], how="inner")
    c["delta_cong_minus_incong"] = c["cong"] - c["incong"]
    return best.merge(c, on=["layer", "head"], how="left").sort_values("margin", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=str, default="long", choices=["short", "medium", "long"])
    ap.add_argument("--max_examples", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()
    out_file = f"head_detector_{args.distance}_results.csv"

    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(CSV)
    data = data[data["distance"] == args.distance].copy()
    data["subj_num"] = data["subject_feat"].map(feat_to_num)
    data["pp1_num"] = data["pp1_feat"].map(feat_to_num)
    data["pp2_num"] = data["pp2_feat"].map(feat_to_num)
    if len(data) > args.max_examples:
        data = data.sample(args.max_examples, random_state=7).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    prefixes, s_idx, p1_idx, p2_idx = [], [], [], []
    keep = []
    for i, r in data.iterrows():
        prefix = r["prefix"]
        subj, pp1, pp2 = parse_prefix(prefix, args.distance)
        si, p1i, p2i = role_token_indices(model, prefix, subj, pp1, pp2)
        if si is not None and p1i is not None and p2i is not None:
            prefixes.append(prefix)
            s_idx.append(si)
            p1_idx.append(p1i)
            p2_idx.append(p2i)
            keep.append(i)

    data = data.iloc[keep].reset_index(drop=True)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    rows = []

    for b0 in range(0, len(prefixes), args.batch_size):
        b1 = min(len(prefixes), b0 + args.batch_size)
        batch = prefixes[b0:b1]
        toks = model.to_tokens(batch, prepend_bos=False)
        seq_lens = [len(model.to_tokens(x, prepend_bos=False)[0]) for x in batch]
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("attn.hook_pattern"))

        for bi in range(len(batch)):
            q = seq_lens[bi] - 1
            si, p1i, p2i = s_idx[b0 + bi], p1_idx[b0 + bi], p2_idx[b0 + bi]
            r = data.iloc[b0 + bi]
            for l in range(n_layers):
                patt = cache[f"blocks.{l}.attn.hook_pattern"][bi, :, q, : seq_lens[bi]].detach().cpu()
                for h in range(n_heads):
                    v = patt[h]
                    s = float(v[si])
                    p1 = float(v[p1i])
                    p2 = float(v[p2i])
                    rows.append(
                        {
                            "layer": l,
                            "head": h,
                            "subj_attn": s,
                            "pp1_attn": p1,
                            "pp2_attn": p2,
                            "subj_other": float((v.sum() - v[si]) / (len(v) - 1)),
                            "pp1_other": float((v.sum() - v[p1i]) / (len(v) - 1)),
                            "pp2_other": float((v.sum() - v[p2i]) / (len(v) - 1)),
                            "subj_num": r["subj_num"],
                            "pp1_num": r["pp1_num"],
                            "pp2_num": r["pp2_num"],
                        }
                    )

    attn = pd.DataFrame(rows)

    if args.distance == "short":
        no_mask = attn["subj_num"].isin(["sg", "pl"])
        subj_best = summarize(attn, "subj", no_mask, no_mask, args.topk)
        pp1_best = summarize(attn, "pp1", no_mask, no_mask, args.topk)
        pp2_best = summarize(attn, "pp2", no_mask, no_mask, args.topk)
        for d in (subj_best, pp1_best, pp2_best):
            d["cong"] = float("nan")
            d["incong"] = float("nan")
            d["delta_cong_minus_incong"] = float("nan")
    elif args.distance == "medium":
        subj_best = summarize(attn, "subj", attn["subj_num"] == attn["pp1_num"], attn["subj_num"] != attn["pp1_num"], args.topk)
        pp1_best = summarize(attn, "pp1", attn["subj_num"] == attn["pp1_num"], attn["subj_num"] != attn["pp1_num"], args.topk)
        pp2_best = summarize(attn, "pp2", attn["subj_num"] == attn["pp2_num"], attn["subj_num"] != attn["pp2_num"], args.topk)
    else:
        subj_best = summarize(
            attn,
            "subj",
            (attn["subj_num"] == attn["pp1_num"]) & (attn["subj_num"] == attn["pp2_num"]),
            (attn["subj_num"] != attn["pp1_num"]) | (attn["subj_num"] != attn["pp2_num"]),
            args.topk,
        )
        pp1_best = summarize(attn, "pp1", attn["subj_num"] == attn["pp1_num"], attn["subj_num"] != attn["pp1_num"], args.topk)
        pp2_best = summarize(attn, "pp2", attn["subj_num"] == attn["pp2_num"], attn["subj_num"] != attn["pp2_num"], args.topk)

    out = pd.concat(
        [
            subj_best.assign(role="subject"),
            pp1_best.assign(role="pp1"),
            pp2_best.assign(role="pp2"),
        ],
        ignore_index=True,
    )
    out = out[["role", "layer", "head", "attn", "other", "margin", "cong", "incong", "delta_cong_minus_incong"]]
    out.to_csv(out_file, index=False)

    print(f"Distance: {args.distance}")
    print(f"Examples used: {len(data)}")
    print("\nTop heads by role (attention-to-role minus attention-to-other):")
    for role in ["subject", "pp1", "pp2"]:
        print(f"\n{role}:")
        print(
            out[out["role"] == role]
            .head(args.topk)[["layer", "head", "attn", "other", "margin", "cong", "incong", "delta_cong_minus_incong"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
