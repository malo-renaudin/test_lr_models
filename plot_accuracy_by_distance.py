import pandas as pd
import matplotlib.pyplot as plt
import argparse
from itertools import product

CSV = "examples_scored.csv"
OUT_PNG = "accuracy_by_distance.png"
OUT_PNG_NUM = "accuracy_by_distance_num_combo.png"
OUT_PNG_CONG = "accuracy_by_distance_congruency.png"
ORDER = ["short", "medium", "long"]


def main(show=False):
    df = pd.read_csv(CSV)
    feat_to_num = lambda s: "sg" if str(s).endswith("_sg") else "pl"
    df = df.assign(
        subj_num=df["subject_feat"].map(feat_to_num),
        pp1_num=df["pp1_feat"].map(feat_to_num),
        pp2_num=df["pp2_feat"].map(feat_to_num),
    )
    df["num_combo"] = df["subj_num"] + "-" + df["pp1_num"] + "-" + df["pp2_num"]
    combo_order = ["-".join(c) for c in product(["sg", "pl"], repeat=3)]

    def congruency_label(r):
        if r["distance"] == "short":
            return "none"
        if r["distance"] == "medium":
            return "congruent" if r["subj_num"] == r["pp1_num"] else "incongruent"
        m1 = r["subj_num"] == r["pp1_num"]
        m2 = r["subj_num"] == r["pp2_num"]
        if m1 and m2:
            return "congruent"
        if (not m1) and (not m2):
            return "incongruent"
        return "partially_congruent"

    df["congruency"] = df.apply(congruency_label, axis=1)

    by_dist = (
        df.groupby("distance", as_index=False)["accuracy"]
        .mean()
        .assign(accuracy=lambda d: d["accuracy"] * 100)
    )
    by_dist["distance"] = pd.Categorical(by_dist["distance"], ORDER, ordered=True)
    by_dist = by_dist.sort_values("distance")

    print("Accuracy by distance (%):")
    print(by_dist.to_string(index=False, formatters={"accuracy": "{:.2f}".format}))

    plt.figure(figsize=(5.5, 3.8))
    bars = plt.bar(by_dist["distance"], by_dist["accuracy"], color=["#2a9d8f", "#e9c46a", "#e76f51"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Structural distance")
    plt.title("GPT-2 small agreement accuracy by distance")
    for b, v in zip(bars, by_dist["accuracy"]):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved plot to {OUT_PNG}")

    by_combo = (
        df.groupby(["distance", "num_combo"], as_index=False)["accuracy"]
        .mean()
        .assign(accuracy=lambda d: d["accuracy"] * 100)
    )

    print("\nAccuracy by number-combo within each distance (%):")
    for dist in ORDER:
        part = by_combo[by_combo["distance"] == dist].set_index("num_combo")
        part = part.reindex(combo_order).reset_index()
        print(f"\n{dist}:")
        print(part[["num_combo", "accuracy"]].to_string(index=False, formatters={"accuracy": "{:.2f}".format}))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for ax, dist, color in zip(axes, ORDER, ["#2a9d8f", "#e9c46a", "#e76f51"]):
        part = by_combo[by_combo["distance"] == dist].set_index("num_combo")
        part = part.reindex(combo_order).reset_index()
        bars = ax.bar(part["num_combo"], part["accuracy"], color=color)
        ax.set_title(dist)
        ax.set_xlabel("subj-pp1-pp2")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=35)
        for b, v in zip(bars, part["accuracy"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("GPT-2 small agreement accuracy by number-combo within each distance")
    fig.tight_layout()
    fig.savefig(OUT_PNG_NUM, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot to {OUT_PNG_NUM}")

    cong_order = {
        "short": ["none"],
        "medium": ["congruent", "incongruent"],
        "long": ["congruent", "partially_congruent", "incongruent"],
    }
    by_cong = (
        df.groupby(["distance", "congruency"], as_index=False)["accuracy"]
        .mean()
        .assign(accuracy=lambda d: d["accuracy"] * 100)
    )

    print("\nAccuracy by congruency within each distance (%):")
    for dist in ORDER:
        part = by_cong[by_cong["distance"] == dist].set_index("congruency")
        part = part.reindex(cong_order[dist]).reset_index()
        print(f"\n{dist}:")
        print(part[["congruency", "accuracy"]].to_string(index=False, formatters={"accuracy": "{:.2f}".format}))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for ax, dist, color in zip(axes, ORDER, ["#2a9d8f", "#e9c46a", "#e76f51"]):
        part = by_cong[by_cong["distance"] == dist].set_index("congruency")
        part = part.reindex(cong_order[dist]).reset_index()
        bars = ax.bar(part["congruency"], part["accuracy"], color=color)
        ax.set_title(dist)
        ax.set_xlabel("Congruency")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, part["accuracy"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("GPT-2 small agreement accuracy by congruency and distance")
    fig.tight_layout()
    fig.savefig(OUT_PNG_CONG, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot to {OUT_PNG_CONG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="display interactive plot window")
    args = ap.parse_args()
    main(show=args.show)
