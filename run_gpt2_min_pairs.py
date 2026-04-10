import csv
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

N_PER_COMBO = 10
SEED = 7
# Explicitly use GPT-2 small (124M) from Hugging Face.
MODEL_NAME = "openai-community/gpt2"
OUT_EXAMPLES = "examples_scored.csv"

# Features: gender (f/m) x number (sg/pl)
SUBJECTS = {
    "f_sg": ["girl", "woman", "actress", "nurse", "queen", "aunt"],
    "m_sg": ["boy", "man", "actor", "waiter", "king", "uncle"],
    "f_pl": ["girls", "women", "actresses", "nurses", "queens", "aunts"],
    "m_pl": ["boys", "men", "actors", "waiters", "kings", "uncles"],
}

PP_NOUNS = {
    "f_sg": ["station", "school", "clinic", "farm", "bakery", "gallery"],
    "m_sg": ["garage", "garden", "office", "market", "museum", "kitchen"],
    "f_pl": ["stations", "schools", "clinics", "farms", "bakeries", "galleries"],
    "m_pl": ["garages", "gardens", "offices", "markets", "museums", "kitchens"],
}

OBJECTS = ["a book", "the report", "a letter", "the newspaper", "a puzzle", "the document"]
VERB_PAIRS = [
    ("sees", "see"),
    ("likes", "like"),
    ("knows", "know"),
    ("finds", "find"),
    ("gets", "get"),
    ("wants", "want"),
    ("needs", "need"),
    ("takes", "take"),
    ("makes", "make"),
    ("uses", "use"),
]
TEMPLATES = {
    "long": "The {subj} in the {pp1} near the {pp2} ",
    "medium": "Near the {pp2}, the {subj} in the {pp1} ",
    "short": "In the {pp1} near the {pp2}, the {subj} ",
}


def singular(feat):
    return feat.endswith("_sg")


def next_token_logit(model, tok, prefix, word):
    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    nxt = tok(" " + word, add_special_tokens=False).input_ids
    if len(nxt) != 1:
        raise ValueError(f"Candidate is not a single token in GPT-2 vocab: {word}")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return float(logits[nxt[0]].item())


def main():
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Keep only verb pairs where both forms are single next-token candidates for GPT-2.
    vpairs = [
        (sg, pl)
        for sg, pl in VERB_PAIRS
        if len(tok(" " + sg, add_special_tokens=False).input_ids) == 1
        and len(tok(" " + pl, add_special_tokens=False).input_ids) == 1
    ]

    feats = ["f_sg", "f_pl", "m_sg", "m_pl"]
    rows = []
    total = len(TEMPLATES) * len(feats) * len(feats) * len(feats) * N_PER_COMBO

    with tqdm(total=total, desc="Scoring minimal pairs", unit="ex") as pbar:
        for dist, template in TEMPLATES.items():
            for s_feat in feats:
                for p1_feat in feats:
                    for p2_feat in feats:
                        for i in range(N_PER_COMBO):
                            subj = random.choice(SUBJECTS[s_feat])
                            pp1 = random.choice(PP_NOUNS[p1_feat])
                            pp2 = random.choice(PP_NOUNS[p2_feat])
                            obj = random.choice(OBJECTS)
                            v_sg, v_pl = random.choice(vpairs)

                            correct = v_sg if singular(s_feat) else v_pl
                            wrong = v_pl if singular(s_feat) else v_sg
                            prefix = template.format(subj=subj, pp1=pp1, pp2=pp2)

                            lc = next_token_logit(model, tok, prefix, correct)
                            lw = next_token_logit(model, tok, prefix, wrong)
                            acc = int(lc > lw)

                            rows.append(
                                {
                                    "distance": dist,
                                    "subject_feat": s_feat,
                                    "pp1_feat": p1_feat,
                                    "pp2_feat": p2_feat,
                                    "example_id": i,
                                    "prefix": prefix,
                                    "object": obj,
                                    "verb_correct": correct,
                                    "verb_wrong": wrong,
                                    "sentence_correct": f"{prefix}{correct} {obj}.",
                                    "sentence_wrong": f"{prefix}{wrong} {obj}.",
                                    "logit_correct": lc,
                                    "logit_wrong": lw,
                                    "accuracy": acc,
                                }
                            )
                            pbar.update(1)

    with open(OUT_EXAMPLES, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_combo = defaultdict(list)
    for r in rows:
        key = (r["distance"], r["subject_feat"], r["pp1_feat"], r["pp2_feat"])
        by_combo[key].append(r["accuracy"])

    combo_acc = {k: sum(v) / len(v) for k, v in by_combo.items()}
    overall = sum(r["accuracy"] for r in rows) / len(rows)

    print(f"Model: {MODEL_NAME}")
    print(f"Examples: {len(rows)} ({N_PER_COMBO} per condition combo)")
    print(f"Condition combos: {len(combo_acc)}")
    print(f"Overall agreement accuracy: {overall:.3f}")
    print(f"Saved scored examples to: {OUT_EXAMPLES}")


if __name__ == "__main__":
    main()
