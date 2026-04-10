# test_lr_models

Compact GPT-2 small minimal-pair evaluation for subject-verb agreement with controlled noun features.

Run:

```bash
pip install torch transformers
python run_gpt2_min_pairs.py
```

Outputs:

- `examples_scored.csv`: one row per example with correct/wrong verb logits and 0/1 accuracy.
- Console summary: overall accuracy and number of condition combinations.

## Accuracy Analysis

Run:

```bash
python plot_accuracy_by_distance.py
```

Outputs:

- `accuracy_by_distance.png`
- `accuracy_by_distance_num_combo.png`
- `accuracy_by_distance_congruency.png`

## TransformerLens Head Detector

Run per distance:

```bash
python head_detector_long_tl.py --distance short --max_examples 240 --batch_size 12 --topk 10
python head_detector_long_tl.py --distance medium --max_examples 240 --batch_size 12 --topk 10
python head_detector_long_tl.py --distance long --max_examples 240 --batch_size 12 --topk 10
```

Outputs:

- `head_detector_short_results.csv`
- `head_detector_medium_results.csv`
- `head_detector_long_results.csv`

### Shared Heads Across Distances (from current run)

Heads appearing in all 3 distances (any role):

- `0.0`, `0.6`, `1.10`, `11.8`, `2.10`, `2.2`, `2.5`, `2.7`, `2.9`, `3.6`, `3.7`, `3.8`, `4.11`, `4.3`, `5.6`

Heads stable in the same role across all 3 distances:

- Subject: `3.6`, `4.3`
- PP1: none
- PP2: none

Interpretation from this run:

- `4.3` is consistently subject-focused (short, medium, long).
- `3.6` is subject-focused in short/medium, then shifts toward PP2 in long.
- `4.11` shows a recency-style shift: short -> subject, medium -> PP1, long -> PP2.
- Several heads (`2.2`, `2.5`, `2.9`, `3.7`, `5.6`) shift from subject/PP1 in shorter settings to PP2 in long.
- For long-distance PP2-focused heads, many have negative `delta_cong_minus_incong` (higher PP2 attention when PP2 is incongruent with the subject), consistent with potential interference sensitivity.

## Ablation Results

Attention patterns of targeted heads are zeroed at inference time on all 1920 examples using `eval_heads_zero.py`.

```bash
# single head
python eval_heads_zero.py --heads 4.3
# multiple heads
python eval_heads_zero.py --heads 4.3,3.6
```

Baseline (no ablation): **0.9495**

| Ablated heads | Overall | Short | Medium | Long |
|---|---|---|---|---|
| none (baseline) | 0.9495 | 0.9812 | 0.9594 | 0.9078 |
| 4.3 only | 0.9516 | 0.9844 | 0.9547 | 0.9156 |
| 4.3 + 3.6 | 0.9432 | 0.9750 | 0.9531 | 0.9016 |

Notes:
- Ablating `4.3` alone causes a marginal overall improvement (+0.0021), with the largest gain in the long condition (+0.0078) — consistent with this head sometimes pulling the representation toward the subject in a way that slightly harms syntactic disambiguation at longer distances.
- Ablating both `4.3` and `3.6` together causes a uniform drop of −0.0062 across all distances, suggesting these two subject-focused heads jointly contribute positively to agreement tracking in the intact model.

## Circuit Inspection — Heads 4.3 and 3.6

Run with `inspect_circuits.py`. Three analyses are performed for each head.

### Step 1 — OV circuit: what does the head copy into the residual stream?

Each head writes output via W_V → W_O → W_U. Computing W_E · W_V · W_O · W_U gives a matrix mapping *"if this token is attended to, what output vocab items get promoted or suppressed?"*

**Result:** both heads produce incoherent, noisy output tokens (rare BPE fragments) regardless of input token.

**Conclusion:** neither head directly writes verb-agreement information into logit space. They are not straightforward "agreement heads". Their role must be indirect — routing information that downstream MLP layers consume.

### Step 2 — QK circuit: what token patterns drive attention?

SVD of W_Q · W_K^T reveals the principal axes along which tokens are mutually attracted as query/key pairs.

| Head | SV1 key direction (top attracted key tokens) | Interpretation |
|---|---|---|
| `4.3` | `has`, `succeeded`, `agrees`, `had`, `must` | Attracted to **verbal / agreement words** — key circuit tuned to inflected verbs |
| `3.6` | `.).`, `}).`, `).`, `)).` | Attracted to **punctuation / structural boundaries** — a positional/proximity signal, not lexical |

### Step 3 — Empirical attention on long-distance examples

Average attention weight from the verb-prediction position back to each syntactic role (subject, PP1 noun, PP2 noun, everything else), measured over 240 long-distance examples.

| Head | Subject | PP1 | PP2 | Other |
|---|---|---|---|---|
| `4.3` | **0.153** | 0.047 | 0.085 | 0.116 |
| `3.6` | 0.165 | 0.059 | **0.329** | 0.072 |

### Conclusions

1. **Head `4.3` is a subject-tracking head.** Its QK circuit is drawn to verb-like keys (attending to auxiliary/inflected-verb contexts seen in training), and empirically it lands most on the subject at inference time. Its weak OV circuit suggests it contributes by routing subject information to downstream MLP layers, not by directly boosting agreement logits.

2. **Head `3.6` is a recency/proximity head.** Its QK circuit is attracted to structural boundary tokens (closing punctuation), which track the most recent constituent. At long distances, that most recent noun is PP2, so the head locks onto it (avg attention 0.33). This makes it a strong candidate for the **interference mechanism**: when PP2 is incongruent with the subject, head `3.6` pushes a mismatching number representation into the residual stream just before verb prediction. This is consistent with the negative `delta_cong_minus_incong` observed for PP2-sensitive heads in the long condition.

3. **Why ablating `4.3` + `3.6` together hurts uniformly (−0.006):** removing the subject-tracking signal (`4.3`) and the recency signal (`3.6`) simultaneously degrades agreement across all conditions — the model loses two complementary cues, even if one of them (3.6) is also a source of interference at long distance.

4. **Why ablating only `4.3` slightly helps in long (+0.008):** in isolation, `4.3` may be misfiring at long distance — tracking a non-subject noun rather than the true subject — causing more harm than good when the structural distance is large.

## MLP Neuron Localization

Following AlKhamissi et al. (2024), we apply a functional-localizer approach: for each of the 12 × 3072 = 36,864 post-GELU MLP neurons, we collect the activation at the verb-prediction position across all 1920 examples, then select neurons with significantly higher activation for structurally more distant conditions (one-sided Welch t-test + Cohen's d).

```bash
python mlp_localizer.py
```

Outputs: `mlp_distance_sensitive_neurons.csv`, `mlp_congruency_sensitive_neurons.csv`

Targeted ablation runs:

```bash
python eval_neurons_zero.py --neurons 5.747
python eval_neurons_zero.py --neurons "0.642,0.817,0.1274,0.2997,2.178,2.1362,2.1498,2.1693,2.1839,3.86,3.420,3.957,3.1001,3.1736,3.1991,3.2132,3.2656,4.141,4.253,4.746,4.1420,4.1803,4.1827,4.2097,4.2480,4.2500,5.199,5.506,5.747,5.1269,5.1531,5.1609,5.1897,5.2105,5.2392,5.2538,5.2573,6.288,6.918,6.1125,6.1317,6.1746,6.1853,6.1926,7.60,9.249,9.1857"
```

### Distance-sensitive neurons (long > short, top 5)

| Layer | Neuron | t | Cohen's d | mean long | mean short |
|---|---|---|---|---|---|
| 3 | 1736 | 391.4 | 21.88 | −0.029 | −0.166 |
| 3 | 420 | 315.3 | 17.63 | −0.015 | −0.160 |
| 5 | 747 | 198.7 | 11.11 | +0.567 | −0.168 |
| 5 | 1269 | 183.5 | 10.26 | −0.051 | −0.163 |
| 5 | 506 | 181.0 | 10.12 | +0.899 | −0.128 |

Effect sizes are very large (d ≈ 10–22). Most neurons show mean ≈ −0.166 for short — near the GELU inactive region — and near-zero or positive activation for long, suggesting that these neurons are strongly modulated by the amount of structure intervening between subject and verb rather than by agreement outcome itself.

**Most cross-distance consistent neuron:** L5.747 appears in all three pairwise contrasts (long>short d=11.1, long>medium d=6.6, medium>short d=5.6) with a strongly positive long-distance mean (+0.567). It is the single most reliably distance-sensitive neuron.

Layer distribution of top-20 distance neurons: early-to-mid layers dominate (layers 2–7 for long>short; layers 0–5 for long>medium), consistent with structural encoding occurring before the upper-layer attention heads integrate number information.

### Congruency-sensitive neurons (within long condition, top 5 for PP2)

| Layer | Neuron | t | Cohen's d | mean incong | mean cong |
|---|---|---|---|---|---|
| 5 | 1015 | 11.0 | 0.868 | −0.002 | −0.008 |
| 4 | 1597 | 9.6 | 0.759 | −0.159 | −0.166 |
| 4 | 1880 | 9.3 | 0.739 | −0.163 | −0.167 |
| 8 | 459 | 7.9 | 0.626 | −0.167 | −0.168 |
| 7 | 2141 | 7.8 | 0.616 | −0.163 | −0.167 |

Congruency effects are far weaker (d < 1) than distance effects, confirming that subject-verb agreement is not strongly localized to individual MLP neurons — it is distributed across attention routing. PP1-congruency neurons (within long) are even weaker (d < 0.5) and cluster in layers 10–11, suggesting that subject–PP1 number integration occurs late in the network.

### Neuron ablations

Baseline (no ablation): **0.9495**

| Ablated neurons | Overall | Short | Medium | Long |
|---|---|---|---|---|
| none (baseline) | 0.9495 | 0.9812 | 0.9594 | 0.9078 |
| `L5.747` | 0.9521 | 0.9812 | 0.9594 | 0.9156 |
| all 47 distance-sensitive neurons | 0.9516 | 0.9734 | 0.9594 | 0.9219 |

Interpretation from ablation:

- **`L5.747` is selectively harmful in the long condition.** Zeroing it leaves short and medium untouched, but improves long by +0.0078. That is consistent with a neuron that becomes active only when the dependency is structurally difficult and slightly biases the model away from the correct agreement computation.
- **The full distance-sensitive set is not a pure "depth support" circuit.** Ablating all 47 neurons improves long substantially (+0.0141) but hurts short (−0.0078), with medium unchanged. This pattern is more consistent with a mixed structural heuristic: these neurons help when the subject is nearby, but become misleading when intervening phrases create a longer dependency.
- **So the localizer is finding behaviorally real neurons, but not a clean "agreement buffer."** Their collective effect looks closer to a proximity/interference representation than a dedicated mechanism for preserving subject number across distance.

### Conclusions

1. **Structural distance is the dominant MLP signal.** Effect sizes for distance contrasts (d ≈ 10–22) dwarf congruency effects (d < 1), so the strongest single-neuron effects track the structural configuration of the dependency much more than agreement success or failure.
2. **The main activation pattern is a controlled GELU release, not raw length.** All conditions are matched for word count, so these neurons are not responding to simple sequence length. They are responding to how much syntactic material intervenes between subject and verb.
3. **`L5.747` is the most robust single neuron** across all three distance contrasts, and ablating it improves only the long condition, making it a candidate interference neuron rather than a support neuron.
4. **The full distance-sensitive set behaves like a proximity/interference circuit.** Joint ablation helps long but hurts short, which argues against a pure structural-depth representation and suggests a heuristic that is useful for local dependencies but counterproductive for long ones.
5. **Congruency is not strongly localized.** No single neuron shows a large effect for agreement mismatch, consistent with the agreement signal residing primarily in attention head composition and residual-stream superposition rather than individual MLP neurons.