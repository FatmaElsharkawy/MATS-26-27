# Supplementary Material: Behavioral vs. Representational Forgetting in Pretrained ViTs

This document accompanies the executive summary and Colab notebook. It (1) defines every quantity reported in the notebook, (2) explains exactly how Figures 1 and 2 are computed from those quantities, and (3) reports the full raw results for all four experiments in one place.

---

## 1. Experimental setup (recap)

Two-task class-incremental learning (CIL), no replay, naive fine-tuning (backbone unfrozen in Task 2):

| Experiment | Backbone | Dataset | Task 1 classes | Task 2 classes |
|---|---|---|---|---|
| 1A | DINOv2-S/14 | CIFAR-10 | 0–4 (5 classes) | 5–9 (5 classes) |
| 1B | Vanilla ViT-S/16 (supervised) | CIFAR-10 | 0–4 (5 classes) | 5–9 (5 classes) |
| 2A | DINOv2-S/14 | CIFAR-100 | 0–49 (50 classes) | 50–99 (50 classes) |
| 2B | Vanilla ViT-S/16 (supervised) | CIFAR-100 | 0–49 (50 classes) | 50–99 (50 classes) |

Every quantity below is computed **only on the old-task test set** (the classes learned in Task 1), unless stated otherwise.

---

## 2. Metric glossary

| Symbol used below | Notebook label | Definition |
|---|---|---|
| $\phi_1$ | — | Backbone weights right after Task 1 (before Task 2 fine-tuning) |
| $\phi_2$ | — | Backbone weights after Task 2 (drifted, since backbone was not frozen) |
| $W_1$ | — | Task-1 classifier head (trained on $\phi_1$, old classes only) |
| $W_2$ | — | Full classifier head after Task 2 (old + new class rows, trained jointly) |
| $\text{Acc}_1$ | "Task 1 accuracy before Task 2" | Accuracy of $(\phi_1, W_1)$ on old-class test images |
| $\text{Acc}_{old}$ | "Old-class accuracy after Task 2" | Accuracy of $(\phi_2, W_2)$ on old-class test images, decided by argmax over **all** classes (old + new). This is the real, reported behavioral number. |
| Forgetting | "Forgetting" | $\text{Acc}_1 - \text{Acc}_{old}$ |
| Probe$_{before}$ / Probe$_{after}$, per layer | "before" / "after" in the layer table | Accuracy of a **freshly fit** linear probe on frozen features at that layer, using $\phi_1$ (before) or $\phi_2$ (after). A fresh probe is refit each time, so this measures pure geometric/linear separability of old-class information, independent of any specific classifier head. |
| Representation forgetting (layer $\ell$) | "representation_forgetting" | $\text{Probe}_{before,\ell} - \text{Probe}_{after,\ell}$ |
| L1 | "Post-Task-2 Layer 12 + original Task-1 head" | Accuracy of $(\phi_2, W_1)$: drifted backbone, but scored with the **original, frozen** Task-1 head, decision made only among old classes. Isolates backbone drift without letting the classifier adapt. |
| L2 | "Post-task W2 old accuracy" | Accuracy using the **old-class rows of $W_2$** (the actual post-Task-2 classifier), but with the decision restricted to old classes only — new-class logits are excluded from the argmax. Isolates classifier-weight drift for old classes, with competition from new classes removed. |
| — | "Original W1 accuracy" | Restates L1 for reference (same number, same quantity) |
| Cosine similarity (per class) | "cosine_similarity" | $\cos(w_{1,c}, w_{2,c})$ between the Task-1 weight vector and the corresponding old-class row of $W_2$, for class $c$. Measures whether the classifier direction rotated. |
| Relative weight change (per class) | "relative_weight_change" | $\lVert w_{2,c} - w_{1,c} \rVert / \lVert w_{1,c} \rVert$. Measures magnitude of change, independent of direction. |
| Mean absolute logit difference | "Mean absolute logit difference" | Mean over old-class test images of $\lvert z_{W_1}(\phi_1(x)) - z_{W_2}(\phi_2(x)) \rvert$ for the true old-class logit. Measures how much the raw logit value for the correct old class shifted, even where the predicted label doesn't change. *(Please double-check this definition against the notebook cell that produced it — this is the most likely definition given where it appears in the output, but confirm before citing it in the writeup.)* |
| Prediction agreement | "Prediction agreement" | Fraction of old-class test images for which $(\phi_2, W_1)$ and $(\phi_1, W_1)$ produce the **same predicted label**. |
| Fraction where new class beats all old | "Fraction where new class beats all old classes" | Over old-class test images, the fraction for which **at least one new-class logit** (from $W_2$) exceeds every old-class logit (from $W_2$). Direct measurement of recency-bias severity: how often the joint argmax would pick a new class purely due to logit-scale competition. |

---

## 3. How Figure 1 is computed

**Figure 1** ("Behavioral vs. representation forgetting") plots two bars per experiment:

- **Behavioral forgetting** = $\text{Acc}_1 - \text{Acc}_{old}$ (the "Forgetting" row)
- **Representation forgetting** = the layer-12 (final layer) value of $\text{Probe}_{before} - \text{Probe}_{after}$

Both are expressed in percentage points. The four experiments (1A, 1B, 2A, 2B) form the four groups on the x-axis.

Layer-12 was chosen because it is the final backbone layer feeding the classifier — the most relevant layer for the behavioral comparison. The full per-layer table (layers 3, 6, 9, 12) is reported in Sections 5–6 below and shows that representation forgetting is concentrated at the last layer; layers 3, 6, 9 show negligible (sometimes negative, i.e. improved) forgetting.

---

## 4. How Figure 2 is computed

**Figure 2** ("Decomposition of behavioral forgetting") explains the *gap* between representation forgetting and behavioral forgetting from Figure 1 by inserting two intermediate checkpoints between $\text{Acc}_1$ and $\text{Acc}_{old}$:

$$\text{Acc}_1 \;\rightarrow\; L1 \;\rightarrow\; L2 \;\rightarrow\; \text{Acc}_{old}$$

Each arrow removes one confound relative to the fully-adapted, fully-competing final classifier:

| Step | Comparison | What it isolates |
|---|---|---|
| $\text{Acc}_1 \rightarrow L1$ | $(\phi_1,W_1) \rightarrow (\phi_2,W_1)$ | **Backbone drift**: cost of the backbone changing, with the classifier held fixed |
| $L1 \rightarrow L2$ | $(\phi_2,W_1) \rightarrow (\phi_2,W_2^{old})$, no competition | **Classifier-weight drift**: additional cost of the old-class weight rows themselves changing during Task-2 training |
| $L2 \rightarrow \text{Acc}_{old}$ | old-only decision $\rightarrow$ full joint decision | **Recency-bias / logit competition**: additional cost of letting new-class logits compete in the argmax |

Because these are three consecutive differences along a single chain from $\text{Acc}_1$ to $\text{Acc}_{old}$, they sum **exactly** to total forgetting:

$$(\text{Acc}_1 - L1) + (L1 - L2) + (L2 - \text{Acc}_{old}) = \text{Acc}_1 - \text{Acc}_{old} = \text{Forgetting}$$

No approximation or fitting is involved — each term is a direct accuracy difference read off the notebook output.

### Decomposition table (percentage points)

| Experiment | Acc₁ | L1 | L2 | Acc_old | Backbone drift | Classifier-weight drift | Recency bias | Total forgetting |
|---|---|---|---|---|---|---|---|---|
| 1A: CIFAR-10, DINOv2 | 97.86 | 94.50 | 93.18 | 0.52 | 3.36 | 1.32 | 92.66 | 97.34 |
| 1B: CIFAR-10, Vanilla ViT | 99.18 | 97.94 | 97.86 | 25.56 | 1.24 | 0.08 | 72.30 | 73.62 |
| 2A: CIFAR-100, DINOv2 | 90.24 | 76.34 | 71.74 | 7.28 | 13.90 | 4.60 | 64.46 | 82.96 |
| 2B: CIFAR-100, Vanilla ViT | 93.16 | 87.76 | 86.90 | 38.24 | 5.40 | 0.86 | 48.66 | 54.92 |

---

## 5. Full results — Experiment 1: CIFAR-10 (5 → 5 classes)

### 1A. DINOv2-S/14

| Quantity | Value |
|---|---|
| Task 1 accuracy before Task 2 | 97.86% |
| Old-class accuracy after Task 2 | 0.52% |
| New-class accuracy after Task 2 | 99.22% |
| All-class accuracy | 49.87% |
| Forgetting | 97.34 pp |

**Layer-wise representation forgetting:**

| Layer | Probe before | Probe after | Representation forgetting |
|---|---|---|---|
| 3 | 47.60% | 47.64% | −0.04 pp |
| 6 | 63.96% | 65.38% | −1.42 pp |
| 9 | 82.70% | 78.96% | 3.74 pp |
| 12 | 98.18% | 96.50% | 1.68 pp |

**Decomposition inputs:**

| Quantity | Value |
|---|---|
| L1: drifted backbone + original Task-1 head | 94.50% |
| L2: old-class rows of $W_2$, no competition | 93.18% |
| Mean absolute logit difference | 1.383 |
| Fraction where new class beats all old | 99.80% |
| Prediction agreement |  94.26% |

**Per-class weight comparison ($W_1$ vs. old-class rows of $W_2$):**

| Class | Cosine similarity | Relative weight change |
|---|---|---|
| 0 | 0.9786 | 0.2083 |
| 1 | 0.9843 | 0.1780 |
| 2 | 0.9521 | 0.3075 |
| 3 | 0.9759 | 0.2203 |
| 4 | 0.9826 | 0.1879 |
| **Mean** | **0.9747** | **0.2204** |

### 1B. Vanilla ViT-S/16

| Quantity | Value |
|---|---|
| Task 1 accuracy before Task 2 | 99.18% |
| Old-class accuracy after Task 2 | 25.56% |
| New-class accuracy after Task 2 | 99.74% |
| All-class accuracy | 62.65% |
| Forgetting | 73.62 pp |

**Layer-wise representation forgetting:**

| Layer | Probe before | Probe after | Representation forgetting |
|---|---|---|---|
| 3 | 59.98% | 60.28% | −0.30 pp |
| 6 | 75.80% | 75.38% | 0.42 pp |
| 9 | 82.82% | 82.26% | 0.56 pp |
| 12 | 99.20% | 98.36% | 0.84 pp |

**Decomposition inputs:**

| Quantity | Value |
|---|---|
| L1: drifted backbone + original Task-1 head | 97.94% |
| L2: old-class rows of $W_2$, no competition | 97.86% |
| Mean absolute logit difference | 0.532 |
| Fraction where new class beats all old | 74.50% |
| Prediction agreement | 99.28% |

**Per-class weight comparison:**

| Class | Cosine similarity | Relative weight change |
|---|---|---|
| 0 | 0.9933 | 0.1155 |
| 1 | 0.9951 | 0.0989 |
| 2 | 0.9926 | 0.1219 |
| 3 | 0.9959 | 0.0905 |
| 4 | 0.9960 | 0.0897 |
| **Mean** | **0.9946** | **0.1033** |

---

## 6. Full results — Experiment 2: CIFAR-100 (50 → 50 classes)

### 2A. DINOv2-S/14

| Quantity | Value |
|---|---|
| Task 1 accuracy before Task 2 | 90.24% |
| Old-class accuracy after Task 2 | 7.28% |
| New-class accuracy after Task 2 | 88.78% |
| All-class accuracy | 48.03% |
| Forgetting | 82.96 pp |

**Layer-wise representation forgetting:**

| Layer | Probe before | Probe after | Representation forgetting |
|---|---|---|---|
| 3 | 14.02% | 14.30% | −0.28 pp |
| 6 | 35.10% | 32.86% | 2.24 pp |
| 9 | 57.58% | 57.66% | −0.08 pp |
| 12 | 91.48% | 87.68% | 3.80 pp |

**Decomposition inputs:**

| Quantity | Value |
|---|---|
| L1: drifted backbone + original Task-1 head | 76.34% |
| L2: old-class rows of $W_2$, no competition | 71.74% |
| Mean absolute logit difference | 0.365 |
| Fraction where new class beats all old | 89.30% |
| Prediction agreement | 83.82% |

**Per-class weight comparison (50 classes, summary statistics — full per-class table in the Colab notebook):**

| Statistic | Cosine similarity | Relative weight change |
|---|---|---|
| Mean | 0.9363 | 0.3453 |
| Min | 0.8693 (class 31) | 0.1771 (class 30) |
| Max | 0.9842 (class 30) | 0.5214 (class 24) |

### 2B. Vanilla ViT-S/16

| Quantity | Value |
|---|---|
| Task 1 accuracy before Task 2 | 93.16% |
| Old-class accuracy after Task 2 | 38.24% |
| New-class accuracy after Task 2 | 92.24% |
| All-class accuracy | 65.24% |
| Forgetting | 54.92 pp |

**Layer-wise representation forgetting:**

| Layer | Probe before | Probe after | Representation forgetting |
|---|---|---|---|
| 3 | 23.02% | 20.46% | 2.56 pp |
| 6 | 44.16% | 45.96% | −1.80 pp |
| 9 | 48.00% | 41.28% | 6.72 pp |
| 12 | 93.18% | 91.20% | 1.98 pp |

**Decomposition inputs:**

| Quantity | Value |
|---|---|
| L1: drifted backbone + original Task-1 head | 87.76% |
| L2: old-class rows of $W_2$, no competition | 86.90% |
| Mean absolute logit difference | 0.724 |
| Fraction where new class beats all old | 54.46% |
| Prediction agreement | 96.52% |

**Per-class weight comparison (50 classes, summary statistics):**

| Statistic | Cosine similarity | Relative weight change |
|---|---|---|
| Mean | 0.9689 | 0.2458 |
| Min | 0.9494 (class 11) | 0.1745 (class 1) |
| Max | 0.9847 (class 49) | 0.3152 (class 11) |

---

## 7. Notes and caveats

- All four experiments are **single runs** (no seeds/error bars). Magnitude comparisons between backbones — especially in Section 2 (CIFAR-100) — should be treated as suggestive rather than statistically confirmed until replicated.
- Layer 9 for the Vanilla ViT / CIFAR-100 run shows more representation forgetting (6.72 pp) than layer 12 (1.98 pp), which breaks the otherwise monotonic "forgetting concentrated near the output" pattern seen elsewhere. This is flagged as a possible single-run artifact rather than a robust finding.
- The exact definition of "Mean absolute logit difference" given in Section 2 is inferred from context; confirm against the notebook implementation before using it in the main text.

