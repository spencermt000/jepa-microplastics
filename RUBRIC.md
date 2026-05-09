# Evaluation Rubric — I-JEPA Microplastics

## 1. Pretraining Health (checked during training)

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Loss trend | Decreasing over 10-epoch window | Flat for >20 epochs | Increasing |
| Loss at epoch 50 | < 0.25 | 0.25–0.35 | > 0.35 |
| Loss at epoch 100 | < 0.15 | 0.15–0.22 | > 0.22 |
| Loss at epoch 200 | < 0.10 | 0.10–0.15 | > 0.15 |
| Loss at epoch 300 | < 0.07 | 0.07–0.12 | > 0.12 |
| Embedding variance | > 0.1 | 0.01–0.1 | < 0.01 (collapse) |
| LR schedule | Follows cosine curve | Minor deviation | Stuck at 0 or exploding |

## 2. Representation Quality — Linear Probe (run after pretraining)

### Polymer classification (primary task)
| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Top-1 accuracy | > 60% | 40–60% | < 40% |
| Macro F1 | > 0.55 | 0.35–0.55 | < 0.35 |
| Beats random init baseline | Yes (>5% margin) | Ties | No |
| Beats ImageNet supervised | Yes | Within 5% | No |

### Morphology classification (secondary)
| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Top-1 accuracy | > 70% | 50–70% | < 50% |
| Macro F1 | > 0.65 | 0.45–0.65 | < 0.45 |

### Composition (PEESEgroup 42-class)
| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Top-1 accuracy | > 40% | 25–40% | < 25% |
| Macro F1 | > 0.35 | 0.20–0.35 | < 0.20 |

## 3. Label Efficiency

| Label fraction | Green | Yellow | Red |
|----------------|-------|--------|-----|
| 1% | > 35% acc | 20–35% | < 20% |
| 5% | > 45% acc | 30–45% | < 30% |
| 10% | > 55% acc | 40–55% | < 40% |
| 25% | > 65% acc | 50–65% | < 50% |
| Win condition | Matches supervised-100% at ≤25% labels | Needs 50% | Needs >50% |

## 4. Embedding Quality (t-SNE / UMAP)

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| Cluster separation | Clear polymer clusters | Partial | Random scatter |
| Intra-class variance | Tight clusters | Moderate spread | No structure |
| Inter-class distance | Visually distinct | Overlapping edges | Fully overlapping |

## 5. Collapse Detection

Checked every 45 minutes during training:

| Check | Pass | Warning | Fail |
|-------|------|---------|------|
| Embedding variance | > 0.1 | 0.01–0.1 | < 0.01 |
| Loss not NaN | True | — | False |
| Loss not stuck | Changes > 1e-4 per epoch | Changes > 1e-5 | Flat |
| Gradient norm | < 10 after warmup | 10–50 | > 50 or NaN |

## 6. Overall Grade

- **A (Ship-ready):** All primary metrics Green, label efficiency win condition met
- **B (Strong):** Primary metrics Green, label efficiency Yellow
- **C (Acceptable):** Primary metrics Yellow, beats random init
- **D (Needs work):** Any primary metric Red, or collapse detected
- **F (Restart):** Collapse, NaN loss, or worse than random init
