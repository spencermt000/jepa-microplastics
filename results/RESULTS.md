# Experiment Results — jepa-microplastics

> **Validity note**: Results generated before commit `f43e8f0` (2026-05-10) used incorrect
> metadata where `label_polymer` was the dataset folder name, not the polymer type. Those
> results (analyze.py run showing 91.3%) are archived in `_archive/` and should be ignored.
> Everything below uses the corrected metadata (rebuilt 2026-05-10 15:08 on yomp).

---

## 1. Linear Probe — 5-Fold CV (Polymer, 8 classes)

Source: `results/baselines/*.csv`, `results/probe_sweep.csv`
All runs: post metadata rebuild (2026-05-10 15:08+), validated against independent diagnostic.

| Encoder                     | Mean Acc | Notes |
|-----------------------------|----------|-------|
| ImageNet ViT-Tiny (frozen)  | 87.6%    | strong transfer; best linear probe |
| Supervised ViT-Tiny (e2e)   | 87.4%    | end-to-end fine-tune |
| DINOv2 ViT-S/14 (frozen)    | 84.6%    | 22M params vs 5.7M |
| JEPA ep050 (frozen)         | 70.3%    | best JEPA checkpoint |
| FOCAL ep050 (frozen)        | — (TBD)  | not yet probed at ep050 |
| FOCAL ep300 (frozen)        | 66.3%    | from diag_focal.py, 5-fold |
| Random ViT-Tiny (frozen)    | 58.3%    | surprising: beats JEPA ep300 |
| JEPA ep300 (frozen)         | 52.2%    | COLLAPSED (see §4) |
| JEPA ep300 rerun            | 53.0%    | confirmed collapse |
| ANCHOR/SimCLR ep300         | — (TBD)  | not yet probed |

### Fold-by-fold (JEPA ep300 vs ImageNet):
| Fold | JEPA ep300 | ImageNet |
|------|-----------|---------|
| 0    | 50.9%     | 85.7%   |
| 1    | 49.8%     | 86.8%   |
| 2    | 52.1%     | 88.2%   |
| 3    | 55.2%     | 88.2%   |
| 4    | 53.1%     | 89.3%   |

---

## 2. JEPA Collapse Timeline (probe_sweep.csv)

Linear probe accuracy over training — clear collapse after ep050.

| Epoch | Linear Probe | MLP Probe |
|-------|-------------|-----------|
| ep050 | **70.6%**   | **86.2%** |
| ep100 | 66.9%       | 81.9%     |
| ep150 | 66.1%       | 79.5%     |
| ep200 | 60.1%       | 71.8%     |
| ep250 | 54.8%       | 68.1%     |
| ep300 | 52.9%       | 62.3%     |

**Key finding**: JEPA peaked at ep050 and degraded monotonically — representation collapse.
MLP probe degrades more slowly (nonlinear probe can still extract signal), but linear
separability collapses. Need VICReg or stronger collapse prevention.

---

## 3. Label Efficiency (Polymer, 5 seeds per fraction)

Source: `results/baselines/*.csv` (all post-metadata-fix)

| Encoder        | 1%    | 5%    | 10%   | 25%   | 50%   | 100%  |
|----------------|-------|-------|-------|-------|-------|-------|
| ImageNet       | 32.3% | 55.7% | 66.2% | 81.5% | 85.7% | 87.1% |
| DINOv2         | 30.0% | 49.3% | 62.1% | 74.8% | 79.7% | 85.2% |
| Supervised     | 35.2% | 55.4% | 62.7% | 75.9% | 79.5% | 86.0% |
| JEPA ep050     | 30.7% | 42.4% | 49.5% | 59.6% | 64.6% | 67.4% |
| Random         | 33.5% | 45.5% | 48.2% | 52.9% | 55.2% | 58.2% |
| JEPA ep300     | 25.3% | 32.0% | 40.0% | 46.7% | 49.9% | 54.7% |

**Goal not met**: JEPA@25% should match Supervised@100% (~87%). Actual JEPA@25% = 47–60%.
ImageNet transfer dominates at all label fractions.

---

## 4. Representation Collapse Analysis

Evidence of collapse in JEPA ep300:
- Linear probe below random init (52% vs 58%) — random weights have better linear structure
- Monotonic degradation from ep050 to ep300
- MLP probe still gets 62% at ep300 (features not completely dead, just not linearly separable)

**Root cause**: Standard JEPA masking without VICReg. CLAUDE.md predicted this:
> "Representation collapse: Monitor embedding variance per dim. If < 1e-4, add VICReg."

Embedding variance from MLflow (pretrain experiment):
- Early training: ~0.9 (healthy)
- wistful-koi run: 0.1857 (degraded — well below healthy threshold)

**Recommended fix**: Re-pretrain with VICReg regularization loss term.

---

## 5. Cross-Source Generalization (drift experiment)

Source: results from `scripts/cross_source.py` run (jepa-drift-cross-source experiment)

Key finding: peese_pspmma→other transfer is only 8.5% accuracy. Blend class labels
(PS50PMMA50, PS25PMMA75, etc.) from peese_pspmma don't appear in other sources (which
only have PE, PS, PHA, PMMA). Label mismatch causes near-random performance on
cross-source eval for those sources.

---

## 6. Pending / Missing Results

- [ ] ANCHOR/SimCLR ep300 linear probe (not yet evaluated)
- [ ] FOCAL ep050 linear probe
- [ ] Re-analyze FOCAL collapse timeline (probe_sweep for focal checkpoints)
- [ ] VICReg re-pretrain experiment
- [ ] analyze.py re-run with correct metadata (for morphology/composition probes + UMAP/attention)

---

## MLflow Experiment Index

| Experiment | ID | Status | Notes |
|---|---|---|---|
| jepa-microplastics-pretrain | 3 | valid | base JEPA run |
| focal-jepa | 5 | valid | particle-aware masking |
| jepa-drift-cross-source | 6 | valid | cross-source eval |
| jepa-microplastics-analysis | 7 | **INVALID** | pre-metadata-fix, see _archive/ |
| anchor-simclr-v2 | 8 | valid | SimCLR training |
| jepa-microplastics-baselines | 9 | valid | all baseline comparisons |
