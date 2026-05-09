
Project Summary
Self-supervised pretraining with I-JEPA (ViT-Tiny) on fluorescence microscopy images of microplastics. Goal: learn representations that outperform supervised baselines for polymer classification in the low-label regime. First application of JEPA to microplastic imagery. Eventual target: Deheyn Lab at Scripps (UCSD) / AxonJay pipeline.
Architecture Decisions (Locked)

Encoder: I-JEPA with ViT-Tiny (5.7M params, 12 layers, 192-dim, 3 heads, patch_size=16)
Masking: Standard random block masking (I-JEPA default). Mask multiple target blocks per image, predict their representations from spatially-distributed context.
Target encoder: EMA of online encoder. Momentum: 0.996 → 1.0 cosine schedule.
Predictor: Narrow Transformer (2-4 layers, ≤128-dim) conditioned on positional tokens for target block locations.
Loss: SmoothL1 in latent space. Add VICReg regularization if collapse is observed.
Multi-modal: NOT implemented now. Architecture uses clean interfaces so a spectral encoder can be plugged in later without refactoring.

Key Design Constraint: Modular Interfaces
All components communicate through a standardized Embedding dataclass:
python@dataclass
class Embedding:
    tokens: Tensor       # (B, N, D) — sequence of token embeddings
    mask: Tensor         # (B, N) — bool, True = this token is present/valid
    positions: Tensor    # (B, N, 2) or (B, N) — spatial or sequential positions
    metadata: dict       # optional: modality, source, etc.

Encoder takes raw input (image tensor, or future: spectrum tensor) → returns Embedding
Predictor takes context Embedding + target positions → returns predicted Embedding
Loss takes predicted Embedding + target Embedding → returns scalar

The predictor and loss never see raw images. They only see embeddings. Adding a new modality = writing a new encoder, nothing else changes.
Hardware

Training: yomp (Ryzen 5 2600, GTX 1660 6GB VRAM, Debian 13)
Batch size: 16-32 (monitor VRAM, use gradient accumulation if needed)
Image size: 224×224 (14×14 = 196 patches with patch_size=16)
Mixed precision: Use AMP (torch.cuda.amp) to fit in 6GB
Docker: Run via offload2yomp stack (Airflow + MLflow + Docker)

Data
Phase 0 (public datasets — what we start with)
Primary: PEESEgroup/Microplastic-Project on GitHub

8,400 images across 42 classes (concentration × composition)
Compositions: PS, PS75/PMMA25, PS50/PMMA50, PS25/PMMA75, PMMA, No-MP
Concentrations: 0, 20, 200, 400, 800 mg/L
Images are brightfield optical microscopy on LC-aqueous interface
0.28mm real-world width per image
Also has 846-image PS vs PE set with Grad-CAM analysis

Secondary (combine for larger pretraining corpus):

Kaggle imtkaggleteam/microplastic-dataset-for-computer-vision
Kaggle sivajyothis/microplastic-dataset
GitHub ymzhu19eee/dataset_microplastics (holographic)

Data Pipeline
raw_images/
├── peese/           # PEESEgroup dataset
├── kaggle_cv/       # Kaggle CV dataset
├── kaggle_mp/       # Kaggle MP dataset
└── holographic/     # Holographic dataset

processed/
├── pretrain/        # All images, labels stripped, for JEPA pretraining
├── eval/
│   ├── morphology/  # Labeled by shape (fiber/fragment/bead/film/foam)
│   ├── polymer/     # Labeled by polymer type (PE/PS/PMMA/etc.)
│   └── composition/ # PEESEgroup 42-class labels
└── metadata.csv     # Image path, source dataset, available labels, split assignment

Resize all to 224×224 (or center-crop if aspect ratio varies)
Normalize: compute per-dataset mean/std, or use ImageNet stats if warm-starting
Pretrain split: use ALL images (labels stripped)
Eval split: stratified k-fold (5-fold) within each labeled subset
For label-efficiency experiments: subsample labeled set at 1%, 5%, 10%, 25%, 50%, 100%

Project Structure
jepa-microplastics/
├── CLAUDE.md
├── configs/
│   ├── pretrain.yaml
│   ├── probe.yaml
│   └── data.yaml
├── src/
│   ├── __init__.py
│   ├── embedding.py       # Embedding dataclass
│   ├── encoder.py         # ViT-Tiny encoder → Embedding
│   ├── predictor.py       # Narrow Transformer predictor
│   ├── masking.py         # Block masking strategy
│   ├── jepa.py            # JEPA training module
│   ├── probe.py           # Linear probe / MLP probe heads
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── download.py
│   └── utils/
│       ├── __init__.py
│       ├── ema.py
│       ├── vicreg.py
│       └── metrics.py
├── scripts/
│   ├── pretrain.py
│   ├── probe.py
│   ├── label_efficiency.py
│   └── visualize.py
├── notebooks/
├── checkpoints/
├── results/
└── Dockerfile
Training Protocol
Phase 1: JEPA Pretraining
yamlencoder:
  arch: vit_tiny
  patch_size: 16
  img_size: 224
  embed_dim: 192
  depth: 12
  num_heads: 3
  # Optional: init from DINOv2/I-JEPA ImageNet weights

predictor:
  depth: 4
  embed_dim: 128
  num_heads: 4

masking:
  num_targets: 4
  target_scale: [0.15, 0.2]
  target_aspect: [0.75, 1.5]
  context_scale: [0.85, 1.0]

ema:
  momentum_start: 0.996
  momentum_end: 1.0
  schedule: cosine

training:
  epochs: 300
  batch_size: 16
  grad_accum_steps: 2      # effective batch = 32
  optimizer: adamw
  lr: 1.5e-4
  weight_decay: 0.05
  warmup_epochs: 15
  lr_schedule: cosine
  amp: true
Phase 2: Linear Probe Evaluation

Freeze encoder entirely
Train a single linear layer: 192-dim → num_classes
Use [CLS] token embedding (or mean-pool all patch tokens)
Train for 100 epochs with SGD, lr=0.1, cosine decay
Report top-1 accuracy, macro F1, per-class recall

Phase 3: Label Efficiency

For each label fraction f ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.0}: subsample labeled training set (stratified), train linear probe, evaluate on full test set
Repeat 5 times with different random subsamples, report mean ± std

Phase 4: Baselines

Random init — ViT-Tiny from scratch, linear probe
ImageNet supervised — ViT-Tiny pretrained on ImageNet, linear probe
DINOv2 — ViT-Tiny pretrained with DINOv2, linear probe
SimCLR — ViT-Tiny pretrained with SimCLR on microplastic data, linear probe
Supervised fine-tune — ViT-Tiny trained end-to-end supervised

Logging & Tracking

MLflow on yomp: loss curves, probe accuracy, label efficiency, embedding viz. Tag by experiment type, dataset, masking strategy.
Checkpoints: every 50 epochs + best by probe accuracy
Visualizations: t-SNE/UMAP colored by class at epoch 0, 100, 200, 300
Attention maps: verify model attends to particles, not background

Success Criteria
ExperimentWin ConditionPretrain convergesLoss decreases smoothly, no collapse (embedding variance > 1e-4)Morphology probeAccuracy > supervised-from-scratch baselinePolymer probeAccuracy above chance (>20% for 5-class)Label efficiencyMatches supervised-100% with ≤25% labelsEmbedding qualityVisible clustering by polymer type in t-SNE/UMAP
Common Pitfalls

Representation collapse: Monitor embedding variance per dim. If < 1e-4, add VICReg.
Background dominance: Check attention maps. If diffuse over background, upgrade to particle-aware masking.
Overfitting: dropout=0.1, weight_decay=0.05, monitor train/val gap.
VRAM OOM: Reduce batch to 8, increase grad_accum to 4.
Slow convergence: Don't judge at epoch 50. Probe at 100, 200, 300.

Commands Reference
bashpython scripts/pretrain.py --config configs/pretrain.yaml
python scripts/probe.py --config configs/probe.yaml --checkpoint checkpoints/jepa_ep300.pt
python scripts/label_efficiency.py --checkpoint checkpoints/jepa_ep300.pt --fractions 0.01,0.05,0.10,0.25,0.50,1.0
python scripts/probe.py --config configs/probe.yaml --encoder dinov2
python scripts/visualize.py --checkpoint checkpoints/jepa_ep300.pt --method tsne
Non-Goals (for now)

Multi-modal cross-modal JEPA (wait for Deheyn Lab data)
Particle detection / instance segmentation
Real-time inference or deployment
Saliency-guided masking (only if random masking fails)
Video JEPA on degradation time-lapse