# Fake News Detection Using ModernBERT and Bidirectional GRU (ModernBERT-Bi-GRU)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-ModernBERT%20%2B%20Bi--GRU-purple.svg)](#system-architecture)

---

## 📌 Abstract & Research Overview

The proliferation of misinformation across digital journalism and social media demands robust, generalizable Natural Language Processing (NLP) models capable of nuanced semantic analysis without relying on trivial stylistic shortcuts. 

This repository houses the research codebase, model architecture, training protocols, and empirical evaluations for **ModernBERT-Bi-GRU**: a hybrid deep neural architecture for automated fake news detection. The framework couples the contextual representation capabilities of **ModernBERT** (`answerdotai/ModernBERT-base`) with a **Bidirectional Gated Recurrent Unit (Bi-GRU)** and a **Dual Masked Mean-and-Max Pooling Head**.

### Key Contributions
- **Modern Transformer Backbone**: Leverages ModernBERT, integrating Rotary Position Embeddings (RoPE), GeGLU activations, unpadded sequence processing, and native support for up to 8k context lengths.
- **Sequential Recurrent Dynamics**: Contextual embeddings are passed into a bidirectional GRU layer via packed sequence padding to capture temporal rhetorical flow and narrative progression.
- **Dual Masked Pooling Head**: Combines attention-masked mean pooling (aggregate semantic summary) and masked max pooling (salient lexical cues) into a 512-dimensional joint representation.
- **Leakage-Resistant Dataset Engineering**: Removes systemic data leakage endemic to the ISOT benchmark (e.g., Reuters agency datelines, photo credits, URL markers, and duplicate articles across splits) to ensure authentic linguistic learning.
- **Out-of-Distribution (OOD) Domain Resilience**: Validates cross-dataset transfer on the external McIntire dataset (`fake_or_real_news.csv`), exhibiting significantly lower performance degradation ($25.07$ pts F1 drop) compared to conventional baselines ($42$–$45$ pts F1 drop).

---

## 🏛 System Architecture

The overall model architecture follows an encoder-recurrent-pooling hierarchical design:

```
                                  [Raw News Article: Title + Body]
                                                 │
                                                 ▼
                              [Unicode NFKC Clean & Anti-Leakage Filter]
                                                 │
                                                 ▼
                               [ModernBERT Tokenizer (Max-Len = 256)]
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     │                                                       │
               [Input IDs]                                           [Attention Mask]
                     │                                                       │
                     └───────────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    ModernBERT-base (149M Params)      │
                             │  • Layers 1–20: FROZEN                │
                             │  • Layers 21–22: FINE-TUNED           │
                             │  • Final LayerNorm: FINE-TUNED        │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼  Last Hidden States [B, L, 768]
                             ┌───────────────────────────────────────┐
                             │       Pack Padded Sequence (RNN)      │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │         Bidirectional GRU             │
                             │  • Hidden Size: 128 (per direction)   │
                             │  • Forward + Backward Output: 256     │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼  Unpacked Hidden States [B, L, 256]
                                 ┌───────────────┴───────────────┐
                                 │                               │
                                 ▼                               ▼
                     ┌───────────────────────┐       ┌───────────────────────┐
                     │  Masked Mean Pooling  │       │   Masked Max Pooling  │
                     │       [B, 256]        │       │       [B, 256]        │
                     └───────────────────────┘       └───────────────────────┘
                                 │                               │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼  Concatenated Vector [B, 512]
                             ┌───────────────────────────────────────┐
                             │       Dropout (p = 0.3)               │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    Linear Classifier (512 -> 2)       │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                                   [Logits: Real (0) vs. Fake (1)]
```

### Mathematical Formulation

1. **Contextual Token Encoding**:
   $$\mathbf{H} = \text{ModernBERT}(\mathbf{x}) \in \mathbb{R}^{B \times L \times 768}$$
   where the first $20$ layers are frozen to retain general language priors, and the final $2$ transformer blocks along with `final_norm` are unfrozen for domain adaptation.

2. **Bidirectional Recurrence**:
   $$\mathbf{h}_t = [\overrightarrow{\text{GRU}}(\mathbf{H}_t) \,\|\, \overleftarrow{\text{GRU}}(\mathbf{H}_t)] \in \mathbb{R}^{256}$$

3. **Dual Masked Pooling**:
   $$\mathbf{p}_{\text{mean}} = \frac{\sum_{t=1}^L \mathbf{h}_t \odot m_t}{\max\left(1, \sum_{t=1}^L m_t\right)} \in \mathbb{R}^{256}$$
   $$\mathbf{p}_{\text{max}} = \max_{t \in \{1,\dots,L\}} \left( \mathbf{h}_t \odot m_t + (-\infty) \cdot (1 - m_t) \right) \in \mathbb{R}^{256}$$
   $$\mathbf{z} = [\mathbf{p}_{\text{mean}} \,\|\, \mathbf{p}_{\text{max}}] \in \mathbb{R}^{512}$$

4. **Classification & Prediction**:
   $$\hat{\mathbf{y}} = \mathbf{W}_c \cdot \text{Dropout}(\mathbf{z}) + \mathbf{b}_c, \quad \mathbf{W}_c \in \mathbb{R}^{2 \times 512}$$

---

## 🔬 Dataset Preprocessing & Anti-Leakage Protocol

The ISOT dataset is widely recognized in fake news detection literature, but naive evaluations frequently exploit superficial artifacts:
1. **Reuters Dateline Bias**: Ground-truth real articles often begin with `WASHINGTON (Reuters) -` or contain Reuters datelines, giving classifiers a trivial shortcut.
2. **Footer & Credit Artifacts**: Fake news articles frequently embed standalone lines like `Featured image via...` or `Photo by...`.
3. **Cross-Partition Duplicates**: Raw datasets contain duplicate stories that, if split randomly, leak identical texts between training and test partitions.

### Preprocessing Pipeline:
- **Unicode NFKC Normalization**: Standardizes typography, quotes, and accents.
- **Regex-Guided Dateline Neutralization**: Strips `\(\s*Reuters\s*\)|\bReuters\b` without leaving orphaned parentheses.
- **Credit Line Stripping**: Removes photo agency footnotes and trailing attribution lines.
- **Length Thresholding & Deduplication**: Discards fragments with fewer than $8$ whitespace-delimited tokens and executes deduplication prior to partitioning.

### Dataset Partitions:
| Dataset | Total Cleaned Samples | Real Articles (Class 0) | Fake Articles (Class 1) | Partition Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **ISOT Benchmark** | $39,085$ | $21,194$ | $17,891$ | $80\%$ Train ($31,268$), $10\%$ Val ($3,908$), $10\%$ Test ($3,909$) |
| **McIntire External Dataset** | $6,298$ | $3,153$ | $3,145$ | Zero-shot OOD Evaluation & $5$-Fold Cross-Validation |

---

## ⚙️ Hyperparameters & Training Protocols

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Base Model** | `answerdotai/ModernBERT-base` | Modernized BERT architecture with RoPE & GeGLU |
| **Max Sequence Length** | $256$ tokens | Dynamic batch padding applied |
| **Batch Size** | $32$ (Train), $64$ (Validation/Test) | Optimized for memory footprint and gradient stability |
| **Fine-Tuning Scope** | Last $2$ encoder blocks + final norm | $10,721,538$ trainable vs. $138,983,424$ frozen params |
| **Head Architecture** | $1$-layer Bi-GRU ($h=128$) | Bidirectional output dimension $= 256$ |
| **Dropout** | $0.3$ | Regularization preceding classification projection |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | Weight decay $= 0.01$ |
| **Learning Rate (Head & GRU)** | $1 \times 10^{-3}$ | Standard learning rate for newly initialized layers |
| **Learning Rate (Encoder)** | $2 \times 10^{-5}$ | Conservative rate for unfrozen transformer blocks |
| **LR Scheduler** | Linear warmup with decay | Warmup steps $= 10\%$ of total training steps |
| **Loss Function** | Weighted Cross-Entropy | Balanced class weighting: $w_{\text{real}} \approx 0.922, w_{\text{fake}} \approx 1.092$ |
| **Mixed Precision** | PyTorch AMP (`GradScaler`) | FP16 auto-casting enabled |
| **Early Stopping** | Patience $= 2$ epochs | Checkpoint monitored on Validation Macro-F1 |

---

## 📊 Experimental Results

### 1. In-Domain 5-Fold Stratified Cross-Validation on ISOT (Table 1)
Evaluated across all 5 folds using stratified splits on the cleaned ISOT corpus:

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + Naive Bayes | $95.89 \pm 0.29$ | $95.90 \pm 0.29$ | $95.89 \pm 0.29$ | $95.89 \pm 0.29$ | $99.10 \pm 0.09$ |
| TF-IDF + Logistic Regression | $99.27 \pm 0.14$ | $99.27 \pm 0.14$ | $99.27 \pm 0.14$ | $99.27 \pm 0.14$ | $99.94 \pm 0.02$ |
| TF-IDF + Linear SVM | $99.42 \pm 0.11$ | $99.42 \pm 0.11$ | $99.42 \pm 0.11$ | $99.42 \pm 0.12$ | $99.97 \pm 0.01$ |
| **ModernBERT + Bi-GRU (Proposed)** | **$100.00 \pm 0.01$** | **$100.00 \pm 0.01$** | **$100.00 \pm 0.01$** | **$100.00 \pm 0.01$** | **$100.00 \pm 0.00$** |

### 2. ISOT Holdout Test Set Performance (n = 3,909)
Evaluated using the optimal checkpoint on the unseen $10\%$ stratified holdout partition:

```
              precision    recall  f1-score   support
        Real     1.0000    0.9995    0.9998      2120
        Fake     0.9994    1.0000    0.9997      1789

    accuracy                         0.9997      3909
   macro avg     0.9997    0.9998    0.9997      3909
weighted avg     0.9997    0.9997    0.9997      3909

Test Accuracy: 99.97% | ROC-AUC: 1.0000
```

### 3. Out-of-Distribution (OOD) Domain Generalizability (Table 2)
To examine vulnerability to distribution shift, models trained exclusively on ISOT were deployed **zero-shot** on the completely independent **McIntire Dataset** (`fake_or_real_news.csv`, $n=6,298$) without fine-tuning:

| Model | ISOT Acc (%) | ISOT F1 (%) | McIntire Acc (%) | McIntire Precision (%) | McIntire Recall (%) | McIntire F1 (%) | McIntire AUC (%) | F1 Drop (pts) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ModernBERT + Bi-GRU (Proposed)** | **$99.97$** | **$99.97$** | **$75.04$** | **$75.62$** | **$75.04$** | **$74.90$** | **$82.72$** | **$25.07$** |
| TF-IDF + Logistic Regression | $99.44$ | $99.44$ | $60.19$ | $66.07$ | $60.19$ | $56.22$ | $73.30$ | $43.21$ |
| TF-IDF + Linear SVM | $99.39$ | $99.39$ | $58.37$ | $64.01$ | $58.37$ | $53.75$ | $71.48$ | $45.63$ |
| TF-IDF + Naive Bayes | $96.21$ | $96.21$ | $56.54$ | $58.56$ | $56.54$ | $53.87$ | $58.85$ | $42.34$ |

> **Key Observation**: While classical linear models collapse to near-random prediction ($53\%$–$56\%$ F1, dropping over $42$ points), ModernBERT-Bi-GRU preserves structural linguistic comprehension ($74.90\%$ F1, $82.72\%$ AUC), cutting the domain performance drop by nearly half.

### 4. 5-Fold Stratified Cross-Validation on McIntire From Scratch (Table 3)
When trained directly on the McIntire dataset, ModernBERT-Bi-GRU achieves state-of-the-art in-domain performance:

| Metric | ISOT (mean ± std) | McIntire (mean ± std) | Difference (ISOT - McIntire) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $1.0000 \pm 0.0001$ | $0.9713 \pm 0.0031$ | $+0.0287$ |
| **Precision** | $1.0000 \pm 0.0001$ | $0.9714 \pm 0.0031$ | $+0.0286$ |
| **Recall** | $1.0000 \pm 0.0001$ | $0.9713 \pm 0.0031$ | $+0.0287$ |
| **F1 (Weighted)** | $1.0000 \pm 0.0001$ | $0.9713 \pm 0.0031$ | $+0.0287$ |
| **Macro-F1** | $1.0000 \pm 0.0001$ | $0.9713 \pm 0.0031$ | $+0.0287$ |
| **ROC-AUC** | $1.0000 \pm 0.0000$ | $0.9960 \pm 0.0011$ | $+0.0040$ |

---

## 📂 Repository File Structure

```
.
├── .gitignore                          # Excludes large binaries, checkpoints, & raw CSVs
├── main.ipynb                          # Complete end-to-end research notebook (training & evaluation)
├── table1_isot_comparison.csv          # 5-fold CV baseline comparison on ISOT
├── table2_external_comparison.csv      # Out-of-distribution cross-dataset transfer results
├── cv_comparison_isot_vs_mcintire.csv  # 5-fold CV comparisons (ISOT vs. McIntire)
├── cv_isot_retrain.json                # Raw fold-by-fold metrics for ISOT retraining
├── cv_mcintire_scratch.json            # Raw fold-by-fold metrics for McIntire scratch training
└── README.md                           # Comprehensive architectural and empirical report
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/subhampadhi33537-collab/FND-Using-MordenBERT-Bi-GRU.git
cd FND-Using-MordenBERT-Bi-GRU

python -m venv venv
source venv/bin/activate       # On Linux/macOS
# or: venv\Scripts\activate   # On Windows
```

Install the required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate scikit-learn pandas numpy matplotlib tqdm
```

### 2. Dataset Preparation

Download the required datasets:
- **ISOT Dataset**: Place `True.csv` and `Fake.csv` in the root or data directory.
- **McIntire Dataset**: Place `fake_or_real_news.csv` in the root directory for cross-domain experiments.

### 3. Running the Research Pipeline

Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter lab
```

Open `main.ipynb` to execute:
1. Data cleaning, normalization, and deduplication.
2. Model definition with dynamic sequence packing.
3. Training with AdamW, differential learning rates, and AMP.
4. Validation, checkpointing, and confusion matrix generation.
5. In-domain baseline comparisons and OOD transfer evaluation.

---

## 📜 Citation & Attribution

If you utilize this architecture, preprocessing methodology, or benchmark evaluation in your research, please cite:

```bibtex
@article{padhi2026modernbertbigru,
  title   = {Fake News Detection via ModernBERT Representations and Bidirectional Gated Recurrent Units with Dual Masked Pooling},
  author  = {Padhi, Subham and Collaborators},
  journal = {AI Research Project Repository},
  year    = {2026},
  url     = {https://github.com/subhampadhi33537-collab/FND-Using-MordenBERT-Bi-GRU}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
