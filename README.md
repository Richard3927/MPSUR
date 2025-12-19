# MPSUR: Predicting Causes of Unplanned Reoperations with an AI-based Multi‑Modal System

This repository provides a research implementation of **MPSUR** (Multi‑modal Prediction System for Causes of Unplanned Reoperation), an AI framework that predicts the **cause category** of unplanned reoperations from **multi‑modal electronic medical records (EMRs)**.

Paper (proof in this repo): `bmjdhai-2025-000248_Proof_hi (1).pdf`

![Overview of MPSUR](Overview_of_the_MPSUR.png)

---

## Abstract (from the paper)
Unplanned reoperations are widely recognized as key indicators of surgical quality and patient safety. This study develops an artificial intelligence (AI)-based multimodal system integrating structured and unstructured EMR data across institutions. We develop **MPSUR**, trained on a retrospective cohort of **2,921 cases** collected from **15 departments across eight hospitals (2015–2024)**. The system integrates structured series data (eg, age and sex) and clinical text (eg, diagnoses and procedures). In a clinical reader study, MPSUR outperformed surgeons across departments. MPSUR achieved **70.0%** and **63.0%** accuracy on internal and external datasets, respectively, outperforming classical baselines by up to 10% and remaining robust when only one modality is available.

---

## What is in this repo
- `Muti_Modal.py`: the multi‑modal model (GCN for structured series + TFRNN for text + linear fusion head).
- `dataread.py`: dataset loaders and modality-specific input reshaping for multiple text backbones.
- `data_process_V2.py`: an example preprocessing/feature‑extraction script (Excel → structured coding + text embeddings).
- `train_SKnet_V1_5fold.py`: training/evaluation script with 5‑fold CV logic (paths need to be adapted).
- `abstract.png`, `Overview_of_the_MPSUR.png`: figures used for documentation.

> Note: The current code contains hard‑coded absolute paths (eg, `/disk1/xly/...`). For open‑sourcing and reproducibility, you should replace them with relative paths or a configuration file.

---

## Problem setup
**Task:** multi‑class classification of *cause category* of unplanned reoperation:
- bleeding
- infection
- other factors

**Modalities:**
1. **Structured series**: 20 encoded variables (demographics + perioperative status + comorbidities + operation indicators).
2. **Clinical text**: 7 text fields embedded by a large language model (LLM) and concatenated into a single vector.

**Graph construction (series modality):** the paper describes constructing a shared adjacency matrix by connecting two nodes when the absolute Pearson correlation coefficient exceeds a threshold (e.g., 0.5). The code expects a precomputed `adj_template_1.txt` (shape `20×20`) and reuses it across samples.

---

## Text backbone options (CB / LLaMA2 / LLaMA3)
In the paper, we evaluate several mainstream language models for text feature extraction, including **BERT (CB)**, **LLaMA 2**, and **LLaMA 3**; BERT achieves the best mean accuracy for this task.

In this repo, the downstream classifier is fixed, while the **text embedding dimension depends on the backbone**:

| Backbone | Folder (recommended) | Hidden size | 7-field concat dim |
|---|---|---:|---:|
| CB (Chinese BERT / BERT-family) | `Chinses_bert/` | 768 | 5376 |
| LLaMA 2 | `LLama2_Chinese_Med/` | 5120 (typical) | 35840 |
| LLaMA 3 (8B) | `Llama3-8B-Chinese-Chat/` | 4096 (typical) | 28672 |

> The folder names above match the repository layout. You may use any compatible Hugging Face checkpoint; just ensure the hidden size matches your exported embedding vectors.

---

## Deploying / downloading models from Hugging Face
This project uses Hugging Face **Transformers** checkpoints as *local feature extractors*. We **do not** redistribute model weights in this repository; download them from Hugging Face.

### 0) Prerequisites
- Install dependencies:
  - `pip install -U transformers accelerate huggingface_hub safetensors`
- (Recommended) Install Git LFS if you plan to download via `git clone`:
  - https://git-lfs.com
- Login (required for gated models such as official Meta LLaMA checkpoints):
  - `huggingface-cli login`

You can download models either with `huggingface-cli download` or `git lfs clone`.

### 1) CB (Chinese BERT / BERT-family, 768‑d)
Choose a BERT‑family checkpoint available on Hugging Face (e.g., a Chinese BERT or a clinical-domain BERT). Then:

```bash
huggingface-cli download <BERT_CHECKPOINT_ID> \
  --local-dir Chinses_bert \
  --local-dir-use-symlinks False
```

The folder should contain files such as `config.json`, tokenizer files, and model weights (e.g., `model.safetensors` or `pytorch_model.bin`).

### 2) LLaMA 2 (causal LM; use hidden states as embeddings)
If you use the official Meta LLaMA 2 checkpoints, you must request/accept access on Hugging Face first. Then:

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir LLama2_Chinese_Med \
  --local-dir-use-symlinks False
```

> If you use a community Chinese/medical fine‑tuned checkpoint instead, replace the model id with your chosen `<LLAMA2_CHECKPOINT_ID>`.

### 3) LLaMA 3 (8B; causal LM; use hidden states as embeddings)
Similarly, request/accept access for official Meta LLaMA 3 checkpoints if needed:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir Llama3-8B-Chinese-Chat \
  --local-dir-use-symlinks False
```

### 4) Minimal embedding extraction (Transformers)
Below is a minimal, backbone‑agnostic example that mean‑pools the last hidden states:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_dir = "Chinses_bert"  # or LLama2_Chinese_Med / Llama3-8B-Chinese-Chat
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
model.eval()

text = "diagnosis / procedure description ..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
with torch.no_grad():
    out = model(**inputs).last_hidden_state  # [B, T, H]
emb = out.mean(dim=1).squeeze(0).float().cpu().numpy()  # [H]
print(emb.shape)
```

**Practical notes**
- LLaMA models are large; for limited GPU memory, consider 4‑bit quantization (`bitsandbytes`) or serving with `vllm`. If you only need BERT‑level embeddings, CB is much lighter and typically faster.
- For gated models (official Meta LLaMA), do **not** commit weights to GitHub; provide download instructions only.

---

## Data and file formats
Due to patient privacy and institutional constraints, **raw EMR data is not included**. The code expects **preprocessed numeric matrices**.

### Structured series files
The training script assumes a plain‑text numeric matrix (loaded via `np.loadtxt`) where:
- column 0: `patient_id` (or record index)
- column 1: `label` (`0/1/2` corresponding to `other/bleeding/infection` in the current preprocessing)
- columns 2..: **20 structured features**

### Text embedding files
Text embeddings are exported as a separate plain‑text numeric matrix (`np.loadtxt`), one row per patient, typically built from **7 text fields** concatenated:
- CB: `7 × 768 = 5376`
- LLaMA 2 (typical): `7 × 5120 = 35840`
- LLaMA 3 (typical): `7 × 4096 = 28672`

During training, the scripts concatenate structured features and text embeddings along feature dimension.

---

## Training / evaluation (script-level)
1. Edit paths in `train_SKnet_V1_5fold.py` to point to your prepared data files and adjacency matrix.
2. Run:
   - `python train_SKnet_V1_5fold.py`

The script reports metrics for three inference branches:
- `all`: fused multi‑modal prediction
- `series`: structured‑only prediction
- `text`: text‑only prediction

---

## Open‑sourcing checklist (recommended for a CS-style release)
Before publishing to GitHub:
1. **Remove all sensitive data** (raw EMRs, identifiers, hospital‑internal paths, credentials).
2. **Do not upload model weights** (especially LLaMA). Provide Hugging Face download instructions.
3. Add a `LICENSE` (choose according to your institution’s policy; common choices: MIT/Apache-2.0).
4. Add `.gitignore` rules for `data/`, `*.ckpt`, `*.pth`, `*.bin`, `*.safetensors`, `wandb/`, etc.
5. Provide a reproducible environment (`requirements.txt` or `environment.yml`) and document GPU/CPU assumptions.
6. Add a `CITATION.cff` / BibTeX entry and a short “Data Availability / Ethics” statement consistent with the paper.

---

## Citation
If you use this code, please cite:

```bibtex
@article{xie2025mpsur,
  title   = {Predicting the causes of unplanned reoperations using an AI-based multi-modal system: A multi-center, multi-departmental study based on EMRs},
  author  = {Xie, Luyuan and Zhao, Congpu and Zhao, Pengyu and Li, Shengyang and Tan, Xutong and Shen, Qingni and Wu, Zhonghai and Yan, Guochen and Gong, Xuan and Luan, Tianyu and Wang, Dan and Zhou, Jiong and Ma, Xiaojun},
  journal = {BMJ Digital Health and AI},
  year    = {2025},
  note    = {Manuscript ID: bmjdhai-2025-000248},
}
```

---

## Contact
For questions about the release, reproducibility, or data use agreements, please open an issue or contact the corresponding authors listed in the paper.
