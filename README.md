# MPSUR: Predicting Causes of Unplanned Reoperations with an AI-based Multi‑Modal System

This repository contains research code for **MPSUR** (Multi‑modal Prediction System for Causes of Unplanned Reoperation), a multi‑modal learning pipeline for categorizing unplanned reoperation causes from **multi‑modal EMR features**.

We fuse **21 structured variables** with embeddings derived from **multiple clinical text fields**, provide example preprocessing to convert tabular sources into model‑ready matrices, and include a training script for cross‑validation experiments.

![Overview of MPSUR](Overview_of_the_MPSUR.png)

---

## Highlights
- Multi‑modal fusion of:
  - **Structured series features** (21 variables)
  - **Clinical text embeddings** (7 text fields concatenated)
- A shared **graph template** (`21×21`) is used for the structured branch.
- Supports multiple text backbones by swapping the exported embedding matrix:
  - CB (BERT-family, 768‑d)
  - LLaMA 2 / LLaMA 3 (hidden states used as embeddings)

---

## Repository structure
- `Muti_Modal.py`: multi‑modal model (structured branch + text branch + fusion head).
- `dataread.py`: dataset utilities / reshaping helpers for different text backbones.
- `data_process.py`: example preprocessing / feature extraction script.
- `train_5fold.py`: training & evaluation script (5‑fold CV logic; paths need to be adapted).
- `data/`: placeholder directory (no raw EMR data is included).
- `Chinses_bert/`, `LLama2_Chinese_Med/`, `Llama3-8B-Chinese-Chat/`: placeholders for local checkpoints (weights are not redistributed).

> Note: Some scripts contain local path placeholders and should be edited to match your environment and prepared data files.

---

## Task definition
**Goal:** multi‑class classification of the cause category (e.g., bleeding / infection / other factors).

**Inputs:**
1. **Structured series**: 21 encoded variables.
2. **Clinical text**: 7 text fields embedded by a backbone model and concatenated.

**Graph template:** the structured branch expects a precomputed adjacency matrix (e.g., `adj_template_1.txt`, shape `21×21`) reused across samples.

---

## Environment
This repo is provided as research code. A typical environment includes:
- Python 3.8+
- PyTorch
- NumPy / pandas
- (Optional, for embedding extraction) Hugging Face Transformers

---

## Data format (expected by the scripts)
Due to patient privacy and institutional constraints, **raw EMR data is not included**. The code expects **preprocessed numeric matrices** (loaded via `np.loadtxt`).

### Structured series matrix
A plain‑text numeric matrix where:
- column 0: `patient_id` (or record index)
- column 1: `label` (`0/1/2` according to your preprocessing)
- columns 2..: **21 structured features**

### Text embedding matrix
A plain‑text numeric matrix where each row is the concatenation of **7 text fields**:
- CB: 7 x 768 = 5376
- LLaMA 2 (typical): 7 x 5120 = 35840
- LLaMA 3 (typical): 7 x 4096 = 28672

During training, the scripts concatenate structured features and text embeddings along the feature dimension.

---

## Text backbones (embeddings)
This repo **does not** redistribute model weights. If you need to reproduce text embeddings, download checkpoints from Hugging Face (or use your own clinical encoder) and export embeddings to a matrix matching the expected dimensions.

Folder names in this repo match the intended local directories:
- `Chinses_bert/` (folder name kept for compatibility with existing scripts)
- `LLama2_Chinese_Med/`
- `Llama3-8B-Chinese-Chat/`

---

## Download models from Hugging Face (into the three local folders)
This repository does **not** redistribute model weights. Download checkpoints from Hugging Face into the corresponding local folders (and do not commit weights to GitHub).

### 0) Install & login
- Install (for downloading and embedding extraction):

```bash
pip install -U transformers accelerate huggingface_hub safetensors
```

- Login (required for gated models; accept the license on the model page first, then login with a token):

```bash
huggingface-cli login
```

> Hugging Face website: `https://huggingface.co`

### 1) Download into `Chinses_bert/` (BERT-family / 768d)
Choose any available Chinese BERT (or a clinical BERT) checkpoint, then run:

```bash
huggingface-cli download <BERT_CHECKPOINT_ID> --local-dir Chinses_bert --local-dir-use-symlinks False
```

After download, the folder should contain `config.json`, tokenizer files, and model weights (e.g., `model.safetensors` / `pytorch_model.bin`).

### 2) Download into `LLama2_Chinese_Med/` (LLaMA 2)
If you use the official LLaMA 2 checkpoint (or any gated model), request/accept access on Hugging Face first:

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir LLama2_Chinese_Med --local-dir-use-symlinks False
```

If you use a Chinese/medical fine‑tuned LLaMA 2, replace `meta-llama/Llama-2-7b-hf` with your model id.

### 3) Download into `Llama3-8B-Chinese-Chat/` (LLaMA 3)

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir Llama3-8B-Chinese-Chat --local-dir-use-symlinks False
```

### 4) Notes
- **Do not commit weights**: `.gitignore` ignores the model folders (only `.gitkeep` placeholders are tracked).
- **Dimension matching**: downstream scripts assume your exported text embedding dimensions match the chosen backbone (e.g., the 7-field concatenated dimension).

---

## Training / evaluation
1. Prepare:
   - structured series matrix
   - text embedding matrix
   - graph template `adj_template_1.txt`
2. Update file paths in `train_5fold.py` to point to your prepared files.
3. Run:

```bash
python train_5fold.py
```

The script reports metrics for three branches:
- `all`: fused multi‑modal prediction
- `series`: structured‑only prediction
- `text`: text‑only prediction

---

## License
This repository is released under the **Apache License 2.0**. See `LICENSE`.

---

## Contact
Please open an issue for questions about reproducibility or usage.
