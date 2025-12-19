# MPSUR: Predicting Causes of Unplanned Reoperations with an AI-based Multi‑Modal System

本仓库提供 **MPSUR** 的研究代码：面向**非计划再手术原因**的多模态预测（结构化变量 + 临床文本嵌入）。

Research code for **MPSUR** (Multi‑modal Prediction System for Causes of Unplanned Reoperation): a multi‑modal classifier that predicts the **cause category** of unplanned reoperations from **multi‑modal EMR features**.

![Overview of MPSUR](Overview_of_the_MPSUR.png)

---

## Highlights
- Multi‑modal fusion of:
  - **Structured series features** (20 variables)
  - **Clinical text embeddings** (7 text fields concatenated)
- A shared **graph template** (`20×20`) is used for the structured branch.
- Supports multiple text backbones by swapping the exported embedding matrix:
  - CB (BERT-family, 768‑d)
  - LLaMA 2 / LLaMA 3 (hidden states used as embeddings)

---

## Repository structure
- `Muti_Modal.py`: multi‑modal model (structured branch + text branch + fusion head).
- `dataread.py`: dataset utilities / reshaping helpers for different text backbones.
- `data_process_V2.py`: example preprocessing / feature extraction script.
- `train_SKnet_V1_5fold.py`: training & evaluation script (5‑fold CV logic; paths need to be adapted).
- `data/`: placeholder directory (no raw EMR data is included).
- `Chinses_bert/`, `LLama2_Chinese_Med/`, `Llama3-8B-Chinese-Chat/`: placeholders for local checkpoints (weights are not redistributed).

> Note: Some scripts contain local path placeholders and should be edited to match your environment and prepared data files.

---

## Task definition
**Goal:** multi‑class classification of the cause category (e.g., bleeding / infection / other factors).

**Inputs:**
1. **Structured series**: 20 encoded variables.
2. **Clinical text**: 7 text fields embedded by a backbone model and concatenated.

**Graph template:** the structured branch expects a precomputed adjacency matrix (e.g., `adj_template_1.txt`, shape `20×20`) reused across samples.

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
- columns 2..: **20 structured features**

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

## Training / evaluation
1. Prepare:
   - structured series matrix
   - text embedding matrix
   - graph template `adj_template_1.txt`
2. Update file paths in `train_SKnet_V1_5fold.py` to point to your prepared files.
3. Run:

```bash
python train_SKnet_V1_5fold.py
```

The script reports metrics for three branches:
- `all`: fused multi‑modal prediction
- `series`: structured‑only prediction
- `text`: text‑only prediction

---

## License
This repository is released under the **Apache License 2.0**. See `LICENSE`.

---

## Citation
If you use this code, please cite:

```bibtex
@article{xie2025mpsur,
  title   = {Predicting the causes of unplanned reoperations using an AI-based multi-modal system: A multi-center, multi-departmental study based on EMRs},
  author  = {Xie, Luyuan and Zhao, Congpu and Zhao, Pengyu and Li, Shengyang and Tan, Xutong and Shen, Qingni and Wu, Zhonghai and Yan, Guochen and Gong, Xuan and Luan, Tianyu and Wang, Dan and Zhou, Jiong and Ma, Xiaojun},
  journal = {BMJ Digital Health and AI},
  year    = {2025}
}
```

---

## Contact
Please open an issue for questions about reproducibility or usage.
