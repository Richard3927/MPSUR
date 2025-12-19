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

## 从 Hugging Face 获取模型（放到本仓库的三个目录）
本仓库只提供**下游训练/融合**代码，不提供大模型权重。请从 Hugging Face 下载到对应目录（不要把权重提交到 GitHub）。

### 0) 安装与登录
- 安装（用于下载与推理提取 embedding）：

```bash
pip install -U transformers accelerate huggingface_hub safetensors
```

- 登录（如模型为 gated/需授权，必须先在网页同意协议并使用 token 登录）：

```bash
huggingface-cli login
```

> Hugging Face 官网：`https://huggingface.co`

### 1) 下载到 `Chinses_bert/`（BERT-family / 768d）
选择任意可用的中文 BERT（或医疗领域 BERT）checkpoint，然后执行：

```bash
huggingface-cli download <BERT_CHECKPOINT_ID> ^
  --local-dir Chinses_bert ^
  --local-dir-use-symlinks False
```

下载完成后，目录里通常应包含 `config.json`、分词器文件以及权重文件（如 `model.safetensors`/`pytorch_model.bin`）。

### 2) 下载到 `LLama2_Chinese_Med/`（LLaMA 2）
如果使用官方 LLaMA 2（或任何 gated 模型），需要先在 Hugging Face 页面申请/同意协议后再下载：

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf ^
  --local-dir LLama2_Chinese_Med ^
  --local-dir-use-symlinks False
```

如你使用的是中文/医疗微调版 LLaMA 2，请把 `meta-llama/Llama-2-7b-hf` 替换为你的模型 ID。

### 3) 下载到 `Llama3-8B-Chinese-Chat/`（LLaMA 3）

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct ^
  --local-dir Llama3-8B-Chinese-Chat ^
  --local-dir-use-symlinks False
```

### 4) 注意事项
- **不要提交模型权重**：本仓库 `.gitignore` 已对上述目录做了忽略（仅保留 `.gitkeep` 占位）。
- **Windows 命令行**：上面示例用 `^` 进行换行（PowerShell/CMD 通用写法之一）；如你使用 bash，请把 `^` 改为 `\\`。
- **维度匹配**：下游脚本假设你导出的文本 embedding 维度与所选 backbone 对应（例如 7 字段拼接后的总维度）。

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
