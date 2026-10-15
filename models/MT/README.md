# mBART-50 — Tboli-to-English Machine Translation

**Notebook:** [mt_mbart_tboli_lang.ipynb](mt_mbart_tboli_lang.ipynb)  
**Pre-DPO model:** [crslnjoyz/mt-pre-RLHF-v2](https://huggingface.co/crslnjoyz/mt-pre-RLHF-v2)  
**DPO-refined model:** [crslnjoyz/mt-mbart-50-rlhf-dpo-tboli-lang](https://huggingface.co/crslnjoyz/mt-mbart-50-rlhf-dpo-tboli-lang)

This notebook documents the **neural machine translation (MT)** component of the Tboli-to-English speech translation project. It fine-tunes **mBART-50** on Tboli–English text pairs and explores further refinement through **Direct Preference Optimization (DPO)** using native-speaker feedback.

## Notebook contents

- Load the parallel Tboli–English text dataset and fine-tune mBART-50.
- Generate sample translations for human rating.
- Prepare chosen/rejected preference pairs from translation evaluations.
- Refine the supervised MT model with DPO through Hugging Face TRL.
- Compare pre-DPO and post-DPO translations using BLEU and ChrF, including paired statistical analysis.

## Key training configuration

| Stage | Setting | Value |
| --- | --- | --- |
| Supervised MT | Base model | `facebook/mbart-large-50-many-to-many-mmt` |
| Supervised MT | Epochs / learning rate | `10` / `3e-5` |
| DPO refinement | Epochs / learning rate | `5` / `2e-5` |
| DPO refinement | Beta | `0.5` |
| DPO refinement | Per-device batch size | `2` |
| DPO refinement | Maximum sequence length | `128` |
| Tokenization | Source → target | `tl_XX` → `en_XX` |

`tl_XX` (Tagalog) was used as a **proxy source language code for Tboli**. The model translates **Tboli text to English text** and does not take audio as input.

## Evaluation

| Corpus-level metric | Pre-DPO | Post-DPO |
| --- | ---: | ---: |
| BLEU | **94.60** | **91.87** |
| ChrF | **96.38** | **92.79** |

These are the thesis's reported closed-vocabulary MT test results, evaluated from **reference Tboli text**, not ASR-generated transcripts. DPO reduced both reference-based scores; the reported experiment should not be described as an improvement in overall translation accuracy or as proof of reliable preference alignment.

## Running the notebook

1. Open the notebook in **Google Colab** with a suitable GPU, and run the installation cells.
2. Provide authorized Tboli–English data (`metadata_clean.csv`) and, for DPO, the separately collected native-speaker feedback file. These are private and **not** included in this repository.
3. Follow the notebook's supervised MT, feedback preparation, DPO refinement, and evaluation sections as needed.
4. Review the `MODEL_PATH`, `OUTPUT_DIR`, `PRE_RLHF_DIR`, and `POST_DPO_DIR` settings. Some cells reference **older model repository IDs or experiment versions**; replace them with the intended checkpoints before rerunning.
5. Check Hugging Face publishing settings (`push_to_hub` / `HF_PUSH`) before executing training or upload cells.

**Scope:** The text pairs come from a predefined vocabulary also represented in training. Results do not establish translation quality for unseen Tboli expressions or unrestricted conversations.
