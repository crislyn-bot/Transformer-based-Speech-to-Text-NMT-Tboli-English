# Whisper Large V3 — Tboli Automatic Speech Recognition

**Notebook:** [asr_whisper_tboli_lang.ipynb](asr_whisper_tboli_lang.ipynb)  
**Published model:** [crslnjoyz/asr-whisper-large-v3-tboli-lang](https://huggingface.co/crslnjoyz/asr-whisper-large-v3-tboli-lang)

This notebook documents the Whisper Large V3 ASR experiment for transcribing **spoken Tboli words and short phrases into Tboli text**. The model was adapted with **PEFT / Low-Rank Adaptation (LoRA)** for a predefined, closed vocabulary.

## Notebook contents

- Load and preprocess Tboli recordings and aligned transcripts (16 kHz audio).
- Combine clean and augmented training recordings; keep validation and test recordings unaugmented.
- Adapt Whisper Large V3 using LoRA, with most encoder layers frozen.
- Evaluate transcriptions using Word Error Rate (WER) and Character Error Rate (CER).
- Save and reload the trained adapter with its Whisper base model.


## Key training configuration

| Setting | Value |
| --- | --- |
| Base model | `openai/whisper-large-v3` |
| Fine-tuning | PEFT / LoRA (`q_proj`, `v_proj`) |
| LoRA rank / alpha / dropout | `16` / `32` / `0.05` |
| Encoder | Frozen except for the last two layers before LoRA adaptation |
| Tokenizer language | Tagalog (`tl`) as a proxy for Tboli |
| Training epochs | `12` |
| Learning rate | `5e-6` |
| Batch size / gradient accumulation | `8` / `4` |
| Mixed precision | FP16 |

The tokenizer's Tagalog setting is a **proxy configuration**; the intended transcription output is **Tboli**, not Tagalog.

## Evaluation

| Evaluation | CER | WER |
| --- | ---: | ---: |
| Last logged training-stage evaluation (step 2,500) | 15.93% | 57.50% |
| Held-out test recordings | **15.06%** | **50.42%** |

Test recordings were drawn from the predefined speech script, so these figures should not be interpreted as open-vocabulary ASR performance. The paper's held-out test results describe its evaluated checkpoint; confirm checkpoint identity when reproducing them from published weights.

## Running the notebook

1. Open the notebook in **Google Colab** with a suitable GPU and run the setup cells.
2. Provide authorized dataset files at the paths expected by the notebook, such as `/content/tboli_dataset/metadata_clean.csv`, `/content/tboli_dataset/metadata_aug_1.csv`, and the corresponding private audio folders.
3. Run the dataset, training, and evaluation cells in order. Outputs and execution history were removed from this public copy.
4. If you do **not** wish to publish a new checkpoint, change `push_to_hub=True` to `False` and skip `trainer.push_to_hub()` before running.

**Data availability:** The underlying Tboli recordings and transcriptions are private and are not included here. The Hugging Face model repository contains **LoRA adapter weights**, not a complete standalone Whisper model.

