# Tboli–English Speech-to-Text Translation

A research implementation of **Tboli-to-English speech translation** for a low-resource indigenous language. The system uses a **cascaded, automated pipeline**: a fine-tuned Whisper Large V3 model transcribes Tboli speech, and a fine-tuned mBART-50 model translates the resulting Tboli text into English.

```text
Tboli speech → Whisper Large V3 (ASR) → Tboli text → mBART-50 (MT) → English text
```

The two components are trained separately; this is an end-to-end **application workflow**, not a single jointly trained speech-to-translation model.

## Models

| Component | Method | Hugging Face |
| --- | --- | --- |
| **Automatic Speech Recognition (ASR)** | Whisper Large V3 fine-tuned with PEFT/LoRA | [Tboli ASR model](https://huggingface.co/crslnjoyz/asr-whisper-large-v3-tboli-lang) |
| **Machine Translation (MT)** | mBART-50 fine-tuned for Tboli → English and refined with DPO | [Tboli–English MT model](https://huggingface.co/crslnjoyz/mt-mbart-50-rlhf-dpo-tboli-lang) |

The ASR repository provides LoRA adapter weights, which must be loaded with the original Whisper Large V3 model. Model weights are hosted on Hugging Face rather than in this GitHub repository.

## Repository structure

```text
.
├── README.md
├── .gitignore
└── models/
    ├── ASR/
    │   ├── README.md
    │   └── asr_whisper_tboli_lang.ipynb
    └── MT/
        ├── README.md
        └── mt_mbart_tboli_lang.ipynb
```

- **[ASR notebook and documentation](models/ASR/README.md):** speech preprocessing, PEFT/LoRA fine-tuning, and ASR evaluation.
- **[MT notebook and documentation](models/MT/README.md):** supervised mBART fine-tuning, native-speaker feedback processing, DPO refinement, and evaluation.

## Reported results

| Component / evaluation | Metric | Result |
| --- | --- | ---: |
| ASR — held-out recordings | Character Error Rate (CER) | **15.06%** |
| ASR — held-out recordings | Word Error Rate (WER) | **50.42%** |
| MT — pre-DPO, clean Tboli text | Corpus BLEU / ChrF | **94.60 / 96.38** |
| MT — post-DPO, clean Tboli text | Corpus BLEU / ChrF | **91.87 / 92.79** |

These are results reported in the accompanying thesis, not a direct measurement of the full speech-to-English pipeline. MT scores were obtained from **reference Tboli text**, rather than error-containing ASR output. The post-DPO model scored lower than the pre-DPO model on both reference-based MT metrics.

## Dataset and limitations

The experiments used a **private, closed-vocabulary** Tboli dataset covering **310 predefined word and phrase pairs**. Speech data included **4,650 recordings**, with recording-level training, validation, and test splits. The same text pairs can appear across those splits as different recordings, so reported metrics do **not** establish performance on unseen vocabulary or unrestricted conversations.

**The audio recordings, transcripts, translation pairs, and human-feedback data are not provided in this repository.** The notebooks require separately authorized access to these resources.

## Getting started

1. Open the [ASR](models/ASR/README.md) or [MT](models/MT/README.md) folder and read its instructions.
2. Run the corresponding `.ipynb` notebook in a Python environment with a suitable GPU (the notebooks were developed for Google Colab).
3. Supply your own authorized data and adjust paths and model IDs as needed. The notebooks contain some historical experiment paths and are research notebooks rather than one-click deployment scripts.
4. Authenticate with Hugging Face interactively only if you intend to download restricted resources or publish a checkpoint. Review `push_to_hub` settings before running training.

## Research context

Developed as part of **“End-to-End Transformer-Based Speech-to-Text Neural Machine Translation for the Low-Resource Tboli-English Language Pair.”** The project explores transformer-based ASR and MT for Tboli in a limited, closed-vocabulary research setting.
