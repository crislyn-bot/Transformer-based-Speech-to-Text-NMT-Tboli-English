# Tboli-to-English Speech-to-Text Neural Machine Translation
A transformer-based speech-to-text translation system developed for the low-resource Tboli language. The project uses a cascaded architecture combining fine-tuned Whisper Large V3 for Automatic Speech Recognition (ASR) and mBART-50 for Tboli-to-English Neural Machine Translation (NMT), refined using Direct Preference Optimization (DPO).

## Models

### Automatic Speech Recognition (ASR)
Fine-tuned Whisper Large V3 using PEFT/LoRA for Tboli speech transcription.

[View ASR Model on Hugging Face](https://huggingface.co/crslnjoyz/asr-whisper-large-v3-tboli-lang)

### Neural Machine Translation (MT)
Fine-tuned mBART-50 with DPO refinement for translating Tboli text into English.

[View MT Model on Hugging Face](https://huggingface.co/crslnjoyz/mt-mbart-50-rlhf-dpo-tboli-lang)

## Dataset Availability
The models were developed using a privately maintained, closed-vocabulary Tboli dataset. The dataset is not publicly available.

## Project Scope
This project supports research into low-resource speech recognition, neural machine translation, and indigenous language technologies.
