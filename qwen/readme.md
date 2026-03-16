# Qwen LoRA Sequence Classification

Minimal guide to train and test the refactored Qwen classification pipeline.

## 1. What this project does

This project fine-tunes a Qwen sequence classification model with LoRA / QLoRA for binary classification using:

- commit message
- patch / diff
- commit message + patch

The input mode is selected inside the script by uncommenting the matching line in `build_text_input()`.

## 2. Required CSV columns

Your train and test CSV files must contain these columns:

- `manually_label`
- `clean_message`
- `patch`

Accepted label values:

- positive, 1, 1.0
- negative, 0, 0.0

## 3. Environment

Install the required Python packages before running:

```bash
pip install torch transformers datasets peft bitsandbytes scikit-learn pandas accelerate
