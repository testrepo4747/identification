# LLaMA for Commit Classification

This project provides two scripts using the LLaMA model with LoRA to classify software commits as **security-related** or **not**, based on either:
- Commit messages
- Code diffs (patches)

---

## 🔧 Prerequisites

Install the required libraries:

```bash
pip install transformers datasets peft accelerate scikit-learn pandas
```

You will also need:
- A CUDA-enabled GPU setup
- Python 3.8+
- Hugging Face token (if using LLaMA from their hub)

---

## 📁 Input Format

Prepare a CSV file with:
- For **commit message model** (`auto_llama.py`):
  - `clean_message`: The commit message.
  - `manually_label`: `1` (positive) or `0` (negative).
- For **code diff model** (`llama_codebase.py`):
  - `patch`: The code diff (git patch).
  - `manually_label`: `1` (positive) or `0` (negative).

---

## 🚀 Training Commit Message Model

Script: `auto_llama.py`

### Run:

```bash
python3 auto_llama.py \
  --lora_r 256 \
  --lora_alpha 1024 \
  --lora_dropout 0.05 \
  --learning_rate 5e-5 \
  --num_train_epochs 6 \
  --data_path '/your/path/to/your_data.csv'
```

- 90% of the data will be used for training, 10% for evaluation.
- Model will be saved to:
  ```
  ./saved_llama_models/
  ```

---

## 🚀 Training Code Diff Model

Script: `llama_codebase.py`

### Steps:

1. Edit the script to set the `DATA_PATH` variable:
   ```python
   DATA_PATH = '/your/path/to/your_data.csv'
   ```
2. Then run:
   ```bash
   python3 llama_codebase.py
   ```

- The script uses `patch` as input and maps `manually_label` to 0/1.
- 90% training, 10% test.
- Trained model saved in:
  ```
  ./saved_llama_models/
  ```

---

## 📊 Evaluation

Both scripts output:
- Accuracy, precision, recall, F1 score
- Confusion matrix
- Classification report

Predicted results may be saved to:
```
./results/
```

---

## ⚠️ Notes

- LoRA and quantized loading are already configured (`bnb_config` with 4-bit precision).
- Environment is set up for multi-GPU use by default.
- You can tune LLaMA settings or switch to another HuggingFace-supported model if needed.

---


