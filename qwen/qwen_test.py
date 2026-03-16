import os
import argparse

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
)

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Cleaning helpers
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Keep ASCII characters only."""
    if pd.isna(text):
        return ""
    return str(text).encode("ascii", "ignore").decode("ascii")


def preprocess_patch(patch: str) -> str:
    """
    Clean git patch text.

    - Removes commit metadata lines
    - Normalizes tabs to spaces
    - Adds a space after leading '+' or '-' for readability
    """
    if pd.isna(patch):
        return ""

    patch = str(patch)
    lines = patch.splitlines()

    filtered_lines = []
    for line in lines:
        if line.startswith(("commit", "Author:", "Date:")):
            continue

        line = line.replace("\t", "    ").rstrip()

        # Keep file header markers like +++ / --- unchanged
        if line.startswith("+") and not line.startswith("+++"):
            line = "+ " + line[1:]
        elif line.startswith("-") and not line.startswith("---"):
            line = "- " + line[1:]

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def normalize_label(value):
    """Convert labels to 0 or 1. Return None for invalid values."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"positive", "1", "1.0"}:
        return 1
    if value in {"negative", "0", "0.0"}:
        return 0

    return None


# -------------------------------------------------------------------
# Input builder
# -------------------------------------------------------------------
def build_text_input(clean_message: str, patch: str) -> str:
    """
    Choose the inference input here.

    IMPORTANT:
    This must match the exact input mode used during training.

    Uncomment exactly ONE return line below.

    1) For inference with commit message only:
       uncomment the 'clean_message' line

    2) For inference with diffs or diff only:
       uncomment the 'patch' line

    3) For inference with diff + commit message:
       uncomment the combined line
    """

    # Uncomment this when the model was trained with commit message only:
    # return clean_message

    # Uncomment this when the model was trained with diffs or diff only:
    return patch

    # Uncomment this when the model was trained with diff + commit message:
    # return f"Commit message:\n{clean_message}\n\nPatch:\n{patch}"


# -------------------------------------------------------------------
# Data processing
# -------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean labels, text fields, and build final inference input."""
    df = df.copy()

    required_columns = {"manually_label", "clean_message", "patch"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize labels and keep only valid binary samples
    df["manually_label"] = df["manually_label"].apply(normalize_label)
    df = df[df["manually_label"].isin([0, 1])].copy()

    # Clean text fields
    df["clean_message"] = df["clean_message"].apply(clean_text)
    df["patch"] = df["patch"].apply(preprocess_patch)

    # Build final model input
    df["text"] = [
        build_text_input(clean_message, patch)
        for clean_message, patch in zip(df["clean_message"], df["patch"])
    ]

    # Drop empty samples after selecting input mode
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    df["manually_label"] = df["manually_label"].astype(int)

    pos_count = int((df["manually_label"] == 1).sum())
    neg_count = int((df["manually_label"] == 0).sum())

    print(f"Processed shape: {df.shape}")
    print(f"Positive samples: {pos_count}")
    print(f"Negative samples: {neg_count}")

    return df


# -------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------
def build_tokenize_function(tokenizer, max_length: int = 512):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [int(label) for label in examples["manually_label"]]
        return tokenized

    return tokenize_function


# -------------------------------------------------------------------
# Evaluation utilities
# -------------------------------------------------------------------
def evaluate_and_save_results(predictions, labels, results_file: str, run_name: str):
    preds = predictions.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    class_report = classification_report(labels, preds, digits=4, zero_division=0)
    conf_matrix = confusion_matrix(labels, preds)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Classification Report:")
    print(class_report)
    print("Confusion Matrix:")
    print(conf_matrix)

    with open(results_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-score: {f1}\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n\n")


# -------------------------------------------------------------------
# Arg parsing
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Qwen sequence classification model with LoRA adapter"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_adapter_path = os.path.join(
        script_dir,
        "saved_models",
        "3b",
        "qwen_refactored_seqcls",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=default_adapter_path,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        # default="Qwen/Qwen3-1.7B",  # you can also try "Qwen/Qwen3-1.7B-Base"
        default="Qwen/Qwen2.5-3B",
        # default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        help="Base Qwen model name",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/working/bert/data/20_c_cpp_patches_v3.csv",
        help="Path to the CSV test file",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="./qwen_test_results.txt",
        help="File to append evaluation results to",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="qwen_refactored_inference",
        help="Label written into the results file",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()

    print("Adapter path:", args.adapter_path)
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_path}")

    print("Loading test data from:", args.test_path)
    test_df = pd.read_csv(args.test_path).reset_index(drop=True)
    test_df = process_dataframe(test_df)

    # Keep only the final columns needed for inference
    test_df = test_df[["text", "manually_label"]]
    print("Total testing set:", test_df.shape)

    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Load tokenizer from adapter folder
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # BitsAndBytes config to match training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model:", args.base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        print(
            f"Resizing token embeddings from {model_vocab_size} "
            f"to {tokenizer_vocab_size} to match tokenizer"
        )
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    print("Loading LoRA adapter from:", args.adapter_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenize_function = build_tokenize_function(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    tokenized_test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    test_predictions = trainer.predict(test_dataset=tokenized_test_dataset)

    preds = test_predictions.predictions
    labels = test_predictions.label_ids

    evaluate_and_save_results(
        predictions=preds,
        labels=labels,
        results_file=args.results_file,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()