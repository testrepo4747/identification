import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Paths and config
# -------------------------------------------------------------------
TRAIN_PATH = "/working/bert/data/android_tf_opencv_patches.csv"
TEST_PATH = "/working/bert/data/20_c_cpp_patches_v3.csv"

# You can keep these commented model options for future experiments.
# model_name = "Qwen/Qwen3-1.7B"  # you can also try "Qwen/Qwen3-1.7B-Base"
model_name = "Qwen/Qwen2.5-3B"
# model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"

OUTPUT_DIR = "./results"
LOG_DIR = "./logs"
MODEL_SAVE_DIR = "./saved_models/3b/qwen_refactored_seqcls/"
MAX_LENGTH = 512
RANDOM_SEED = 42


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
    Choose the training input here.

    IMPORTANT:
    Uncomment exactly ONE return line below.

    1) For training with commit message only:
       uncomment the 'clean_message' line

    2) For training with diff only:
       uncomment the 'patch' line

    3) For training with diff + commit message:
       uncomment the combined line
    """

    # Uncomment this when training with commit message only:
    # return clean_message

    # Uncomment this when training with diffs or diff only:
    return patch

    # Uncomment this when training with diff + commit message:
    # return f"Commit message:\n{clean_message}\n\nPatch:\n{patch}"


# -------------------------------------------------------------------
# Data processing
# -------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean labels, message, patch, and build final text input."""
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
# Metrics
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    print(metrics)
    return metrics


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
    test_df = pd.read_csv(TEST_PATH).reset_index(drop=True)

    # Process datasets
    train_df = process_dataframe(train_df)
    test_df = process_dataframe(test_df)

    # Split train/validation
    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=train_df["manually_label"],
    )

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = [int(label) for label in examples["manually_label"]]
        return tokenized

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_valid_dataset = valid_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )
    tokenized_test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=256,
        lora_alpha=1024,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=5e-5,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # Evaluate
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print("Test Results:", test_results)
    print(f'Accuracy:  {test_results["eval_accuracy"]}')
    print(f'Precision: {test_results["eval_precision"]}')
    print(f'Recall:    {test_results["eval_recall"]}')
    print(f'F1-score:  {test_results["eval_f1"]}')


if __name__ == "__main__":
    main()