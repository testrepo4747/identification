import os
import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset
from transformers import BitsAndBytesConfig
import json

# Set environment variables for multi-GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use GPUs 0,1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Function to preprocess patch
# def preprocess_patch(patch):
#     if pd.isna(patch):
#         return ""  # Return an empty string for NaN entries
#     lines = patch.split('\n')
#     filtered_lines = [line for line in lines if not (line.startswith('commit') or 
#                                                      line.startswith('Author:') or 
#                                                      line.startswith('Date:'))]
#     normalized_lines = [line.replace('\t', '    ').strip() for line in filtered_lines]
#     encoded_lines = ['+ ' + line[1:] if line.startswith('+') else '- ' + line[1:] if line.startswith('-') else line for line in normalized_lines]
#     return '\n'.join(encoded_lines)

def preprocess_patch(patch):
    if pd.isna(patch):
        return ""  # Return an empty string for NaN entries or consider other imputation strategies
    
#     # Split the patch into lines
    lines = patch.split('\n')
    
    # Find the starting point: start from the line that contains "diff" and onwards
#     diff_start_index = next((i for i, line in enumerate(lines) if line.startswith('diff')), None)
#        Try to find the starting point: keep content from the line containing "diff" onwards
#     diff_start_index = next((i for i, line in enumerate(lines) if line.startswith('diff')), None)
    
# #     # If "diff" is found, slice the lines from that point onwards
#     if diff_start_index is not None:
#         lines = lines[diff_start_index:]
    

#     lines = patch.split('\n')
    filtered_lines = [line for line in lines if not (line.startswith('commit') or 
                                                     line.startswith('Author:') or 
                                                     line.startswith('Date:'))]
    normalized_lines = [line.replace('\t', '    ').strip() for line in filtered_lines]
    encoded_lines = ['+ ' + line[1:] if line.startswith('+') else '- ' + line[1:] if line.startswith('-') else line for line in normalized_lines]
    return '\n'.join(encoded_lines)



# # Function to process the dataframe
# def process_dataframe(df):
#     df = df.dropna(subset=['patch'])
#     df['manually_label'] = df['manually_label'].map(lambda x: 1 if x in ["positive", '1.0', '1'] else 0 if x in ["negative", '0.0', '0'] else x)
#     df = df[df['manually_label'].isin([0, 1])]
#     df['patch'] = df['patch'].apply(preprocess_patch)
#     return df

def process_dataframe(df):
    # Drop rows where 'patch' is NaN
#     df = df.dropna(subset=['patch'])
    
    
    # Convert "manually_label" to numeric values (1 or 0)
    df['manually_label'] = df['manually_label'].map(lambda x: 1 if str(x).lower() in ["positive", "1.0", "1"] else 0 if str(x).lower() in ["negative", "0.0", "0"] else None)

    # Drop rows where 'manually_label' is None (i.e., invalid values)
    df = df.dropna(subset=['manually_label'])

    # Convert the labels to integers explicitly
    df['manually_label'] = df['manually_label'].astype(int)

    # Ensure 'patch' is a string and apply preprocessing
    df['patch'] = df['patch'].astype(str).apply(preprocess_patch)
    df = df.dropna(subset=['patch'])

    # Print counts for verification
    df_pos = df[df['manually_label'] == 1]
    df_neg = df[df['manually_label'] == 0]
    print(f"Positive samples: {df_pos.shape[0]}, Negative samples: {df_neg.shape[0]}")

    return df

# Metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# # Compute metrics
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels, preds, average='binary'
#     )
#     acc = accuracy_score(labels, preds)
    
#     # Print the metrics for immediate feedback
#     print({
#         "accuracy": acc,
#         "f1": f1,
#         "precision": precision,
#         "recall": recall
#     })
#     return {
#         "accuracy": acc,
#         "f1": f1,
#         "precision": precision,
#         "recall": recall
#     }

# Load dataset C/C++
train_path = "/working/bert/data/android_tf_opencv_patches.csv"
test_path = "/working/bert/data/20_c_cpp_patches_v3.csv"


# Load dataset JS
# train_path = "/working/bert/data/6_js_train_with_patches.csv"
# test_path = "/working/bert/data/20_js_test_with_patches.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)

print("Total Training set: ", train_df.shape)
print("Total Testing set: ", test_df.shape)

# Split training data into train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Convert DataFrames to Hugging Face datasets
train_dataset = HFDataset.from_pandas(train_df)
valid_dataset = HFDataset.from_pandas(val_df)
test_dataset = HFDataset.from_pandas(test_df)

# Load the Llama tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Tokenize function
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["patch"], padding='longest', truncation=True, max_length=1024)
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": examples["manually_label"]
    }

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Bits and Bytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Configure LoRA for efficient training
peft_config = LoraConfig(
    r=512,
    lora_alpha=1024,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "mlp.fc_in", "mlp.fc_out", "mlp.layer_norm"]
)

# Load the LlamaForSequenceClassification Model
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False  # Disable cache to support gradient checkpointing
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    gradient_checkpointing=True,
    evaluation_strategy='epoch',  # Evaluate after each epoch
    logging_dir="./logs",
    logging_steps=250,
    save_strategy="epoch",
    learning_rate=5e-5,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Train the model
trainer.train()

# # Save the model and tokenizer
model_path = "./saved_models/llama3.2-3b_v1_c_cpp_diff_msg_1024token/"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)


# Load the saved model for testing
# saved_model = LlamaForSequenceClassification.from_pretrained('./saved_models/llama3.2-3b_v2_codebase_codepatch/', device_map="auto", trust_remote_code=True)
# saved_model.config.pad_token_id = tokenizer.pad_token_id
# saved_model.resize_token_embeddings(len(tokenizer))

# trainer.model = saved_model

# Load the saved model for testing
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
# print("Test Results:", test_results)

# # Print the performance metrics
# print(f'Accuracy: {test_results["eval_accuracy"]}')
# print(f'Precision: {test_results["eval_precision"]}')
# print(f'Recall: {test_results["eval_recall"]}')
# print(f'F1-score: {test_results["eval_f1"]}')

test_predictions = trainer.predict(tokenized_test_dataset)
test_preds = test_predictions.predictions.argmax(-1)
test_labels = test_predictions.label_ids

# Compute classification report and confusion matrix
class_report = classification_report(test_labels, test_preds, digits=4)
conf_matrix = confusion_matrix(test_labels, test_preds)


print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
