# Import necessary libraries
import os
import sys  # Import sys to get command-line arguments
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset

# Set environment variables for multi-GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use GPUs 2 and 3
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Llama model with LoRA')

# LoRA parameters
parser.add_argument('--lora_r', type=int, default=16, help='LoRA r parameter')
parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha parameter')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter')

# Training arguments
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--num_train_epochs', type=int, default=6, help='Number of training epochs')

args = parser.parse_args()

# Function to clean and process the DataFrame
def clean_text(text):
    return text.encode('ascii', 'ignore').decode('ascii')

# def process_dataframe(df):
# #     df = df.dropna(subset=['patch'])
#     dic_list = []
#     for index, row in df.iterrows():
#         res = row['manually_label']
#         if res in ["positive", '1.0', '1']:
#             row['manually_label'] = 1
#         elif res in ["negative", '0.0', '0']:
#             row['manually_label'] = 0
#         dic_list.append(row)
    
#     df_split = pd.DataFrame(dic_list)
#     df_pos = df_split.loc[df_split['manually_label'] == 1]
#     df_neg = df_split.loc[df_split['manually_label'] == 0]
#     print("pos:{}, neg:{}".format(df_pos.shape, df_neg.shape))
        
#     df = pd.concat([df_pos, df_neg], ignore_index=True)

#     df['clean_message'] = df['clean_message'].astype(str).apply(clean_text)
#     print(df.shape)
    
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

#     # Ensure 'patch' is a string and apply preprocessing
#     df['patch'] = df['patch'].astype(str).apply(preprocess_patch)
#     df = df.dropna(subset=['patch'])

    # Print counts for verification
    df_pos = df[df['manually_label'] == 1]
    df_neg = df[df['manually_label'] == 0]
    print(f"Positive samples: {df_pos.shape[0]}, Negative samples: {df_neg.shape[0]}")
    
    df['clean_message'] = df['clean_message'].astype(str).apply(clean_text)
    print(df.shape)

    return df


# # Define file paths.
# train_path = "/working/bert/data/android_tf_opencv_patches.csv"
# test_path  = "/working/codebert/data/20_c_cpp_v2_processed_patches.csv"

# # Load the CSV files.
# print("Loading CSV files...")
# train_df = pd.read_csv(train_path, encoding='utf-8')
# test_df  = pd.read_csv(test_path, encoding='utf-8').reset_index(drop=True)

# # Process each DataFrame.
# train_df = process_dataframe(train_df)
# test_df = process_dataframe(test_df)

# # Combine the two DataFrames.
# combined_df = pd.concat([train_df, test_df], ignore_index=True)
# print("Combined dataset shape:", combined_df.shape)

# # Check unique class values in "manually_label"
# unique_classes = combined_df['manually_label'].unique()

# # Determine half of the sample size for each class
# half_sample_size = 5666 // 2

# # Randomly sample half from each class
# sampled_class_0 = combined_df[combined_df['manually_label'] == unique_classes[0]].sample(n=half_sample_size, random_state=42)
# sampled_class_1 = combined_df[combined_df['manually_label'] == unique_classes[1]].sample(n=half_sample_size, random_state=42)

# # Combine the sampled datasets
# sampled_df = pd.concat([sampled_class_0, sampled_class_1], ignore_index=True)


# # Randomly sample 5,666 rows from the combined DataFrame.
# # sampled_df = combined_df.sample(n=5666, random_state=42)
# train_df = sampled_df[['manually_label', 'clean_message']]
# print("Sampled dataset shape:", sampled_df.shape)
    
# # Load and process dataset
# df = pd.read_csv('./6600v2_(android_tf_opencv_chatgpt).csv').reset_index(drop=True)
# df = process_dataframe(df)
# data = df
# mask = data['project'].str.contains('Android|Tensorflow', case=False, na=False)
# # mask = data['project'].str.contains('Tensorflow', case=False, na=False)
# df = data[mask]
# print("Total Training set: ",df.shape)
# # android_tf_data = android_tf_data.sample(frac = 1)
# # android_tf_data.shape

test_df = pd.read_csv("/working/codebert/data/20_c_cpp_v2_processed_patches.csv").reset_index(drop=True)
# test_df = process_dataframe(test_df)
train_path = "/working/bert/data/android_tf_opencv_patches.csv"
# Load dataset JS
# train_path = "/working/bert/data/6_js_train_with_patches.csv"
# test_path = "/working/bert/data/20_js_test_with_patches.csv"
# test_path = "/working/bert/data/20_c_cpp_patches_v3.csv"
train_df =  pd.read_csv(train_path, encoding='utf-8')
# test_df = pd.read_csv(test_path, encoding='utf-8')

# df = pd.read_csv("/working/bert/data/20_c_cpp_patches_with_llama_messages.csv", encoding='utf-8')
# # test_df = test_df[['manually_label', 'llama_massages','project']]
# test_df = df[['manually_label', 'llama_massages']]
# test_df =  test_df.rename(columns={"llama_massages": "clean_message"})

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)


# # Load and process dataset
# df = pd.read_csv('./6600v2_(android_tf_opencv_chatgpt).csv').reset_index(drop=True)
# df = pd.read_csv('/working/bert/data/20_c_cpp_patches_v3.csv').reset_index(drop=True) 
# df = process_dataframe(df)
# data = df

# # Identify rows matching the mask condition
# mask = data['project'].str.contains('Android|Tensorflow', case=False, na=False)

# # Create the training set by selecting rows matching the mask
# # df = data[mask]
# print("Total Training set: ", df.shape)

# # Load and process the test dataset
# # test_df = pd.read_csv("/working/codebert/data/20_c_cpp_v2_processed_patches.csv").reset_index(drop=True)
# test_df = pd.read_csv('/working/bert/data/20_c_cpp_patches_v3.csv').reset_index(drop=True) 

# test_df = process_dataframe(test_df)

# Create the test set by combining rows not matching the mask with the original test_df
# test_df = pd.concat([data[~mask], test_df], ignore_index=True)


print("Total testing set: ", test_df.shape)

# Split the dataset into training and validation
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

# # Get the unique projects
# projects = test_df['project'].unique()
# accuracies = []

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["clean_message"],
        padding='longest',
        truncation=True,
        max_length=1024
    )
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": examples["manually_label"]
    }

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Configure LoRA with custom parameters
peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "mlp.fc_in", "mlp.fc_out", "mlp.layer_norm"
    ]
)

# Load the LlamaForSequenceClassification Model
model = LlamaForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.2-1B",
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

# TrainingArguments with custom parameters
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=args.num_train_epochs,
    gradient_checkpointing=True,
    evaluation_strategy='epoch',  # Evaluate after each epoch
    logging_dir="./logs",
    logging_steps=250,
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
    metric_for_best_model="accuracy",
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    # Print the metrics for immediate feedback
    print({
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    })
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

from transformers import EarlyStoppingCallback

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,  # Use validation dataset during training
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
trainer.train()

# # Save the model and tokenizer
# model_path = "./saved_models/llama3.2-3b_v1_msgbased_c_cpp(random_sample_5666)/"
# trainer.save_model(model_path)
# tokenizer.save_pretrained(model_path)


# Load the saved model for testing
# saved_model = LlamaForSequenceClassification.from_pretrained('./saved_models/llama3.2-1b_v5', device_map="auto", trust_remote_code=True)
# saved_model.config.pad_token_id = tokenizer.pad_token_id
# saved_model.resize_token_embeddings(len(tokenizer))


# # Load the saved model for testing
# saved_model = LlamaForSequenceClassification.from_pretrained('./saved_models/llama3.2-1b_v1_msgbased_6_js_train', device_map="auto", trust_remote_code=True)
# saved_model.config.pad_token_id = tokenizer.pad_token_id
# saved_model.resize_token_embeddings(len(tokenizer))

# trainer.model = saved_model

# Evaluate the model on the test set
# test_results = model.evaluate(eval_dataset=tokenized_test_dataset)
# print("Test Results:", test_results)

# Compute predictions on test set
test_predictions = trainer.predict(test_dataset=tokenized_test_dataset)
# test_predictions = trainer.predict(tokenized_test_dataset)
test_preds = test_predictions.predictions.argmax(-1)
test_labels = test_predictions.label_ids

# Compute classification report and confusion matrix
class_report = classification_report(test_labels, test_preds, digits=4)
conf_matrix = confusion_matrix(test_labels, test_preds)

# Print the performance metrics
# print(f'Accuracy: {test_results["eval_accuracy"]}')
# print(f'Precision: {test_results["eval_precision"]}')
# print(f'Recall: {test_results["eval_recall"]}')
# print(f'F1-score: {test_results["eval_f1"]}')
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
      

# ----------------------------
# New code to save predictions
# ----------------------------
# Add the predicted labels to the original test DataFrame under a new column 'llama_label'
# test_df['llama_label'] = test_preds

# # Specify the output file path (you can change the path and file name as needed)
# output_csv_path = "./20_js_test_results_with_llama_labels.csv"

# # Save the DataFrame with the new column to a CSV file
# test_df.to_csv(output_csv_path, index=False)
# print(f"Test results with 'llama_label' have been saved to: {output_csv_path}")

# Save the test results and parameters to a text file
# results_filename = './autollama_test_results_v5.txt'

# # Get the command used to run the script
# command_line = ' '.join(['python'] + sys.argv)

# # Open the file and append the results
# with open(results_filename, 'a') as f:
#     f.write('*' * 70 + '\n')
#     f.write(f'# Results for file: {command_line}\n\n')
#     f.write('Classification Report:\n')
#     f.write(class_report + '\n')
#     f.write('Confusion Matrix:\n')
#     f.write(str(conf_matrix) + '\n\n')
      
# #       Iterate through each project and evaluate
# for project in projects:
#     project_df = test_df[test_df['project'] == project]
    
#     # Skip if no data for the project
#     if project_df.empty:
#         continue

#     # Prepare test dataset
#     test_dataset = Dataset.from_pandas(project_df)
#     tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

#     # Evaluate the model
#     test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
#     accuracies.append((project, test_results['eval_accuracy']))

# # Display results
# for project, accuracy in accuracies:
#     print(f"{project}: {accuracy:.2f}")

# # Calculate the average accuracy
# average_accuracy = sum([acc[1] for acc in accuracies]) / len(accuracies)
# print(f"Average Accuracy: {average_accuracy:.2f}")
      
      
      
      
