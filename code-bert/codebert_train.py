import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import EncoderDecoderModel, RobertaTokenizer
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk
from tqdm import tqdm
import argparse
import os
from rouge import Rouge
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np
# from transformers import get_scheduler, AdamW

torch.cuda.empty_cache()
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Function to clean and normalize text
def clean_text(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def process_dataframe(df):
    df = df.dropna(subset=['patch'])
    dic_list = []
    for index, row in df.iterrows():
        res = row['manually_label']
        if res == "positive" or res == '1.0' or res == '1':
            row['manually_label'] = 1
        elif res == "negative" or res == '0.0' or res == '0':
            row['manually_label'] = 0
        dic_list.append(row)
    
    df_split = pd.DataFrame(dic_list)
    df_pos = df_split.loc[df_split['manually_label'] == 1]
    df_neg = df_split.loc[df_split['manually_label'] == 0]
    print("pos:{}, neg:{}".format(df_pos.shape, df_neg.shape))
        
#     df = pd.concat([df_neg], ignore_index=True)
    df = pd.concat([df_pos,df_neg], ignore_index=True)

    df['clean_message'] = df['clean_message'].astype(str).apply(clean_text)
    df['patch'] = df['patch'].astype(str).apply(clean_text)
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def preprocess_patch(patch):
    if pd.isna(patch):
        return ""  # Return an empty string for NaN entries or consider other imputation strategies

    # Split the patch into lines
    lines = patch.split('\n')
    
    # Filter out metadata lines (assuming they start with 'commit', 'Author:', or 'Date:')
    filtered_lines = [line for line in lines if not (line.startswith('commit') or 
                                                     line.startswith('Author:') or 
                                                     line.startswith('Date:'))]
    
    # Normalize whitespace: replace tabs with four spaces, strip trailing and leading spaces
    normalized_lines = [line.replace('\t', '    ').strip() for line in filtered_lines]
    
    # Encode additions and deletions (assuming '+' or '-' at the start of the line indicates this)
    encoded_lines = ['+ ' + line[1:] if line.startswith('+') else '- ' + line[1:] if line.startswith('-') else line for line in normalized_lines]
    
    # Join the lines back into a single string
    preprocessed_patch = '\n'.join(encoded_lines)
    
    return preprocessed_patch

def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, "checkpoint-*")):
        return max(
            [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
    return None


# # Display the first few preprocessed entries to verify changes
# data[['patch', 'preprocessed_patch']].head()


def restart_cuda():
#     import torch
#     torch.cuda.empty_cache()
#     # Set the device to a particular GPU
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     torch.cuda.set_device(device)

#     print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
    torch.cuda.empty_cache()
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_loss = float('inf')

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None or 'eval_loss' not in metrics:
            return control

        current_loss = metrics['eval_loss']
        if self.best_loss - current_loss > self.early_stopping_threshold:
            self.best_loss = current_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                control.should_training_stop = True
        return control     
    
    
    
def main(data_path):
    # Download necessary NLTK data
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    # Check GPU availability
    restart_cuda()

    # Load and process the data
    data = pd.read_csv(data_path, encoding='utf-8')
    data = process_dataframe(data)
    
    # Apply the preprocessing function to the 'patch' column
    data['preprocessed_patch'] = data['patch'].apply(preprocess_patch)

#     data = data.dropna(subset=['clean_message', 'manually_label', 'patch'])
    print("Final training dataset", data.shape)
    


    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/codebert-base", "microsoft/codebert-base")

#     model_name = "microsoft/codebert-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Set decoder start token id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to GPU
    model = model.to(device)

    # Custom Dataset
    class CommitDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=128):
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            patch = self.data.iloc[idx]['preprocessed_patch']
            message = self.data.iloc[idx]['clean_message']
            
            inputs = self.tokenizer(patch, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            outputs = self.tokenizer(message, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            
            return {
                "input_ids": inputs.input_ids.squeeze().to(device),
                "attention_mask": inputs.attention_mask.squeeze().to(device),
                "decoder_input_ids": outputs.input_ids.squeeze().to(device),
                "labels": outputs.input_ids.squeeze().to(device),
            }

    # Prepare datasets
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_dataset = CommitDataset(train_data, tokenizer)
    val_dataset = CommitDataset(val_data, tokenizer)
    test_dataset = CommitDataset(test_data, tokenizer)
    

    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./saved_codebert_models/results",
        num_train_epochs=40,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
#         learning_rate=0.001,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./saved_codebert_models/logs',
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=False,  # Ensure CUDA is used if available
        fp16=True,  # Use mixed precision training
        save_total_limit=3,  # Keep only the last 3 checkpoints
#         resume_from_checkpoint=True,  # Enable resuming from checkpoint
    )

    # Custom data collator
    def data_collator(features):
        input_ids = torch.stack([f["input_ids"] for f in features]).to(device)
        attention_mask = torch.stack([f["attention_mask"] for f in features]).to(device)
        decoder_input_ids = torch.stack([f["decoder_input_ids"] for f in features]).to(device)
        labels = torch.stack([f["labels"] for f in features]).to(device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()

    # Save the model
    os.makedirs("./saved_codebert_models", exist_ok=True)
    trainer.save_model("./saved_codebert_models/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CodeBERT for commit message generation")
    parser.add_argument('data_path', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    
    main(args.data_path)


