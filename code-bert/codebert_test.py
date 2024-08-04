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
import re
import json 
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
    
    df = pd.concat([df_pos,df_neg], ignore_index=True)
    print("Final training dataset", df.shape)
    df['clean_message'] = df['clean_message'].astype(str).apply(clean_text)
    df['patch'] = df['patch'].astype(str).apply(clean_text)
    
    return df

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

        
def extract_code_changes(patch):
    # Remove metadata (commit hash, author, date, etc.)
    lines = patch.split('\n')
    code_changes = []
    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            code_changes.append(line)
    return '\n'.join(code_changes)

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

def main(data_path):
    # Download necessary NLTK data
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    # Check GPU availability
    restart_cuda()

    # Load and process the data
    data = pd.read_csv(data_path, encoding='utf-8')
    data = process_dataframe(data)
    print("Final training dataset", data.shape)
#     test_data = df


    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=45)

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/codebert-base", "microsoft/codebert-base")

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
            patch = self.data.iloc[idx]['patch']
            message = self.data.iloc[idx]['clean_message']
            
            inputs = self.tokenizer(patch, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            outputs = self.tokenizer(message, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            
            return {
                "input_ids": inputs.input_ids.squeeze(),
                "attention_mask": inputs.attention_mask.squeeze(),
                "decoder_input_ids": outputs.input_ids.squeeze(),
                "labels": outputs.input_ids.squeeze(),
            }

    # Prepare datasets
#     train_dataset = CommitDataset(train_data, tokenizer)
    test_dataset = CommitDataset(test_data, tokenizer)



    # Custom data collator
    def data_collator(features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        decoder_input_ids = torch.stack([f["decoder_input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }



    # Load the saved model
    loaded_model = EncoderDecoderModel.from_pretrained("./saved_codebert_models/codebert_commit_model_v8")
    loaded_model = loaded_model.to(device)

    def generate_commit_message(patch, min_length=10, max_length=128):
#         code_changes = preprocess_patch(patch)
        inputs = tokenizer(patch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        output = loaded_model.generate(
            **inputs,
            min_length=min_length,
            max_length=32,
            num_beams=10,
            no_repeat_ngram_size=2,
            early_stopping=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
#             do_sample=True,
#             repetition_penalty=1.5,
#             length_penalty=0.8
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
       
    # Print some sample inputs and outputs
        
    print("\nSample inputs and generated commit messages:")
    for i in range(min(5, len(test_data))):
        print(f"Input patch (truncated): {test_data['patch'].iloc[i][:100]}...")
        print(f"Original commit message: {test_data['clean_message'].iloc[i]}")
        generated = generate_commit_message(test_data['patch'].iloc[i])
        print(f"Generated commit message: {generated}")
        print()

    # Use tqdm for progress bar
    print("Generating commit messages")
    test_data['codebert_message'] = [generate_commit_message(patch) for patch in tqdm(test_data['patch'])]

    def calculate_metrics(references, hypotheses):
        # BLEU
        bleu = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])

        # ROUGE
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

        # METEOR
        meteor_scores = []
        for ref, hyp in zip(references, hypotheses):
            try:
                ref_tokens = word_tokenize(ref)
                hyp_tokens = word_tokenize(hyp)
                score = single_meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(score)
            except Exception as e:
                print(f"Error calculating METEOR score: {e}")
                print(f"Reference: {ref}")
                print(f"Hypothesis: {hyp}")
                continue

        meteor = np.mean(meteor_scores) if meteor_scores else 0

        return {
            'BLEU': bleu,
            'ROUGE-L': rouge_scores['rouge-l']['f'],
            'METEOR': meteor
        }

    metrics = calculate_metrics(test_data['clean_message'], test_data['codebert_message'])
    print("Evaluation Metrics:", metrics)

    # Save results
    output_file = './saved_codebert_models/codebert_predictions.csv'
    test_data.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Saving the metrics to a file
    with open('./saved_codebert_models/results.txt', 'a') as file:
        file.write(f"\n# Evaluation Metrics: for: {data_path}\n")
        file.write(json.dumps(metrics, indent=4))
        file.write("\n" + "-"*50 + "\n")  # Adding a separator for clarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CodeBERT for commit message generation")
    parser.add_argument('data_path', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    
    main(args.data_path)

