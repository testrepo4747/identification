
import torch
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import torch
torch.cuda.empty_cache()
import sys
sys.path.append('./')
print(sys.path)
import re
import string
from bert_model import load_bert_tokenizer,find_max_len, tokenzie_all_sentences
from bert_model import load_bert_sequence_classification,set_optimizer_and_learning_rate, train
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# Hyperparameters and constants
BERT = 'bert-base-uncased'
MAX_LEN = 512
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCHS = 4

# Set the device to a particular GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.cuda.empty_cache()
print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')


# Model training function
def train_model(train_dataloader, validation_dataloader, device):
    model = train(train_dataloader,validation_dataloader,BATCH_SIZE,device)
    return model

# Main execution
def main(folder_path):
    


    
    # Load tokenizer and convert commit messaes into vectors using BERT Tokenizer
    

    tokenizer = BertTokenizer.from_pretrained(BERT)
    
    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
#             torch.cuda.empty_cache()
            # Load and preprocess data
            df = pd.read_csv(os.path.join(folder_path, csv_file))
        
            df = df.sample(frac = 1)
            
            num_samples = len(df)
            num_train = int(0.90 * num_samples)
            train_df = df[:num_train]
            valid_df = df[num_train:]

#             train_df = df[:4656]
#             valid_df = df[4100:]
            train_df = train_df[['manually_label','clean_message']]
            valid_df = valid_df[['manually_label','clean_message']]
            train_df.dropna(inplace=True) 
            valid_df.dropna(inplace=True) 

            print("train:{}, valid:{}".format(train_df.shape,valid_df.shape))
            
            train_sentences = train_df.clean_message.values
            train_labels = train_df.manually_label.values
            # Get the lists of sentences and their labels.
            valid_sentences = valid_df.clean_message.values
            valid_labels = valid_df.manually_label.values
            
            
            # ... (other preprocessing steps)
            
            # Tokenize data and create datasets and dataloaders
            # Tokenize all of the sentences and map the tokens to thier word IDs.
            train_input_ids = []
            valid_input_ids = []
            train_attention_masks = []
            valid_attention_masks = []

            find_max_len(train_sentences,tokenizer)
            find_max_len(valid_sentences,tokenizer)


            train_input_ids ,train_attention_masks  = tokenzie_all_sentences(train_sentences,tokenizer)
            valid_input_ids ,valid_attention_masks = tokenzie_all_sentences(valid_sentences,tokenizer)
            
            # Convert the lists into tensors.
            train_input_ids = torch.cat(train_input_ids, dim=0)
            train_attention_masks = torch.cat(train_attention_masks, dim=0)
            train_labels = torch.tensor(train_labels.astype(np.long))
            # train_labels = torch.tensor(train_labels)

            # Convert the lists into tensors.
            valid_input_ids = torch.cat(valid_input_ids, dim=0)
            valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
            valid_labels = torch.tensor(valid_labels.astype(np.long))
            
            train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
            val_dataset = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
            
            

            # The DataLoader needs to know our batch size for training, so we specify it 
            # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
            # size of 16 or 32.
            batch_size = BATCH_SIZE

            # Create the DataLoaders for our training and validation sets.
            # We'll take training samples in random order. 
            train_dataloader = DataLoader(
                        train_dataset,  # The training samples.
                        sampler = RandomSampler(train_dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )

            # For validation the order doesn't matter, so we'll just read them sequentially.
            validation_dataloader = DataLoader(
                        val_dataset, # The validation samples.
                        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                        batch_size = batch_size # Evaluate with this batch size.
                    )
            # ...
            
            # Train the model
            model = train_model(train_dataloader, validation_dataloader, device)
            
            # Save the model
            dataset_name = 'bert_base_uncased__' + csv_file.replace('.csv', '') +'_model'
            saved_model_path = os.path.join("saved_models", dataset_name)
            model.save_pretrained(saved_model_path)
            tokenizer.save_pretrained(saved_model_path)
            
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python bert_train.py <path_to_folder_with_csv_files>")
        sys.exit(1)
    main(sys.argv[1])
