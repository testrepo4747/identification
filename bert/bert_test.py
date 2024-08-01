
import pandas as pd
import numpy as np
import sys
sys.path.append('./')
print(sys.path)
import tensorflow as tf
import torch
torch.cuda.empty_cache()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizer
import os
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split



BERT= 'bert-base-uncased'
MAX_LEN= 512
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCH=3

# Set the device to a particular GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.cuda.empty_cache()
print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')


def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def load_csv_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

def process_dataframe(df):
    
    dic_list = []
    for index,row in df.iterrows():
        res = row['manually_label'] 

        if res == "positive":
            row['manually_label'] = 1
        elif res == "negative":
            row['manually_label'] = 0

        if res == '1.0':
            row['manually_label'] = 1
        elif res == '0.0':
            row['manually_label'] = 0

        if res == '1':
            row['manually_label'] = 1
        elif res == '0':
            row['manually_label'] = 0

        dic_list.append(row)
    

    
    df_split = pd.DataFrame(dic_list)
#     df_split = a

    df_pos = df_split.loc[df_split['manually_label'] == 1]
    df_neg = df_split.loc[df_split['manually_label'] == 0]
    df_pos1 = df_split.loc[df_split['manually_label'] == '1']
    df_neg1 = df_split.loc[df_split['manually_label'] == '0']

    print("pos:{}, neg:{}".format(df_pos.shape,df_neg.shape))
    print("pos:{}, neg:{}".format(df_pos1.shape,df_neg1.shape))
    
    df= pd.concat([df_pos,df_pos1, df_neg,df_neg1], ignore_index = True)
    
    
    return df

def predict(model, tokenizer, df):
    
    sentences = df.clean_message.values
    labels = df['manually_label'].values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            str(sent),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LEN,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.astype(np.long))

    # Set the batch size.  
    batch_size = BATCH_SIZE

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
      # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        model = model.to(device)
        b_input_mask = b_input_mask.to(device)

      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction

        with torch.no_grad():
            # Forward pass, calculate logit predictions

            outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)


        logits = outputs[0]

      # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

      # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
    # Convert logits to class predictions
    flat_predictions = [item for sublist in predictions for item in np.argmax(sublist, axis=1)]
    # Flatten the true labels list
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    return flat_predictions, flat_true_labels
    
#     return predictions,true_labels

def calculate_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print('accuracy:{}, precision:{}, recall:{}, f1:{}'.format(accuracy,precision,recall,f1))
          
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



def modified_save_all_metrics(metrics, true_labels, predictions, filename, output_file):
    # Calculating the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    with open(output_file, 'a') as f:  # Using 'a' mode to append to the file
        # Writing the filename as a comment for distinction
        f.write(f"\n# Results for file: {filename}\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        
        # Writing the confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n" + "-"*50 + "\n")  # Adding a separator for clarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the pretrained BERT model")
    parser.add_argument("folder_path", help="Path to the folder containing CSV files")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    csv_files = load_csv_files(args.folder_path)
    result_file = "/results.txt"
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = process_dataframe(df)
        
        torch.cuda.empty_cache()
        model, tokenizer = load_model(args.model_path)
        
        predictions,true_labels = predict(model, tokenizer, df)
        metrics = calculate_metrics(predictions, true_labels)
        modified_save_all_metrics(metrics, predictions, true_labels, csv_file,result_file)
#         predictions,true_labels = predict(model, tokenizer, df)
#         metrics = calculate_metrics(predictions, true_labels)
# #         metrics = calculate_metrics(predictions, df['manually_label'].values)
#         save_metrics(metrics, predictions, true_labels, csv_file + '_metrics.txt')

if __name__ == "__main__":
    main()
