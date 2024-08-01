import argparse



import os

import torch
import io
import torch.nn.functional as F
import random
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



##Set random values
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)


# If there's a GPU available...

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


# No need to set the device manually using torch.cuda.set_device(device)

print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')



#--------------------------------
#  Transformer parameters
#--------------------------------
max_seq_length = 128
batch_size = 64

#--------------------------------
#  GAN-BERT specific parameters
#--------------------------------
# number of hidden layers in the generator, 
# each of the size of the output space
num_hidden_layers_g = 1; 
# number of hidden layers in the discriminator, 
# each of the size of the input space
num_hidden_layers_d = 1; 
# size of the generator's input noisy vectors
noise_size = 100
# dropout to be applied to discriminator's input vectors
out_dropout_rate = 0.2

# Replicate labeled data to balance poorly represented datasets, 
# e.g., less than 1% of labeled material
apply_balance = True

#--------------------------------
#  Optimization parameters
#--------------------------------
learning_rate_discriminator = 3e-5
learning_rate_generator = 3e-5
epsilon = 1e-8
num_train_epochs = 6
multi_gpu = True
# Scheduler
apply_scheduler = False
warmup_proportion = 0.1
# Print
print_each_n_step = 10




#--------------------------------
#  Adopted Tranformer model
#--------------------------------
# Since this version is compatible with Huggingface transformers, you can uncomment
# (or add) transformer models compatible with GAN

# model_name = "bert-base-cased"
# model_name = "/working/LLM/llama/llama-2-7b/"
# model_name = "huggyllama/llama-7b"
model_name = "bert-base-uncased"
# model_name = "roberta-base"
#model_name = "albert-base-v2"
#model_name = "xlm-roberta-base"
#model_name = "amazon/bort"




label_list = ["UNK_UNK",1,0]

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


transformer = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# If there's a GPU available...

torch.cuda.empty_cache()

transformer.resize_token_embeddings(len(tokenizer))

if multi_gpu:
    transformer = torch.nn.DataParallel(transformer)


# transformer.cuda()

#------------------------------
#   The Generator as in 
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep

#------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
#------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs


# Refactored functions

import pandas as pd


def find_datasets(root_path):
    """
    Finds and returns paths to datasets located in subdirectories of the provided root path.
    Each subdirectory should contain 'labeled.csv', 'unlabeled.csv', and 'test.csv'.
    """
    print("root_path: ",root_path)
    dataset_paths = []
    try:
        for subdir, dirs, files in os.walk(root_path):
#             print("subdir: ",subdir)
#             print("dirs: ",dirs)
#             print("files: ",files)
            if 'labeled.csv' in files and 'unlabeled.csv' in files and 'test.csv' in files:
                dataset_paths.append(subdir)
#                 print("Found dataset paths: ", dataset_paths)
            else:
                print("Dataset paths are not found ")
    except Exception as e:
        print(f"An error occurred while walking through {root_path}: {e}")
    
    dataset_paths.sort()

    print("Found dataset paths: ", dataset_paths)
    return dataset_paths



# Function to clean and normalize text
def clean_text(text):
    return text.encode('ascii', 'ignore').decode('ascii')




def process_dataframe(df):
    # Modify this based on the actual preprocessing steps needed from the notebook
#     df['manually_label'] = df['manually_label'].map({'positive': 1, 'negative': 0})
    
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
    df['clean_message'] = df['clean_message'].astype(str)

    df['clean_message'] = df['clean_message'].apply(clean_text)
    
    
    return df

def load_dataset(input_file):
    '''Creates examples for the training and dev sets using Pandas.'''
    examples = []

    # Read the CSV file into a DataFrame
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file,encoding='utf-8')
    
    df =process_dataframe(df)

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        text = row['clean_message']  # Extract the text from the 'clean_message' column
        label = row['manually_label']  # Extract the label from the 'manually_label' column

        # Add a tuple of the text and label to the examples list
        examples.append((text, label))

    return examples


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_data_loader(input_examples, label_masks, label_map, do_shuffle = False, balance_label_examples = False):
  '''
  Generate a Dataloader given the input examples, eventually masked if they are 
  to be considered NOT labeled.
  '''
  examples = []

  # Count the percentage of labeled examples  
  num_labeled_examples = 0
  for label_mask in label_masks:
    if label_mask: 
      num_labeled_examples += 1
  label_mask_rate = num_labeled_examples/len(input_examples)

  # if required it applies the balance
  for index, ex in enumerate(input_examples): 
    if label_mask_rate == 1 or not balance_label_examples:
      examples.append((ex, label_masks[index]))
    else:
      # IT SIMULATE A LABELED EXAMPLE
      if label_masks[index]:
        balance = int(1/label_mask_rate)
        balance = int(math.log(balance,2))
        if balance < 1:
          balance = 1
        for b in range(0, int(balance)):
          examples.append((ex, label_masks[index]))
      else:
        examples.append((ex, label_masks[index]))

  #-----------------------------------------------
  # Generate input examples to the Transformer
  #-----------------------------------------------
  input_ids = []
  input_mask_array = []
  label_mask_array = []
  label_id_array = []
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  # Tokenization 
  for (text, label_mask) in examples:
    encoded_sent = tokenizer(text[0],max_length=max_seq_length, padding="max_length", truncation=True) #tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
    input_ids.append(encoded_sent.input_ids)
#     input_ids.append(encoded_sent)
    label_id_array.append(label_map[text[1]])
    label_mask_array.append(label_mask)

  # Attention to token (to ignore padded input wordpieces)
  for sent in input_ids:
#     sent = sent.squeeze() 
    att_mask = [int(token_id > 0) for token_id in sent]                          
    input_mask_array.append(att_mask)
  # Convertion to Tensor
  input_ids = torch.tensor(input_ids)
#   input_ids = torch.stack(input_ids)
#   input_mask_array = torch.stack(input_mask_array)
  input_mask_array = torch.tensor(input_mask_array)
  label_id_array = torch.tensor(label_id_array, dtype=torch.long)
  label_mask_array = torch.tensor(label_mask_array)

  # Building the TensorDataset
  dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

  if do_shuffle:
    sampler = RandomSampler
  else:
    sampler = SequentialSampler

  # Building the DataLoader
  return DataLoader(
              dataset,  # The training samples.
              sampler = sampler(dataset), 
              batch_size = batch_size) # Trains with this batch size.


# from transformers import AutoModel, AutoTokenizer

def train_model(train_dataloader,test_dataloader , labeled_examples,unlabeled_examples,discriminator,generator,output_folder):
    '''Function to train the GANBERT model.'''
    

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    #models parameters
    transformer_vars = [i for i in transformer.parameters()]
    d_vars = transformer_vars + [v for v in discriminator.parameters()]
    g_vars = [v for v in generator.parameters()]

    #optimizer
    dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
    gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator) 

    #scheduler
    if apply_scheduler:
        num_train_examples = len(train_examples)
        num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

        scheduler_d = get_constant_schedule_with_warmup(dis_optimizer, 
                                               num_warmup_steps = num_warmup_steps)
        scheduler_g = get_constant_schedule_with_warmup(gen_optimizer, 
                                               num_warmup_steps = num_warmup_steps)

    # For each epoch...
    for epoch_i in range(0, num_train_epochs):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        tr_g_loss = 0
        tr_d_loss = 0

        # Put the model into training mode.
        transformer.train() 
        generator.train()
        discriminator.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every print_each_n_step batches.
            if step % print_each_n_step == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_label_mask = batch[3].to(device)

            real_batch_size = b_input_ids.shape[0]
         
            # Encode real data in the Transformer
            
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            
            # Generate fake data that should have the same distribution of the ones
            # encoded by the transformer. 
            # First noisy input are used in input to the Generator
            noise = torch.zeros(real_batch_size, noise_size, device=device).uniform_(0, 1)
            # Gnerate Fake data
            gen_rep = generator(noise)

            # Generate the output of the Discriminator for real and fake data.
            # First, we put together the output of the tranformer and the generator
            disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            # Then, we select the output of the disciminator
            features, logits, probs = discriminator(disciminator_input)

            # Finally, we separate the discriminator's output for the real and fake
            # data
            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            D_fake_features = features_list[1]
          
            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]
            
            probs_list = torch.split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            #---------------------------------
            #  LOSS evaluation
            #---------------------------------
            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + epsilon))
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg
      
            # Disciminator's LOSS estimation
            logits = D_real_logits[:,0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            label2one_hot = torch.nn.functional.one_hot(b_labels, len(label_list))
            per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
            per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples, 
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)
                     
            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

            #---------------------------------
            #  OPTIMIZATION
            #---------------------------------
            # Avoid gradient accumulation
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Calculate weigth updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward() 
            
            # Apply modifications
            gen_optimizer.step()
            dis_optimizer.step()

            # A detail log of the individual losses
            #print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
            #      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
            #             g_loss_d, g_feat_reg))

            # Save the losses to print them later
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

            # Update the learning rate with the scheduler
            if apply_scheduler:
                scheduler_d.step()
                scheduler_g.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)             
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
        print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
        print("  Training epcoh took: {:}".format(training_time))
        
        # Check if output_folder exists, and create it if it doesn't
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Saving model weights after training
        torch.save(transformer.state_dict(), output_folder+'/transformer_model_weights.pth')
        torch.save(generator.state_dict(), output_folder+'/generator_model_weights.pth')
        torch.save(discriminator.state_dict(), output_folder+'/discriminator_model_weights.pth')
            
        # ========================================
        #     TEST ON THE EVALUATION DATASET
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our test set.
        print("")
        print("Running Test...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        transformer.eval() #maybe redundant
        discriminator.eval()
        generator.eval()

        # Tracking variables 
        total_test_accuracy = 0
       
        total_test_loss = 0
        nb_test_steps = 0

        all_preds = []
        all_labels_ids = []

        #loss
        nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Evaluate data for one epoch
        for batch in test_dataloader:
            
            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, logits, probs = discriminator(hidden_states)
                ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
                filtered_logits = logits[:,0:-1]
                # Accumulate the test loss.
                total_test_loss += nll_loss(filtered_logits, b_labels)
                
            # Accumulate the predictions and the input labels
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        # Report the final accuracy for this validation run.
        all_preds = torch.stack(all_preds).numpy()
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        print("  Accuracy: {0:.3f}".format(test_accuracy))

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_loss = avg_test_loss.item()
        
        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)
        
        print("  Test Loss: {0:.3f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss generator': avg_train_loss_g,
                'Training Loss discriminator': avg_train_loss_d,
                'Valid. Loss': avg_test_loss,
                'Valid. Accur.': test_accuracy,
                'Training Time': training_time,
                'Test Time': test_time
            }
        )


def test_model(test_dataloader, test_examples,output_folder,discriminator,generator):
    '''Function to test the GANBERT model.'''
    # Load the model weights
    transformer.load_state_dict(torch.load(output_folder+'/transformer_model_weights.pth'))
    generator.load_state_dict(torch.load(output_folder+'/generator_model_weights.pth'))
    discriminator.load_state_dict(torch.load(output_folder+'/discriminator_model_weights.pth'))

    transformer.eval()
    generator.eval()
    discriminator.eval()

    print("Running Test...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    transformer.eval() #maybe redundant
    discriminator.eval()
    generator.eval()

    # Tracking variables 
    total_test_accuracy = 0

    total_test_loss = 0
    nb_test_steps = 0

    all_preds = []
    all_labels_ids = []

    #loss
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Evaluate data for one epoch
    for batch in test_dataloader:

        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = discriminator(hidden_states)
            ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:,0:-1]
            # Accumulate the test loss.
            total_test_loss += nll_loss(filtered_logits, b_labels)

        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        all_preds += preds.detach().cpu()
        all_labels_ids += b_labels.detach().cpu()

    # Report the final accuracy for this validation run.
    all_preds = torch.stack(all_preds).numpy()
    all_labels_ids = torch.stack(all_labels_ids).numpy()
    test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
    print("  Accuracy: {0:.3f}".format(test_accuracy))

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_loss = avg_test_loss.item()

    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)

    print("  Test Loss: {0:.3f}".format(avg_test_loss))
    print("  Test took: {:}".format(test_time))

    # Return the test results

    # Convert to numpy arrays if they aren't already
    true_labels = all_labels_ids #np.array(all_labels_ids)
    predictions = all_preds #np.array(all_preds)

    # Classification report
    report = classification_report(true_labels, predictions)
    print("Classification Report:\n", report)

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", conf_matrix)
    return true_labels,predictions

def convert_into_dataloader(labeled_examples,unlabeled_examples,test_examples):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    #------------------------------
    #   Load the train dataset
    #------------------------------
    train_examples = labeled_examples
    #The labeled (train) dataset is assigned with a mask set to True
    train_label_masks = np.ones(len(labeled_examples), dtype=bool)
    #If unlabel examples are available
    if unlabeled_examples:
        train_examples = train_examples + unlabeled_examples
          #The unlabeled (train) dataset is assigned with a mask set to False
        tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
        train_label_masks = np.concatenate([train_label_masks,tmp_masks])

    train_dataloader = generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance)

    #------------------------------
    #   Load the test dataset
    #------------------------------
    #The labeled (test) dataset is assigned with a mask set to True
    test_label_masks = np.ones(len(test_examples), dtype=bool)

    test_dataloader = generate_data_loader(test_examples, test_label_masks, label_map, do_shuffle = False, balance_label_examples = False)
    return label_map,train_dataloader, test_dataloader

def instantiate_discriminator_generator():
    # The config file is required to get the dimension of the vector produced by 
    # the underlying transformer
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]

    #-------------------------------------------------
    #   Instantiate the Generator and Discriminator
    #-------------------------------------------------
    generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
    discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)

    # Put everything in the GPU if available
    if torch.cuda.is_available():    
        generator.cuda()
        discriminator.cuda()
        
    return generator,discriminator

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import json

def save_metrics(y_true, y_pred, output_folder,result_path):
    # Calculating the metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Preparing the metrics dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        'precision': precision,

    }

    # Saving the metrics to a file
    with open(os.path.join(output_folder, 'results.txt'), 'a') as file:
        file.write(f"\n# Results for: {result_path}\n")
        file.write(json.dumps(metrics_dict, indent=4))
        file.write("\nConfusion Matrix:\n")
        file.write(str(conf_matrix))
        file.write("\n" + "-"*50 + "\n")  # Adding a separator for clarity

        
def restart_cuda():
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


# No need to set the device manually using torch.cuda.set_device(device)

print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')

# Main function with automated experimentation
def main():
    print("Starting the experiments.....")
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description='Run GANBERT script with specified parameters.')
    parser.add_argument('-root_dataset_path', type=str, required=True, help='Root path for the dataset')
    parser.add_argument('-output_folder', type=str, required=True, help='Output folder for results')
    args = parser.parse_args()

    root_dataset_path = args.root_dataset_path
    output_folder = args.output_folder
    
    print("root_dataset_path: ",root_dataset_path)
    print("output_folder: ",output_folder)


    labeled_path = os.path.join(root_dataset_path, 'labeled.csv')
    unlabeled_path = os.path.join(root_dataset_path, 'unlabeled.csv')
    test_path = os.path.join(root_dataset_path, 'test.csv')

    # Load and preprocess dataset
    labeled_examples = load_dataset(labeled_path)
    unlabeled_examples = load_dataset(unlabeled_path)
    test_examples = load_dataset(test_path)

    label_map, train_dataloader, test_dataloader = convert_into_dataloader(labeled_examples,unlabeled_examples,test_examples)
    generator,discriminator = instantiate_discriminator_generator()

    transformer.to(device)
    generator.to(device)
    discriminator.to(device)

    # Train the model
    trained_model = train_model(train_dataloader,test_dataloader, labeled_examples,unlabeled_examples,discriminator,generator,result_folder)

    # Test the model
    true_labels,predictions  = test_model(test_dataloader, test_examples,result_folder,discriminator,generator)

    # Save results to output folder
    save_metrics(true_labels,predictions, output_folder,result_folder)

    print(f'Results for {dataset_path} saved to {result_folder}')
    print("\n" + "-"*50 + "\n")



if __name__ == "__main__":
    main()


    
    
#  python3 gan_bert_train_test.py -root_dataset_path '/datasefolder/‘ -output_folder '/output/‘

