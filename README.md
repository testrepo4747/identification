# Security Issues Identification using Commit Messages

This repository contains code for training and testing various language models to identify security issues in commit messages from open-source software (OSS) repositories. The models included are BERT, CodeBERT, GAN-BERT, and other approaches such as LSTM, RNN, PatchRNN, SPI-CM, and E-SPI.

## Directory Structure

- **bert**: Contains code for training and testing the BERT model.
- **code-bert**: Contains code for training and testing CodeBERT for generating commit messages from code changes.
- **gan-bert**: Contains code for training and testing GAN-BERT to identify security issues using labeled and unlabeled data.
- **qwen**: Contains code for training and testing the QWEN-3 and 2.5 model to identify security issues using commit messages, diffs, and both.
- **llama**: Contains code for training and testing the LLAMA-3.2 (1B and 3B) model to identify security issues using commit messages, diffs, and both.
- **other_models**: Contains code for training and testing models such as LSTM, RNN, PatchRNN, SPI-CM, and E-SPI.

## Getting Started


### LLAMA
The `qwen` folder contains code for training and testing the LLAMA-3.2 models to identify security issues using commit messages, diffs, and both featuers. The results and insights are in the paper manuscript.

### QEWEN
The `qwen` folder contains code for training and testing the QWEN-3 and 2.5 models to identify security issues using commit messages, diffs, and both featuers. The results and insights are in the paper manuscript.

### BERT

The `bert` folder contains the code for training and testing the BERT model to identify security issues in commit messages. Refer to the `bert/README.md` file for detailed instructions.

### CodeBERT

The `code-bert` folder contains the code for training and testing CodeBERT to generate commit messages from code changes. Refer to the `code-bert/README.md` file for detailed instructions.

### GAN-BERT

The `gan-bert` folder contains the code for training and testing GAN-BERT to identify security issues using labeled and unlabeled data. Refer to the `gan-bert/README.md` file for detailed instructions.

### Other Models

The `other_models` folder contains the code for training and testing models such as LSTM, RNN, PatchRNN, SPI-CM, and E-SPI. Refer to the `other_models/README.md` file for detailed instructions.

### Topic Analysis 

The `bert` folder contains the BERT_Topic.ipynb file which trains the BERTopic model on commit messages to find top topics from commit messages.
