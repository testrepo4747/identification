# CodeBERT for Generating Commit Messages from Code Changes

This folder contains code for training and testing CodeBERT to generate commit messages from code changes.

## Steps to Train CodeBERT

1. Prepare a CSV file containing code diffs alongside their corresponding commit messages.
2. The training process will use a split of 90% for training and 10% for testing.
3. Run the following command to start the training process:

    ```bash
    python3 codebert_train.py '/your_data_folder'
    ```

#### The weights of the trained model will be stored at `/your_data_folder/saved_codebert_models/`.

## Steps to Test CodeBERT to Evaluate Its Commit Message Generation 

1. Prepare a CSV file containing code diffs alongside their corresponding commit messages.
2. Update the trained CodeBERT model weights in the file like this: `/your_data_folder/saved_codebert_models/codebert_predictions.csv`.
3. Run the following command to start the testing process, provide the same dataset used in training the codeBERT to test on the remaining 10%:

    ```bash
    python3 codebert_test.py '/your_data_folder'
    ```

#### The output will be a CSV file containing generated and original commit messages in `/result.txt`.

This README.md provides a  concise guide for both training and testing CodeBERT.


## For classification on generated Commit Message, follow the steps given in the '/bert/' folder for testing BERT on commit messages.
