

## Steps to Train BERT

1. Prepare a CSV file containing the following columns:
    - `commit_message`: The commit message text.
    - `label`: The label indicating if the commit message is security-related (1) or not (0).
2. The training process will use a split of 90% for training and 10% for testing.
3. Please make sure to check the column names for both the commit message and their labels correct in the file.
4. Run the following command to start the training process:

    ```bash
    python3 bert_train.py '/your_data_folder'
    ```

#### The weights of the trained model will be stored at `/your_data_folder/saved_bert_models/`.

## Steps to Test BERT to Evaluate Its Performance

1. Prepare a CSV file containing the following columns:
    - `commit_message`: The commit message text.
    - `label`: The label indicating if the commit message is security-related (1) or not (0).
2. Update the trained BERT model weights in the file like this: `/your_data_folder/saved_bert_models`.
3. Please make sure to check the column names for both the commit message and their labels correct in the file.
4. Run the following command to start the testing process, providing the same dataset used in training BERT to test on the remaining 10%:

    ```bash
    python3 bert_test.py '/your_data_folder'
    ```

#### The output will be a CSV file containing the predicted labels and original commit messages in `/result.txt`.


## Steps to use BERTopic to find topics from commit messages.

1. Use the BERT_Topic.ipynb file
2. The BERT_Topic.ipynb notebook trains the BERTopic model on commit messages to find top topics from commit messages.
