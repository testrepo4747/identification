# GAN-BERT for Identifying Security Issues in Commit Messages

This folder contains code for training and testing GAN-BERT to identify security issues in commit messages using labeled and unlabeled data.

## Dataset Requirements

The `your_dataset` folder should contain the following files:
1. `labeled.csv`: A CSV file with labeled commit messages.
2. `unlabeled.csv`: A CSV file with unlabeled commit messages.
3. `test.csv`: A CSV file with commit messages for testing.

### CSV File Structure

- `labeled.csv` and `test.csv` should contain the following columns:
  - `commit_message`: The commit message text.
  - `label`: The label indicating if the commit message is security-related (1) or not (0).

## Steps to Train and Test GAN-BERT

1. Ensure your dataset is prepared as described above.
2. Make sure all files contain the correct column names for commit messages as given in the code.
3. Make sure all files contain correct column names for labels as given in the code.
4. Run the following command to start the training and testing process:

    ```bash
    python3 gan_bert_train_test.py -root_dataset_path '/your_dataset/' -output_folder '/output/'
    ```

### Outputs

The output will be saved in the specified `output_folder`. The results will include the model's performance metrics and the predicted labels for the test dataset.

This README.md provides a clear and concise guide for training and testing GAN-BERT. Let me know if any further adjustments are needed!
