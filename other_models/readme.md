# Training and Testing Models: LSTM, RNN, PatchRNN, SPI-CM, and E-SPI

This folder contains the code for training and testing models such as LSTM, RNN, PatchRNN, SPI-CM, and E-SPI.

## Training the Models

1. **Download Vector Embeddings**:
   - Download the vector embeddings required for training the models from [Kaggle](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) and place them in this folder.

2. **Run Jupyter Notebooks**:
   - Use the provided Jupyter notebooks to train and test the models. For example, run `LSTM.ipynb` to train the LSTM model and save the weights.

## Testing the Models

1. **Using Model Weights with test.py**:
   - Update `test.py` with the test data CSV file path and model weights path. For example:
     ```python
     # Load the entire model back.
     model = load_model('./model_weights/simple_rnn.h5')
     ```
   - Run `test.py` to test the models.

2. **Testing E-SPI**:
   - Use the `ESPI.ipynb` notebook and follow the 'Load and test the model' section to test the E-SPI model using the saved model weights.


