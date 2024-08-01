## This folder contains train & test code for models such as LSTM, RNN, PatchRNN, SPI-CM, and E-SPI.

### If training is required, please follow the steps:
1. Download the vector embedding in this folder for training the models from https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300.
2. Run the Jupyter notebook for each model to train and test the model (e.g run LSTM.ipynb for training LSTM and save the weights)

### If testing is required using model weights, please follow the steps:
1. Use test.py to run the models by specifying the test data csv file and model weights path.
2. For E-SPI, use ESPI.ipynb file 'Load and test the model' part to test the model using model weights.
