from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,SimpleRNN,Conv2D,MaxPooling2D
from keras.layers.embeddings import Embedding
import tensorflow as tf
import keras
# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd


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




# Load data
train_data = pd.read_csv('./labeled.csv',encoding='utf-8')
test_data = pd.read_csv('./20_c_cpp_v2.csv',encoding='utf-8')
train_data =  process_dataframe(train_data)
test_data =  process_dataframe(test_data)
# Settings
max_words = 10000 #len(tokenizer.word_index)# 10000  # vocabulary size
max_len = 50       # max length of sequences


# Tokenization
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['clean_message'])

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['clean_message'])
test_sequences = tokenizer.texts_to_sequences(test_data['clean_message'])

# Pad sequences
train_data_padded = pad_sequences(train_sequences, maxlen=max_len)
test_data_padded = pad_sequences(test_sequences, maxlen=max_len)


train_labels = to_categorical(np.asarray(train_data['manually_label']))
test_labels = to_categorical(np.asarray(test_data['manually_label']))



from tensorflow.keras.models import load_model

# Load the entire model back.
model = load_model('./model_weights/simple_rnn.h5')





# model.evaluate(test_data_padded, test_labels)
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score

from numpy import argmax

# Predict probabilities for the positive class
# test_predictions = model.predict(test_data_padded).ravel()  # flatten array to 1D if it's not already
# test_predictions_classes = (test_predictions > 0.8).astype(int)  # Convert probabilities to class predictions
# Predict classes
test_predictions = model.predict(test_data_padded)
test_predictions_classes = argmax(test_predictions, axis=1)
test_true_classes = argmax(test_labels, axis=1)

# Test labels should be in a flat array already if you prepared them for binary classification
# test_true_classes = test_labels

# Classification report
report = classification_report(test_true_classes, test_predictions_classes)
print("Classification Report:\n", report)


# Calculate metrics
accuracy = accuracy_score(test_true_classes, test_predictions_classes)
precision = precision_score(test_true_classes, test_predictions_classes)
recall = recall_score(test_true_classes, test_predictions_classes)
f1 = f1_score(test_true_classes, test_predictions_classes)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)