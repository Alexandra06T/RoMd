import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist

import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from statistics import mean
import operator
from collections import Counter

from utils import *
from logistic_regression import *
from scripts.cnn import *

torch.backends.cudnn.enabled = False

nlp = spacy.load("ro_core_news_sm")


train_data = pd.read_csv('../train_data.csv')
test_data = pd.read_csv('../test_data.csv')
train_data_M = pd.read_csv('../train_data_M.csv')
valid_data_M = pd.read_csv('../validation_data_M.csv')
test_data_M = pd.read_csv('../test_data_M.csv')

results_file_path = '../results/dataset_model_comparison.txt'

# small dataset
print('Nitro dataset')
X_train_d, X_valid_d, y_train_d, y_valid_d, X_test_d, y_test_d = prepare_dataset_for_training(train_data, remove_stopwords=True)
# Morocco dataset
print('Morocco dataset')
X_train_d_M, X_valid_d_M, y_train_d_M, y_valid_d_M, X_test_d_M, y_test_d_M = prepare_dataset_for_training(train_data_M, valid_data_M, test_data_M, remove_stopwords=True)

# test Logistic Regression for dialect classification
# print('Nitro Dataset')
# model_lr_small_d, acc_lr_small_valid_d = train_validate_logistic_regression(X_train_d, y_train_d, X_valid_d, y_valid_d)
# acc_lr_small_d, confusion_matrix_lr_d = test_model(model_lr_small_d, X_test_d, y_test_d)
#
# print('Saving results...')
# with open(results_file_path, 'w', encoding='utf-8') as f:
#     f.write('DIALECT IDENTIFICATION\n')
#     f.write('Logistic Regression | Nitro Dataset: ' + str(acc_lr_small_d) + '\n')
#
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_lr_d,
#                               display_labels=['Română', 'Moldovenească'])
# disp.plot()
# disp.figure_.savefig('lr_small_cm.png')

# print('Morocco Dataset')
# model_lr_M_d, acc_lr_M_valid_d = train_validate_logistic_regression(X_train_d_M, y_train_d_M, X_valid_d_M, y_valid_d_M)
# acc_lr_M_d, confusion_matrix_lr_d = test_model(model_lr_M_d, X_test_d_M, y_test_d_M)
#
# print('Saving results...')
# with open(results_file_path, 'a', encoding='utf-8') as f:
#     f.write('Logistic Regression | Morocco Dataset: ' + str(acc_lr_M_d) + '\n')
#
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_lr_d,
#                               display_labels=['Română', 'Moldovenească'])
# disp.plot()
# disp.figure_.savefig('lr_M_cm.png')


# test CNN for dialect classification

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")
params = {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 4,
    }

print('Nitro Dataset')
train_samples_vectorized_sm, validation_samples_vectorized_sm, test_samples_vectorized_sm, vocab_size = preprocess_dataset_samples(X_train_d, X_valid_d, X_test_d, min_aparitions=1)
model = TextCNN(vocab_size).to(device)
model_cnn_small_d, acc_cnn_valid_small_d = train(model, train_samples_vectorized_sm, y_train_d, validation_samples_vectorized_sm, y_valid_d, device, params)
acc_cnn_small_d, confusion_matrix_cnn_d_small = test_cnn(model_cnn_small_d, test_samples_vectorized_sm, y_test_d, device, params)
with open(results_file_path, 'a', encoding='utf-8') as f:
    f.write('CNN | Nitro Dataset: ' + str(acc_cnn_small_d) + '\n')

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cnn_d_small,
                              display_labels=['Română', 'Moldovenească'])
disp.plot()
disp.figure_.savefig('cnn_small_cm.png')

print('Morocco Dataset')
train_samples_vectorized, validation_samples_vectorized, test_samples_vectorized, vocab_size = preprocess_dataset_samples(X_train_d_M, X_valid_d_M, X_test_d_M, min_aparitions=10)
model = TextCNN(vocab_size).to(device)
model_cnn_M_d, acc_cnn_M_valid_d = train(model, train_samples_vectorized, y_train_d_M, validation_samples_vectorized, y_valid_d_M, device, params)
acc_cnn_M_d, confusion_matrix_cnn_d = test_cnn(model_cnn_M_d, test_samples_vectorized, y_test_d_M, device, params)

print('Saving results...')
with open(results_file_path, 'a', encoding='utf-8') as f:
    f.write('CNN | Morocco Dataset: ' + str(acc_cnn_M_d) + '\n')

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cnn_d,
                              display_labels=['Română', 'Moldovenească'])
disp.plot()
disp.figure_.savefig('cnn_M_cm.png')

