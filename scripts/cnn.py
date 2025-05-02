import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist

import spacy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import operator
from collections import Counter


class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Definim un embedding layer cu un vocabular de dimensiune 291
        # și ca output un embedding de dimensiune 20
        # padding_idx este indexul din vocabular al paddingului (1, în cazul nostru)

        self.embedding = torch.nn.Embedding(vocab_size, 20, padding_idx=1)

        # Definim o secvență de layere

        # Un layer Convolutional 1D cu 20 input channels, 32 output channels, dimensiune kernel = 3 și padding = 1
        # ReLU activation
        # 1D Maxpooling layer de dimensiune 2
        conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        # Un layer Convolutional 1D cu 32 input channels, 32 output channels, dimensiune kernel = 5 și padding = 2
        # ReLU activation
        # 1D Maxpooling layer de dimensiune 2
        conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
        )

        # Global Average pooling layer care, în cazul nostru, este un 1D Avgerage Pooling layer
        # cu dimensiunea de 250 și stride 250
        global_average = torch.nn.AvgPool1d(kernel_size=250, stride=250)

        self.convolutions = torch.nn.Sequential(
            conv1, conv2, global_average
        )

        # Flattening layer
        flatten = torch.nn.Flatten()

        # Linear layer cu 32 input features și 2 outputs fără funcție de activare
        linear = torch.nn.Linear(in_features=32, out_features=2)

        self.classifier = torch.nn.Sequential(flatten, linear)

    def forward(self, input):
        # trecem inputul prin layerul de embedding
        embeddings = self.embedding(input)

        # permutăm inputul astfel încât prima dimensiune este numărul de channels
        embeddings = embeddings.permute(0, 2, 1)

        # trecem inputul prin secvența de layere
        output = self.convolutions(embeddings)
        output = self.classifier(output)
        return output


def transform_to_tokens(data):

    samples = []
    for sample in data:
        sample_tokenized = word_tokenize(sample.lower())
        samples.append(sample_tokenized)

    return samples


def word_freq(data, min_aparitions):

    all_words = [words.lower() for samples in data for words in samples.split()]
    sorted_vocab = sorted(dict(Counter(all_words)).items(), key=operator.itemgetter(1))
    final_vocab = [k for k,v in sorted_vocab if v > min_aparitions]

    return final_vocab

def vectorize_samples(data, word_indices, one_hot = False):
    vectorized = []
    for samples in data:

        # transformam fiecare sample in reprezentarea lui sub forma de indici ai cuvintelor continute
        samples_of_indices = [word_indices[w] if w in word_indices.keys() else word_indices['UNK'] for w in samples.split()]

        # pentru fiecare indice putem face reprezentarea one-hot corespunzatoare
        # sau putem sa nu facem asta si sa adaugam un embedding layer in model care face această transformare
        if one_hot:
            sentences_of_indices = np.eye(len(word_indices))[samples_of_indices]

        vectorized.append(samples_of_indices)

    return vectorized

def pad(samples, max_length):

    return torch.tensor([
        sample[:max_length] + [1] * max(0, max_length - len(sample))
        for sample in samples
    ])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, k):
        """Returneaza al k-lea exemplu din dataset"""
        return self.samples[k], self.labels[k]

    def __len__(self):
        """Returneaza dimensiunea datasetului"""
        return len(self.samples)


def train(model, X_train, y_train, X_valid, y_valid, device, params):

    train = Dataset(X_train, y_train)
    loader_train = torch.utils.data.DataLoader(train, batch_size=params['batch_size'], shuffle=True)
    print('Dataset', loader_train.dataset.__len__())

    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    val_accuracies = []

    best_val_acc = 0
    # Starts training phase
    for epoch in range(params['epochs']):
        # Set model in training model
        model.train()
        predictions = []
        # Starts batch training
        for x_batch, y_batch in loader_train:
            model.zero_grad()
            y_batch = y_batch.type(torch.FloatTensor)

            x_batch = x_batch.long().to(device)
            y_batch = y_batch.long().to(device)

            # Feed the model
            y_pred = model(x_batch)

            # Loss calculation
            loss = criterion(y_pred, y_batch)

            # Gradients calculation
            loss.backward()

            # Gradients update
            optimizer.step()

            # Save predictions
            y_pred = y_pred.argmax(1)
            predictions += list(y_pred.detach().cpu().numpy())

        # Evaluation phase
        validation_predictions = evaluation(model, X_valid, y_valid, device)

        # Metrics calculation
        train_accuracy = calculate_accuracy(y_train, predictions)
        validation_accuracy = calculate_accuracy(y_valid, validation_predictions)

        val_accuracies.append(validation_accuracy)
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Validation accuracy: %.5f" % (epoch + 1, loss.item(), train_accuracy, validation_accuracy))

        if validation_accuracy > best_val_acc:
            torch.save(model.state_dict(), "./model")
            best_val_acc = validation_accuracy

    return model, best_val_acc

def evaluation(model, X_test, y_test, device):

    test = Dataset(X_test, y_test)
    loader_test = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

    # Set the model in evaluation mode
    model.eval()
    predictions = []

    # Starts evaluation phase
    with torch.no_grad():
        for x_batch, y_batch in loader_test:

            x_batch = x_batch.long().to(device)
            y_batch = y_batch.long().to(device)

            with torch.no_grad():
                y_pred = model(x_batch)

            y_pred = y_pred.argmax(1)
            print(y_pred[:10])
            predictions += list(y_pred.detach().cpu().numpy())
    return predictions

def calculate_accuracy(targets, predictions):
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    val_acc = (predictions == targets).float().mean().numpy()
    return val_acc


def test_cnn(model, X_test, y_test, device, params):
    print('Testing model...')

    # Evaluation phase
    test_predictions = evaluation(model, X_test, y_test, device)

    # Metrics calculation
    test_accuracy = calculate_accuracy(y_test, test_predictions)

    cm = confusion_matrix(y_test, test_predictions)

    print('Test accuracy:', test_accuracy)
    return test_accuracy, cm


def preprocess_dataset_samples(X_train, X_valid, X_test, min_aparitions=3):
    vocab = word_freq(X_train, min_aparitions=min_aparitions)
    word_indices = dict((c, i + 2) for i, c in enumerate(vocab))
    indices_word = dict((i + 2, c) for i, c in enumerate(vocab))

    indices_word[0] = 'UNK'
    word_indices['UNK'] = 0

    indices_word[1] = 'PAD'
    word_indices['PAD'] = 1
    print('Dimensiunea vocabularului', len(indices_word))

    train_samples_vectorized = vectorize_samples(X_train, word_indices)
    train_samples_vectorized = pad(train_samples_vectorized, max_length=1000)

    validation_samples_vectorized = vectorize_samples(X_valid, word_indices)
    validation_samples_vectorized = pad(validation_samples_vectorized, max_length=1000)

    test_samples_vectorized = vectorize_samples(X_test, word_indices)
    test_samples_vectorized = pad(test_samples_vectorized, max_length=1000)

    return train_samples_vectorized, validation_samples_vectorized, test_samples_vectorized, len(indices_word)
