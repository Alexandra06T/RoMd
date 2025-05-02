import re
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd

nlp = spacy.load("ro_core_news_sm")

def clean_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)


def remove_diacritics(text):
    return text.replace('ț', 't').replace('ţ', 't').replace('ș', 's').replace('ş', 's').replace('ă', 'a').replace('â', 'a').replace('î', 'i')


def remove_stopwords_from_text(text):
    stopwords_ro = stopwords.words('romanian')
    stopwords_ro = [remove_diacritics(s) for s in stopwords_ro]
    words = text.split()
    words = [word for word in words if remove_diacritics(word) not in stopwords_ro]
    return " ".join(words)


def prepare_dataset_for_training(train_data, valid_data=None, test_data=None, remove_stopwords=False):
    print('Preparing dataset...')
    train_data['cleaned'] = train_data['sample'].apply(clean_and_lemmatize).astype(str)
    # train_data['cleaned'] = train_data.iloc[:100, 1].apply(clean_and_lemmatize).astype(str)

    if remove_stopwords:
        X_train = train_data['cleaned'].apply(remove_stopwords_from_text)
        # X_train = train_data.iloc[:100, 1].apply(remove_stopwords_from_text)
    else:
        X_train = train_data['cleaned']
        # X_train = train_data.iloc[:100, 1]
    y_train = train_data['dialect'].apply(lambda x: x - 1)
    # y_train = train_data.iloc[:100, 2].apply(lambda x: x - 1)

    if test_data is None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        X_train = pd.Series(X_train).reset_index(drop=True)
        X_test = pd.Series(X_test).reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)
    else:
        X_test = test_data['sample'].apply(clean_and_lemmatize)
        # X_test = test_data.iloc[:100, 1].apply(clean_and_lemmatize)
        if remove_stopwords:
            X_test = X_test.apply(remove_stopwords_from_text)
        y_test = test_data['dialect'].apply(lambda x: x - 1)
        # y_test = test_data.iloc[:100, 2].apply(lambda x: x - 1)

    if valid_data is None:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        X_train = pd.Series(X_train).reset_index(drop=True)
        X_valid = pd.Series(X_valid).reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True)
        y_valid = pd.Series(y_valid).reset_index(drop=True)
    else:
        X_valid = valid_data['sample'].apply(clean_and_lemmatize)
        # X_valid = valid_data.iloc[:100, 1].apply(clean_and_lemmatize)
        if remove_stopwords:
            X_valid = X_valid.apply(remove_stopwords_from_text)
        y_valid = valid_data['dialect'].apply(lambda x: x - 1)
        # y_valid = valid_data.iloc[:100, 2].apply(lambda x: x - 1)


    print('Dataset prepared.')
    return X_train, X_valid, y_train, y_valid, X_test, y_test

