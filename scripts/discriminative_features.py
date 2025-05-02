import pandas as pd
from utils import clean_and_lemmatize, remove_diacritics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import shuffle
from nltk.corpus import stopwords
import spacy
import numpy as np
from tqdm import tqdm
import operator
from collections import Counter


train_data = pd.read_csv('../train_data_M.csv')
valid_data = pd.read_csv('../validation_data_M.csv')
test_data = pd.read_csv('../test_data_M.csv')

nlp = spacy.load("ro_core_news_sm")


def prepare_data():
    print('Preparing data...')
    train_data['cleaned'] = train_data['sample'].apply(clean_and_lemmatize).astype(str)
    valid_data['cleaned'] = valid_data['sample'].apply(clean_and_lemmatize).astype(str)
    test_data['cleaned'] = test_data['sample'].apply(clean_and_lemmatize).astype(str)
    # train_data['cleaned'] = train_data.iloc[0:100, 1].apply(clean_and_lemmatize).astype(str)
    # valid_data['cleaned'] = valid_data.iloc[0:100, 1].apply(clean_and_lemmatize).astype(str)
    # test_data['cleaned'] = test_data.iloc[0:100, 1].apply(clean_and_lemmatize).astype(str)

    data = pd.concat([train_data['cleaned'], valid_data['cleaned'], test_data['cleaned']], ignore_index=True)
    # data = pd.concat([train_data.iloc[0:100, 4], valid_data.iloc[0:100, 4], test_data.iloc[0:100, 4]], ignore_index=True)
    data = np.array(data)
    # print(data)

    labels = pd.concat([train_data['dialect'], valid_data['dialect'], test_data['dialect']], ignore_index=True)
    # labels = pd.concat([train_data.iloc[0:100, 2], valid_data.iloc[0:100, 2], test_data.iloc[0:100, 2]], ignore_index=True)
    labels = labels.apply(lambda x: x - 1)
    labels = list(labels)

    # shuffle
    data, labels = shuffle(data, labels)

    print('Data prepared.')

    return data, labels

def translate_to_pos(data):
    print('Translating to POS...')
    pos_data = []
    for sample in tqdm(data):
        tokens = nlp(str(sample))
        translated_sample = ' '.join([token.pos_ for token in tokens if token.pos_ not in ['PUNCT', 'SPACE']])
        pos_data.append(translated_sample)

    print('Data translated.')
    # print(pos_data[:20])
    return np.array(pos_data)

# def translate_to_morph(data):
#     print('Translating to morphological features...')
#     morph_data = []
#     for sample in tqdm(data):
#         tokens = nlp(str(sample))
#         translated_to_morph = []
#         for token in tokens:
#             if token.pos_ in ['VERB', 'AUX'] and len(token.morph.get('VerbForm')) > 0:
#                 translation = [token.morph.get('VerbForm')[0]]
#                 if len(token.morph.get("Tense")) > 0:
#                     translation.append('_')
#                     translation.append(token.morph.get("Tense")[0])
#                 if len(token.morph.get("Mood")) > 0:
#                     translation.append('_')
#                     translation.append(token.morph.get("Mood")[0])
#                 print(translation)
#                 translated_to_morph.append(''.join(translation))
#         translated_to_morph = ' '.join(translated_to_morph)
#         morph_data.append(translated_to_morph)
#
#     print('Data translated.')
#     print(morph_data[:2])
#     return np.array(morph_data)

def train_tf_idf_vectorizer(data, labels, results_filename, vocabulary=None):
    with open(results_filename, 'a') as file:

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

        file.write('\nTrain data: ' + str(X_train.shape[0]))
        file.write('\nTest data: ' + str(X_test.shape[0]) + '\n')
        print('train:', X_train.shape)
        print('test:', X_test.shape)

        # print('train', list(y_train).count(1))

        if vocabulary is None:
            model_tfidf = Pipeline([('tfidf', TfidfVectorizer()), ('mb', MultinomialNB())])
        else:
            model_tfidf = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocabulary)), ('mb', MultinomialNB())])

        model_tfidf.fit(X_train, y_train)

        print('vocabulary', model_tfidf['tfidf'].get_feature_names_out())
        probabilities = model_tfidf.predict_proba(X_test)
        print("First few probability predictions:")
        print(probabilities[:5])

        y_pred_tfidf = model_tfidf.predict(X_test)
        # print(y_test)
        # print(y_pred_tfidf)

        report = classification_report(y_test, y_pred_tfidf)

        file.writelines(['\nAccuracy: ', str(accuracy_score(y_test, y_pred_tfidf))])
        file.write('\n\nClassification Report\n')
        file.write('======================================================')
        file.writelines(['\n', report])
        print('\n Accuracy: ', accuracy_score(y_test, y_pred_tfidf))
        print('\nClassification Report')
        print('======================================================')
        print('\n', report)



data, labels = prepare_data()
print('Data shape', data.shape)
# print(data[:1])


# classify using pos
# translated_data = translate_to_pos(data)
# train_tf_idf_vectorizer(translated_data, labels,
#                         'dialect_classif_pos_results.txt')

# classify using stop words
remove_diacritics_vectorize = np.vectorize(remove_diacritics)
data_without_diacritics = remove_diacritics_vectorize(data)
print(data_without_diacritics)
stopwords_ro = stopwords.words('romanian')
stopwords_ro = list(set([remove_diacritics(s) for s in stopwords_ro]))
train_tf_idf_vectorizer(data_without_diacritics, labels,
                        '../results/dialect_classif_function_words_results.txt',
                        stopwords_ro)


# classify using verb morph
# translated_data = translate_to_morph(data)
# train_tf_idf_vectorizer(translated_data, labels,
#                         'dialect_classif_morph_results.txt')