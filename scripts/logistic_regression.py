from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_validate_logistic_regression(X_train, y_train, X_valid, y_valid):
    print('Training logistic regression...')

    pipeline_lr = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=None)),
        ('clf', LogisticRegression())
    ])

    param_grid_lr = {
        'tfidf__max_features': [500, 1000],
        'tfidf__ngram_range':[(1,1), (1,2), (1,3)],
        'clf__C': [0.1, 1, 3, 5],
        'clf__max_iter': [100, 200]
    }

    grid_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr, scoring='accuracy', cv=5, n_jobs=-2, verbose=2)
    grid_lr.fit(X_train, y_train)
    best_model = grid_lr.best_estimator_

    y_pred_val = best_model.predict(X_valid)
    acc_val = accuracy_score(y_valid, y_pred_val)

    print('Logistic regression validation accuracy:', acc_val)
    return best_model, acc_val


def test_model(model, X_test, y_test):
    print('Testing model...')
    # X_test = X_test.apply(clean_and_lemmatize)
    test_preds = model.predict(X_test)
    acc_val = accuracy_score(y_test, test_preds)

    cm = confusion_matrix(y_test, test_preds)

    print('Test accuracy:', acc_val)
    return acc_val, cm

