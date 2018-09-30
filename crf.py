import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np
from utils import data_converter
from utils import conlleval


def word_features(sent, idx):
    word = sent[idx][0]

    features = {
        "word": word.lower(),
        "first": idx == 0,
        "last": idx == len(sent) - 1,
        "POS": sent[idx][1],
        "POS2": sent[idx][2],
        "upper": word.isupper(),
        "digit": word.isdigit(),
        "title": word.istitle(),
        "bigram1": word[:2],
        "bigram-1": word[-2:],
        "hyphen": "-" in word
    }
    if idx != 0:
        pre = sent[idx-1][0]
        features.update({
            "pre_word": pre.lower(),
            'pre_POS': sent[idx-1][1],
            'pre_POS2': sent[idx-1][2],
            'pre_upper': pre.isupper(),
            'pre_title': pre.istitle(),

        })
    if idx < len(sent) - 1:
        next = sent[idx + 1][0]
        features.update({
            "next_word": next.lower(),
            'next_POS': sent[idx + 1][1],
            'next_POS2': sent[idx + 1][2],
            'next_upper': next.isupper(),
            'next_title': next.istitle(),

        })
    return features
#case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}





def sentence_features(sentence):
    return [word_features(sentence, idx) for idx, word in enumerate(sentence)]


def crf(X, y, X_test, y_test):
    X_features = [sentence_features(sent) for sent in X]
    test_features = [sentence_features(sent) for sent in X_test]

    model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,max_iterations=100, all_possible_transitions=True)
    #choose_model = GridSearchCV(model, cv=3, param_grid={"c1": 0.2 * np.arange(0, 1) + 0.1,"c2": 0.2 * np.arange(0, 1) + 0.1,})
    #choose_model.fit(X_features, y)

    #best_model = choose_model.best_estimator_
    #print(choose_model.best_params_)

    model.fit(X_features, y)
    labels = list(model.classes_)
    labels.remove('O')

    y_pred = model.predict(test_features)
    #print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
    new_conll_sents = data_converter.add_column(X_test, y_pred)
    conlleval.evaluate(data_converter.conll_to_data_stream(new_conll_sents, write_to_file="crf_results.testb"))
