import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np


def get_tf_idf_features(text):
    """
    Gera o TF-IDF para ser utilizado na classificação
    :param text: Redação recebida
    :return: TF-IDF
    """
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(text)

    tf_idf_transformer = TfidfTransformer(use_idf=True)
    return tf_idf_transformer.fit_transform(counts)


class Classifier:
    """
     A classe faz a classificação da nota em dois algoritmos diferente ou ainda obtém a média
     ponderada pelos dois.
    """
    _text = ''

    def __init__(self, text):
        self._text = text

    def linear_score(self):
        """
        Classifica usando regressão linear
        :return: nota predita
        """
        linear_reg_model = joblib.load('ML/models/linear_regression_model.sav')

        tf_idf_features = get_tf_idf_features(self._text)

        return linear_reg_model.predict(tf_idf_features)

    def random_forest_score(self):
        """
        Classifica usando Random Forest Regressor
        :return: nota
        """
        random_forest_model = joblib.load('ML/models/random_forest_model.sav')

        tf_idf_features = get_tf_idf_features(self._text)

        return random_forest_model.predict(tf_idf_features)

    def mean_score(self):
        """

        :return: média ponderada da nota.
        """
        rf_score = self.random_forest_score()
        l_reg_score = self.linear_score()
        return np.mean([0.80 * rf_score, 0.7617450923787529 * l_reg_score])
