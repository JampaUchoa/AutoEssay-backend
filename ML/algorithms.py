from collections import Counter

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
import re
import uol_redacoes_xml
import numpy as np
import liwc
import pandas as pd


def get_tf_idf_features(essay):
    """
    Gera o TF-IDF para ser utilizado na classificação
    :param essay: Redação recebida
    :return: TF-IDF
    """

    essays = uol_redacoes_xml.load()
    texts = [essay.text for essay in essays]

    count_vect = CountVectorizer()
    texts_count = count_vect.fit_transform(texts)
    counts = count_vect.transform(essay)

    tf_idf_transformer = TfidfTransformer(use_idf=True)
    return tf_idf_transformer.fit_transform(counts)


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


def get_liwc_features(essay):
    parse, category_names = liwc.load_token_parser('ML/resources/LIWC2007_Portugues_win.dic.txt')
    counter = Counter({x: 0 for x in category_names})
    tokens = tokenize(essay)
    counter.update(category for token in tokens for category in parse(token))
    dic = {0: counter}
    liwc_df = pd.DataFrame.from_dict(dic, orient='index').fillna(0)
    sc = StandardScaler()
    liwc_features = sc.fit_transform(liwc_df)

    return liwc_features


class CoherenceClf:
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

        tf_idf_features = get_tf_idf_features([self._text])

        return linear_reg_model.predict(tf_idf_features)[0]

    def random_forest_score(self):
        """
        Classifica usando Random Forest Regressor
        :return: nota
        """
        random_forest_model = joblib.load('ML/models/random_forest_model.sav')

        liwc_features = get_liwc_features(self._text)

        return random_forest_model.predict(liwc_features)[0]

    def mean_score(self):
        """

        :return: média ponderada da nota.
        """
        rf_score = self.random_forest_score()
        l_reg_score = self.linear_score()
        return np.mean([0.80 * l_reg_score, 0.7617450923787529 * rf_score])

