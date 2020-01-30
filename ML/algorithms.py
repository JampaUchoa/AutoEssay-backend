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
    parse, category_names = liwc.load_token_parser('resources/LIWC2007_Portugues_win.dic.txt')
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
        linear_reg_model = joblib.load('models/linear_regression_model.sav')

        tf_idf_features = get_tf_idf_features([self._text])

        return linear_reg_model.predict(tf_idf_features)[0]*100

    def random_forest_score(self):
        """
        Classifica usando Random Forest Regressor
        :return: nota
        """
        random_forest_model = joblib.load('models/random_forest_model.sav')

        liwc_features = get_liwc_features(self._text)

        return random_forest_model.predict(liwc_features)[0]*100

    def mean_score(self):
        """

        :return: média ponderada da nota.
        """
        rf_score = self.random_forest_score()
        l_reg_score = self.linear_score()
        return np.mean([0.80 * rf_score, 0.7617450923787529 * l_reg_score])*100


texto = "Na antiguidade, a atribuição de super-poderes à superpoderes a seres mitológicos foi de suma importância " \
        "para a manutenção de uma ordem social em um contexto religioso, haja vista que o receio desses seres " \
        "influenciava o comportamento dos indivíduos. Sob tal ótica, é fato que que, no imaginário humano, " \
        "a ânsia da possibilidade de adquirir habilidades anormais persiste desde os povos antigos. Não obstante, " \
        "na contemporaneidade, com o avanço científico, a capacidade de alterar genes e permitir tais habilidades se " \
        "tornou realidade, assim, de que modo a superação dos limites humanos afetaria a sociedade?" \
        "A priori, deve-se pontuar que, apesar da de a tecnologia ser vista como progressiva e benéfica, uma vez que " \
        "seu uso está sujeito aos valores de uma sociedade, os seus efeitos podem ocasionar diversos malefícios. Em " \
        "um episódio da série Black Mirror, o exército de um estado é submetido aos efeitos de um chip implantado nos " \
        "indivíduos em que, ao distorcer a visão destes sua visão, facilitava a execução de um genocídio. " \
        "Analogamente, partindo do pressuposto de que a imposição de poder já é recorrente na sociedade, " \
        "seja pelo capital ou por força bélica, a ausência de limites para a biotecnologia seria capaz de agravar " \
        "conflitos e ocasionar atrocidades." \
        "Em segundo plano, vale ressaltar que a transcendência dos limites humanos intensificaria a desigualdade " \
        "social. No Manifesto Comunista, Marx e Engels explicitam, através de uma análise histórica, a presença da " \
        "luta de classes nos diversos períodos, em que a dinâmica de um grupo dominar outro sempre foi vigente. Desta " \
        "forma, além de já haver uma diferença entre os grupos sociais, a obtenção de atributos para uma parte da " \
        "sociedade, ou somente para indivíduos com poder aquisitivo, seria determinante para a segregação entre " \
        "humanos." \
        "Em suma, depreende-se que a superação dos limites humanos, por meio das novas tecnologias, " \
        "afetaria negativamente a sociedade em diversos âmbitos, permitindo a inferiorização de grupos sociais e " \
        "destilação de violência. Assim, o Ministério da Educação deve incluir no material didático de filosofia " \
        "atividades de discussão e reflexão acerca da tecnologia e seus limites, visando estimular o senso crítico " \
        "dos discentes no que tange os efeitos sociais da biotecnologia. Destarte, estas tecnologias tenderão a ser " \
        "limitadas, e a superação dos limites humanos se manterão, somente, na mitologia e no imaginário humano. "

clf = CoherenceClf(texto)
print("Coherence Score: %.2f" % (clf.mean_score() * 100))
