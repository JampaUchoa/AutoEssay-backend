import pickle
from self import self
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class classifiers:

    def __init__(self):
        self.texto = ''

    def SVM_Modeler(self, text):
        texto= []
        texto.append(text)
        meu_arquivo = open('Data/Train_text.sav', 'rb')
        treino_texto = pickle.load(meu_arquivo)
        meu_arquivo = open('Data/train_classe.sav', 'rb')
        count_vect = CountVectorizer(encoding='latin-1')
        tfidf_transformer = TfidfTransformer(use_idf=True)
        treino_classe = pickle.load(meu_arquivo)
        X_treino_counts = count_vect.fit_transform(treino_texto)
        X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)
        meu_arquivo = open('SVM_Model.sav', 'rb')
        modelo = pickle.load(meu_arquivo)
        X_teste_counts = count_vect.transform(texto)
        X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
        predicted = modelo.predict(X_teste_tfidf)
        return predicted[0]

    def Tree_Modeler(self, text):
        texto = []
        texto.append(text)
        meu_arquivo = open('Data/Train_text.sav', 'rb')
        treino_texto = pickle.load(meu_arquivo)
        meu_arquivo = open('Data/train_classe.sav', 'rb')
        count_vect = CountVectorizer(encoding='latin-1')
        tfidf_transformer = TfidfTransformer(use_idf=True)
        treino_classe = pickle.load(meu_arquivo)
        X_treino_counts = count_vect.fit_transform(treino_texto)
        X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)
        meu_arquivo = open('DecisionTree_Model.sav', 'rb')
        ModeloTree = pickle.load(meu_arquivo)
        X_teste_counts = count_vect.transform(texto)
        X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
        predTree = ModeloTree.predict(X_teste_tfidf)
        return predTree[0]
