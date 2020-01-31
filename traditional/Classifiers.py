import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class Classifiers:

    def __init__(self, text):
        self.text = text

    def SVM_Modeler(self):
        texto= []
        print(self.text)
        texto.append(self.text)
        meu_arquivo = open('traditional/data/train_text.sav', 'rb')
        treino_texto = pickle.load(meu_arquivo)
        meu_arquivo = open('traditional/data/train_classe.sav', 'rb')
        count_vect = CountVectorizer(encoding='latin-1')
        tfidf_transformer = TfidfTransformer(use_idf=True)
        treino_classe = pickle.load(meu_arquivo)
        X_treino_counts = count_vect.fit_transform(treino_texto)
        X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)
        meu_arquivo = open('traditional/SVM_Model.sav', 'rb')
        modelo = pickle.load(meu_arquivo)
        X_teste_counts = count_vect.transform(texto)
        X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
        predicted = modelo.predict(X_teste_tfidf)
        return predicted[0]

    def Tree_Modeler(self):
        texto = []
        texto.append(self.text)
        meu_arquivo = open('traditional/data/train_text.sav', 'rb')
        treino_texto = pickle.load(meu_arquivo)
        meu_arquivo = open('traditional/data/train_classe.sav', 'rb')
        count_vect = CountVectorizer(encoding='latin-1')
        tfidf_transformer = TfidfTransformer(use_idf=True)
        treino_classe = pickle.load(meu_arquivo)
        X_treino_counts = count_vect.fit_transform(treino_texto)
        X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)
        meu_arquivo = open('traditional/DecisionTree_Model.sav', 'rb')
        ModeloTree = pickle.load(meu_arquivo)
        X_teste_counts = count_vect.transform(texto)
        X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
        predTree = ModeloTree.predict(X_teste_tfidf)
        return predTree[0]
