import pickle
#from self import self
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


meu_arquivo = open('traditional/data/train_text.sav', 'rb')
treino_texto = pickle.load(meu_arquivo)
meu_arquivo = open('traditional/data/train_text.sav', 'rb')
treino_classe = pickle.load(meu_arquivo)

count_vect=CountVectorizer(encoding='latin-1')
tfidf_transformer= TfidfTransformer(use_idf=True)
X_treino_counts = count_vect.fit_transform(treino_texto)
X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)

text = ["treinando d teste"]
meu_arquivo = open('traditional/SVM_Model.sav', 'rb')
modelo = pickle.load(meu_arquivo)
X_teste_counts = count_vect.transform(text)
X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
predicted = modelo.predict(X_teste_tfidf)
print(predicted[0])

meu_arquivo = open('traditional/DecisionTree_Model.sav', 'rb')
ModeloTree = pickle.load(meu_arquivo)
X_teste_counts = count_vect.transform(text)
X_teste_tfidf = tfidf_transformer.transform(X_teste_counts)
predTree = ModeloTree.predict(X_teste_tfidf)
print( predTree[0])
