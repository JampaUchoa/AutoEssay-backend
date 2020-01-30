import pandas as pd
import pickle
import spacy
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



nlp = spacy.load("pt_core_news_sm")

data = pd.read_csv("Banco_redacoes_base.csv")
texto, classe = data["Redacao"] , data["Nota"]

FraseSemStopWord = []
for frase in texto:
    doc = nlp(frase)
    new = [token.text.lower() for token in doc if not (token.is_stop or token.is_punct)]
    new = " ".join(new)

    ##Lemmatization
    lemmas = [token.lemma_.lower() for token in nlp(new)]
    lemmas = " ".join(lemmas)
    FraseSemStopWord.append(lemmas)

Classes = []
for frase in classe:
  new = frase
  if new != 'Nota':
    Classes.append(new)

treino_texto = FraseSemStopWord[1:1900]
teste_texto = FraseSemStopWord[1901:2159]
treino_classe = Classes[1:1900]
teste_classe = Classes[1901:2159]
teste_redacao =  FraseSemStopWord[2160:2164]
teste_nota = Classes[2160:2164]

#Frequencia
count_vect = CountVectorizer(encoding='latin-1')
X_treino_counts = count_vect.fit_transform(treino_texto) # OBS: concatenar array de stop words!!!
# X_treino_counts.shape

#TD_IDF
tfidf_transformer = TfidfTransformer(use_idf=True)#decide se vai usar o TF ou TF-IDF
X_train_tfidf = tfidf_transformer.fit_transform(X_treino_counts)
# X_train_tfidf.shape

clf = DecisionTreeClassifier(criterion = "entropy", random_state = None, max_depth = 10)
model= clf.fit(X_train_tfidf, treino_classe)


Model_SVM = open('DecisionTree_Model.sav',  'wb')

pickle.dump(clf, Model_SVM)
Model_SVM.close()


