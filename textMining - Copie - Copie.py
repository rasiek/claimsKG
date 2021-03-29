
import pandas as pd
import string
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#phase de recuperation de tous les claims
def textmining(fichiers):
    corpus0=pd.read_csv(fichiers)
    corpus= corpus0[['claimReview_claimReviewed']].copy()
    y=corpus0[['rating_alternateName']].copy()
    #tokenisation
    #print(corpus2)
    #Convertir une collection de documents bruts en une matrice de fonctionnalités TF-IDF.
    #analyseur: pour gérer le prétraitement, la tokenisation et la génération de n-grammes
    #instancie l'objet vectoriseur
    vect=TfidfVectorizer(stop_words='english', analyzer='word')# enlever les mots vides
    
    tfidf_mat=vect.fit_transform(corpus.claimReview_claimReviewed.fillna(' '))
    
    feature_names=vect.get_feature_names()
    
    dense=tfidf_mat.todense()#converti matrice en array
   
    denselist=dense.tolist()# convertir array to list
    X= pd.DataFrame(denselist, columns=feature_names) #convertir list en dataframe
    
    
    
    
    clf = KNeighborsClassifier(n_neighbors=3)
    
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y.values.ravel(), cv=8)
    print(scores)
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(clf, X, y.values.ravel(), cv=8)
    labels = ["True","False"]
    #labels = np.asarray(labels).reshape(1,1)
    sns.heatmap(confusion_matrix(y.values.ravel(),predicted,labels=["true", "false"]), annot=True )
    
    plt.ylabel('valeur cible')
    plt.xlabel('Prediction ')
    plt.show()
    
textmining('output_got_politifact.csv')

