import pandas as pd
import string
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#phase de recuperation de tous les claims
def textmining(fichiers):  
    corpus=pd.read_csv(fichiers)
    corpus= corpus[['claimReview_claimReviewed']].copy()
    #tokenisation
    print(corpus)
    vect=TfidfVectorizer(stop_words='english', analyzer='word')
    tfidf_mat=vect.fit_transform(corpus.claimReview_claimReviewed.fillna(' '))
    feature_names=vect.get_feature_names()
    dense=tfidf_mat.todense()
    denselist=dense.tolist()# convertir array to list
    df2= pd.DataFrame(denselist, columns=feature_names) #convertir list en dataframe
    print(df2)
textmining('output_got.csv')
