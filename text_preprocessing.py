import pandas as pd
import sklearn


# sklearn
from sklearn.base import BaseEstimator, TransformerMixin



# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk import word_tokenize



# python
import re
import string



class TextPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self,
                removestopwords=False,
                getstemmer=False,
                getlemmas=False) -> None:
        
        self.removestopwords = removestopwords
        self.getstemmer = getstemmer
        self.getlemmas = getlemmas
        self.stop_words = set(stopwords.words('english'))

    
    def transform(self, X, **transform_params):

        X=X.copy()
        return [self.__text_cleaner(text) for text in X]

    def fit(self, X, y=None, **fit_params):

        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {
            'removestopwords':self.removestopwords,
            'getstemmer': self.getstemmer,
            'getlemmas': self.getlemmas
        }

    def set_params(self, **params):
        for param, val in params.items():
            setattr(self, param, val)

        return self

    
    def __text_cleaner(self, X):

        text = str(X)

        # suppression des caractères spéciaux
        # text = re.sub(r'[^\w\s]',' ', text)
        # suppression de tous les caractères uniques
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # substitution des espaces multiples par un seul espace
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        # decoupage en mots
        tokens = word_tokenize(text)

        tokens = [token.lower() for token in tokens]

        # suppression ponctuation
        table = str.maketrans('', '', string.punctuation)
        words = [token.translate(table) for token in tokens]

        # suppression des tokens non alphabetique ou numerique
        words = [word for word in words if word.isalnum()]        
        # suppression des tokens numerique

        # suppression des stopwords
        if self.removestopwords:
            words = [word for word in words if not word in self.stop_words]

        # lemmatisation
        if self.getlemmas:
            lemmatizer=WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word)for word in words]
            

        # racinisation
        if self.getstemmer:
            ps = PorterStemmer()
            words=[ps.stem(word) for word in words]
            
        text = ' '.join(words)
    
        return text






