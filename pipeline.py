from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from transformers.normalizer import TextNormalizer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd



df = pd.read_csv("output_got.csv")

corpus = df[["claimReview_claimReviewed"]]
labels = df["rating_alternateName"].transform(lambda x : x if x == "True" or x == "False" else "Mixed")

lbl_encoder = LabelEncoder()


X = corpus.claimReview_claimReviewed.fillna(" ")
y = lbl_encoder.fit_transform(labels)

pipe = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("bayes", MultinomialNB())
])

print(pipe.get_params())

grid_params = {
    'vectorizer__analyzer': ['word', 'char', 'char_wb'],
    'vectorizer__binary': [True, False],
    'vectorizer__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
    'bayes__alpha': [0, 1.0],
}

model = GridSearchCV(estimator=pipe,
                    param_grid=grid_params,
                    cv=5)

model.fit(X, y)
resdf = pd.DataFrame(model.cv_results_)

resdf.to_csv('scores_model.csv')


