from numpy import append
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

names = ['Logistic Regression',
        'Gaussian NB',
        'Decision Tree',
        'K Neighboors',
        'System Vector'
]
classifiers =[
LogisticRegression(max_iter=1000),
GaussianNB(),
DecisionTreeClassifier(),
KNeighborsClassifier(),
SVC(kernel='linear')
] 

label_encoder = LabelEncoder()
ngram_vect = CountVectorizer(analyzer='char_wb', ngram_range=(5,5))
data = pd.read_csv('output_got.csv')
corpus = data['claimReview_claimReviewed'].values.tolist()
values = []
for value in data['rating_alternateName'].values.tolist():
    if value == 'True' or value == 'False':
            values.append(value)
    else:
        values.append('Mixed')


X = ngram_vect.fit_transform(corpus)
y = label_encoder.fit_transform(values)


for name, clf in zip(names, classifiers):

    scores = cross_val_score(clf, X.toarray(), y, cv=5)
    print(f'clf: {name}, scores: {scores}')











