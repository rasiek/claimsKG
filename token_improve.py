import nltk
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.metrics import accuracy_score



ps = PorterStemmer()
data = pd.read_csv('output_got.csv')

# data.head()
# # data.info()
# truth_val = data.groupby('rating_alternateName').count()
# print(truth_val)

# plt.bar(truth_val.index.values, truth_val['claimReview_claimReviewed'])
# plt.xlabel('Truth Value')
# plt.ylabel('Number of Claims')
# plt.show()

corpus = data['claimReview_claimReviewed'].values.tolist()
labels = data['rating_alternateName'].values.tolist()



y_temp = [x if x == 'False' or x == 'True' else 'Mixed' for x in labels]
lbl_encoder = LabelEncoder()
y = lbl_encoder.fit_transform(y_temp)



stop_words = set(stopwords.words('english'))

# print(corpus)
corpus_str = ' '.join(corpus)

tokenized_text = word_tokenize(corpus_str)
# print(corpus)
filtered_corpus = []


for word in tokenized_text:
    if word not in stop_words:
        filtered_corpus.append(word)



stemed_words = []

for word in filtered_corpus:
    stemed_words.append(ps.stem(word))

print(filtered_corpus)
print('\nChange\n')
print(stemed_words)



vocabulary = list(set(filter(lambda token: token not in string.punctuation, stemed_words)))

vectorizer = CountVectorizer(vocabulary=vocabulary)
transformer = TfidfTransformer()
temp_X = vectorizer.fit_transform(corpus)
X = transformer.fit_transform(temp_X)

skf = StratifiedKFold(n_splits=2)

X_train = None
X_test = None
y_train = None
y_test = None

for train_i, test_i in skf.split(X, y):
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]


clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

pred = cross_val_score(clf, X_test, y_test)

print(pred)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

tick_labels = ["True","False", "Mixed"]

sns.heatmap(confusion_matrix(y_test, y_pred, labels=[0,1,2]), annot=True, xticklabels=tick_labels, yticklabels=tick_labels)

plt.ylabel('valeur cible')
plt.xlabel('Prediction ')
plt.tick_params()
plt.show()








