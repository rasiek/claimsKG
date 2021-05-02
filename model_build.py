from numpy import quantile, vectorize
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def norm_lbls(x):

    if x == 'True' or x == 'False':
        pass
    else:
        x = 'Mixed'

df = pd.read_csv("output_got.csv")

corpus = df[["claimReview_claimReviewed"]]
labels = df["rating_alternateName"].transform(lambda x : x if x == "True" or x == "False" else "Mixed")

vectorizer = TfidfVectorizer()
lbl_encoder = LabelEncoder()
quantile_scaler = QuantileTransformer()

X = vectorizer.fit_transform(corpus.claimReview_claimReviewed.fillna(" "))
y = lbl_encoder.fit_transform(labels)

# print(y.argmax())
print("0: ",np.count_nonzero(y == 0))
print("1: ",np.count_nonzero(y == 1))
print("2: ",np.count_nonzero(y == 2))

# print(corpus)
# print(X.getrow(100))

# qt = QuantileTransformer()

# X_qt = qt.fit_transform(X, y)

# print(X_qt.getrow(100))


# clf = KNeighborsClassifier()
# skf = StratifiedKFold()

# X_train = None
# y_train = None
# X_test = None
# y_test = None

# for train_index, test_index in skf.split(X, y):

#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# labels = [0,1,2]
# tick_labels = ['True','False','Mixed']

# CM = confusion_matrix(y_test, y_pred, labels=labels)
# sns.heatmap(CM, annot=True, xticklabels=tick_labels, yticklabels=tick_labels)

# plt.ylabel('Valeur cible')
# plt.xlabel('Prediction')
# plt.show()


