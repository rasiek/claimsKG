import json
from numpy import vectorize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pandas as pd


class Value:

    TRUE = 'True'
    FALSE = 'False'
    MIXED = 'Mixed'


class Claim:

    def __init__(self, body, value) -> None:
        
        self.body = body
        self.value = self.set_value(value)

    def set_value(self, value):

        if value == 'True':
            return Value.TRUE
        
        elif value == 'False':
            return Value.FALSE

        else:
            return Value.MIXED



filename = 'output_got.csv'


claims = []

claims_df = pd.read_csv(filename)

claims_body = claims_df['claimReview_claimReviewed'].values.tolist()
claims_value = claims_df['rating_alternateName'].values.tolist()

# print(len(claims_body))
# print(len(claims_value))

for (body, value) in zip(claims_body, claims_value):
    claims.append(Claim(body, value))

# print(claims)


train_set, test_set = train_test_split(claims, test_size=0.33, random_state=42)

train_body = [x.body for x in train_set]
train_value = [x.value for x in train_set]

test_body = [x.body for x in test_set]
test_value = [x.value for x in test_set]

print(len(train_body), len(test_body))


vectorizer = CountVectorizer()

train_body_vectors = vectorizer.fit_transform(train_body)
test_body_vectors = vectorizer.transform(test_body)

print(train_body_vectors)

clf_svm = SVC(kernel= 'linear')

clf_svm.fit(train_body_vectors, train_value)

count = 0
correct_predictions = 0
for i in test_body_vectors:

    prediction = clf_svm.predict(i)

    if prediction == test_value[count]:
        correct_predictions += 1

    count += 1


print(f'{correct_predictions} correct predictions out of {len(test_body)}')