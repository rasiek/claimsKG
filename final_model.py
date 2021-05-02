"""

"""

from plotting import Claim
import nltk
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import string


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


class Final_Model:

    def __init__(self, *args) -> None:
        """

        """

        #Tools
        self.porter_stemmer =  PorterStemmer()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.lem = WordNetLemmatizer()
        self.stop_words = stop_words = set(stopwords.words('english'))

        
        self.df_list = [pd.read_csv(x) for x in args]

        self.vocabulary = self.__create_vocabulary()



        self.claims, self.y = self.__text_minning_tokenization()

        self.models = [
            ('neighbors', KNeighborsClassifier()),
            ('bayes', MultinomialNB()),
            ('complement_NB', ComplementNB()),
            ('svc', SVC(kernel='linear'))
        ]

        self.pipes = {
            'neighbors': Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('neighbors', KNeighborsClassifier())
            ]),
            'bayes': Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('bayes', MultinomialNB())
            ]),
            'complement_nb': Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('complement_nb', ComplementNB())
            ]),
            'svc': Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('svc', SVC()),
            ])
        }

        self.params = {

            'neighbors': {
                'vectorizer__vocabulary': [None, self.vocabulary],
                'vectorizer__stop_words': [None, self.stop_words],
                'vectorizer__analyzer': ['word', 'char', 'char_wb'],
                'vectorizer__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'neighbors__weights': ['uniform', 'distance'],
                'neighbors__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'neighbors__n_neighbors': [3,5,7]
            },
            'bayes': {
                'vectorizer__vocabulary': [None, self.vocabulary],
                'vectorizer__stop_words': [None, self.stop_words],
                'vectorizer__analyzer': ['word', 'char', 'char_wb'],
                'vectorizer__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'bayes__alpha': [1.0, 0.0],
                'bayes__fit_prior': [True, False]
            },
            'svc': {
                'vectorizer__vocabulary': [None, self.vocabulary],
                'vectorizer__stop_words': [None, self.stop_words],
                'vectorizer__analyzer': ['word'],
                'vectorizer__ngram_range': [(1,1), (1,2)],
                'svc__kernel': ['linear'],
                'svc__gamma': ['scale', 'auto'],
                'svc__class_weight': ['balanced', None],
                'svc__decision_function_shape': ['ovr', 'ovo']
            },
            'complement_nb': {
                'vectorizer__vocabulary': [None, self.vocabulary],
                'vectorizer__stop_words': [None, self.stop_words],
                'vectorizer__analyzer': ['word', 'char', 'char_wb'],
                'vectorizer__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'complement_nb__alpha': [1.0, 0.0],
                'complement_nb__norm': [False, True],
            }
        }


        # print(type(self.X))
        # print(self.X)
        # self.__exec_grid()
        self.__scoring()



    def __create_vocabulary(self):
        """

        """

        # print(corpus)

        corpus_str = ''

        for df in self.df_list:
            corpus_str = ' '.join(df['claimReview_claimReviewed'].values.tolist())

        tokenized_text = word_tokenize(corpus_str)
        # print(corpus)
        filtered_corpus = []


        for word in tokenized_text:
            if word not in self.stop_words:
                filtered_corpus.append(word)

        lem_words = [self.lem.lemmatize(word) for word in filtered_corpus]

        # for word in filtered_corpus:
        #     stemed_words.append(self.porter_stemmer.stem(word))

        # print(filtered_corpus)
        # print('\nChange\n')
        # print(stemed_words)

        vocabulary = list(set(filter(lambda token: token not in string.punctuation, lem_words)))

        return vocabulary



    def __text_minning_tokenization(self):
        """
        Returns two lists of the tokenized the body of the Claim and its truth value adjusted to the claimsKG scale 
        """

        claims_list = []

        for df in self.df_list:

            claims_body = df['claimReview_claimReviewed'].values.tolist()
            claims_value = df['rating_alternateName'].values.tolist()

            for (body, value) in zip(claims_body, claims_value):
                claims_list.append(Claim(body, value))
        

        return [x.body for x in claims_list], self.label_encoder.fit_transform([x.value for  x in claims_list])


    def __exec_grid(self):

        scores = {}

        for pipe in self.pipes:

            grid = GridSearchCV(
                estimator = self.pipes[pipe],
                param_grid = self.params[pipe],
                cv = 3
            )

            grid.fit(self.claims, self.y)

            scores[pipe] = [
                grid.best_estimator_,
                grid.best_score_
            ]
            

            # resdf = pd.DataFrame(grid.cv_results_)

            # resdf.to_csv(f'scores_{pipe}.csv')

        for estimator in scores:
            print(f"""{estimator} score: {scores[estimator][1]},
                {estimator} best params: {scores[estimator][0]}
            """)

    def __scoring(self):


        tfdf_trans = TfidfVectorizer()
        X = tfdf_trans.fit_transform(self.claims).toarray()

        for name, model in self.models:

            kfold = KFold(n_splits=5)

            cv_results = cross_val_score(model, X, self.y, cv=kfold, scoring='accuracy')

            print(name)
            print(cv_results)
            print(cv_results.mean())
            print(cv_results.std())
            print("\n")











        




test = Final_Model('output_got.csv', 'output_got_test.csv')
