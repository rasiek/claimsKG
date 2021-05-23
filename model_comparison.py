"""

"""
# local libraries
from plotting import Claim
from text_preprocessing import TextPreprocessing

# nltk 
import nltk
from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# python
import string



# sklearn
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
from sklearn.linear_model import SGDClassifier



class ModelComparison:

    def __init__(self, *args) -> None:
        """

        """

        #Tools
        self.porter_stemmer =  PorterStemmer()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.lem = WordNetLemmatizer()

        
        self.df_list = [pd.read_csv(x) for x in args]

        self.claims, self.y = self.__corpus_recovery()

        self.models = [
            ('neighbors', KNeighborsClassifier()),
            ('bayes', MultinomialNB()),
            ('sgd', ComplementNB()),
            ('svc', SVC(kernel='linear'))
        ]

        self.pipes = {
            'neighbors': Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('neighbors', KNeighborsClassifier())
            ]),
            'bayes': Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('bayes', MultinomialNB())
            ]),
            'sgd': Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('sgd', SGDClassifier())
            ]),
            'svc': Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('svc', SVC()),
            ])
        }

        self.params = {

            'neighbors': {
                'preprocessing__removestopwords': [True,False],
                'preprocessing__getlemmas': [True, False],
                'preprocessing__getstemmer': [True, False],
                'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'tfidf__analyzer': ['word', 'char', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'neighbors__weights': ['uniform', 'distance'],
                'neighbors__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'neighbors__n_neighbors': [3,5,7]
            },
            'bayes': {
                'preprocessing__removestopwords': [True,False],
                'preprocessing__getlemmas': [True, False],
                'preprocessing__getstemmer': [True, False],
                'tfidf__analyzer': ['word', 'char', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'bayes__alpha': [1e-2, 1e-3, 1],
                'bayes__fit_prior': [True, False]
            },
            'svc': {
                'preprocessing__removestopwords': [True,False],
                'preprocessing__getlemmas': [True, False],
                'preprocessing__getstemmer': [True, False],
                'tfidf__analyzer': ['word', 'char', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,4), (1,5), (2,3)],
                'svc__kernel': ['linear'],
                'svc__gamma': ['scale', 'auto'],
                'svc__class_weight': ['balanced', None],
                'svc__decision_function_shape': ['ovr', 'ovo']
            },
            'sgd': {
                'preprocessing__removestopwords': [True,False],
                'preprocessing__getlemmas': [True, False],
                'preprocessing__getstemmer': [True, False],
                'tfidf__analyzer': ['word', 'char', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'sgd__alpha': [1e-2, 1e-3],
                'sgd__tol': [1e-3, None],
                'sgd__random_state': [None, 42]
            }
        }

        # self.__exec_grid()
        # self.__scoring()
        self.boxplot()



    def __corpus_recovery(self):
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
        results = []
        names = []

        for pipe in self.pipes:

            grid = GridSearchCV(
                estimator = self.pipes[pipe],
                param_grid = self.params[pipe],
                cv = 3
            )

            grid.fit(self.claims, self.y)

            scores[pipe] = [
                grid.best_estimator_,
                grid.best_score_,
                grid.best_params_
            ]
            
            results.append(grid.cv_results_)

            resdf = pd.DataFrame(grid.cv_results_)

            resdf.to_csv(f'scores_{pipe}1.csv')
        
        for pipe in scores:
            best_score = 0.0
            best_classifier = []

            if scores[pipe][1] > best_score:
                best_score = scores[pipe][1]
                best_classifier = scores[pipe]

        print(best_classifier)



    def boxplot(self):

        results = []
        names = []

        scoring = "accuracy"

        for pipe in self.pipes.keys():

            kfold = KFold(n_splits=10, random_state=42, shuffle=True)
            cv_results = cross_val_score(
                                        self.pipes[pipe], 
                                        self.claims, self.y, 
                                        cv=kfold, 
                                        scoring=scoring)

            results.append(cv_results)
            names.append(pipe)

            msg = f"{pipe}: {cv_results.mean()}, {cv_results.std()}"
            print(msg)
            print("\n")
            print(cv_results)
            print("\n")


        fig = plt.figure()
        fig.suptitle("Algorithm Comparison")
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

            













        




test = ModelComparison('output_got_complete.csv')
