
# local libraries
from plotting import Claim
from text_preprocessing import TextPreprocessing


# python
import numpy as np
import seaborn as sns
import pandas as pd



# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier



class Final_Model():

    def __init__(self, *args) -> None:
        

        self.label_encoder = LabelEncoder()
        self.skf = StratifiedKFold(n_splits=10)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.df_list = [pd.read_csv(x) for x in args]

        self.X, self.y = self.__corpus_recovery()
        self.kfold = kfold = KFold(n_splits=10, random_state=42, shuffle=True)

        self.model = Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('sgd', SGDClassifier())
            ])

        self.params = {
                'preprocessing__removestopwords': [True,False],
                'preprocessing__getlemmas': [True, False],
                'preprocessing__getstemmer': [True, False],
                'tfidf__analyzer': ['word', 'char', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
                'sgd__alpha': [1e-2, 1e-3],
                'sgd__tol': [1e-3, None],
                'sgd__random_state': [None, 42]
            }

        self.grid = GridSearchCV(
                estimator = self.model,
                param_grid = self.params,
                cv = 10,
                n_jobs=-1
            )

        self.__grid_exec_SGD()

        

        

    
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



    def __prediction(self):


        for train_i, test_i in self.skf.split(np.array(self.X), self.y):
            self.X_train, self.X_test = self.X[train_i], self.X[test_i]
            self.y_train, self.y_test = self.y[train_i], self.y[test_i]

        self.model.fit_transform(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)

        df_metrics = metrics.classification_report(self.y_test, y_pred)

        tick_labels = ["True","False", "Mixed"]

        sns.heatmap(confusion_matrix(self.y_test, y_pred, labels=[0,1,2]), annot=True, xticklabels=tick_labels, yticklabels=tick_labels)

        print(df_metrics)


    def __grid_exec_SGD(self):

        
        self.grid.fit(self.X, self.y)
        resdf = pd.DataFrame(self.grid.cv_results_)
        resdf.to_csv('scores_final_model.csv')

        nl = "\n"
        print(f"""Best Estimator: {self.grid.best_estimator_}{nl}
                Best parametres: {self.grid.best_params_}{nl}
                Best score: {self.grid.best_score_}""")





Final_Model('output_got.csv')









        
