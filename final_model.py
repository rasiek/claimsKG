
# local libraries
from plotting import Claim
from text_preprocessing import TextPreprocessing


# python
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt




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
from sklearn.svm import LinearSVC



class Final_Model():

    def __init__(self, *args, balanced_input=False) -> None:
        

        self.label_encoder = LabelEncoder()
        self.skf = StratifiedKFold(n_splits=10)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if balanced_input != False:
            self.df_balanced = pd.read_csv(balanced_input)
            self.X_balanced, self.y_balanced = self.__corpus_recovery(single=True)

        self.df_list = [pd.read_csv(x) for x in args]

        self.X, self.y = self.__corpus_recovery()
        self.kfold = kfold = KFold(n_splits=10, random_state=42, shuffle=True)

        self.model = Pipeline([
                ('preprocessing', TextPreprocessing()),
                ('tfidf', TfidfVectorizer()),
                ('sgd', SGDClassifier())
            ])

        self.params = {
                # 'preprocessing__removestopwords': [True],
                # 'preprocessing__getlemmas': [True, False],
                # 'preprocessing__getstemmer': [True],
                'tfidf__analyzer': ['word', 'char_wb'],
                'tfidf__ngram_range': [(1,1), (1,2), (1,5)],
                'sgd__alpha': [1e-2, 1e-3],
                # 'sgd__tol': [1e-3, None],
                # 'sgd__random_state': [None, 42]
            }

        self.grid = GridSearchCV(
                estimator = self.model,
                param_grid = self.params,
                n_jobs=-1
            )

        # self.__grid_exec_SGD()
        self.__prediction()
        self.__prediction_balance()


    def __corpus_recovery(self, single=False):
        """
        Returns two lists of the tokenized the body of the Claim and its truth value adjusted to the claimsKG scale 
        """
        claims_list = []

        if single:

            claims_body = self.df_balanced['claimReview_claimReviewed'].values.tolist()
            claims_value = self.df_balanced['rating_alternateName'].values.tolist()

            for (body, value) in zip(claims_body, claims_value):
                claims_list.append(Claim(body, value))
        

            return np.array([x.body for x in claims_list]), self.label_encoder.fit_transform([x.value for  x in claims_list])

        else:

            for df in self.df_list:

                claims_body = df['claimReview_claimReviewed'].values.tolist()
                claims_value = df['rating_alternateName'].values.tolist()

                for (body, value) in zip(claims_body, claims_value):
                    claims_list.append(Claim(body, value))
            

            return np.array([x.body for x in claims_list]), self.label_encoder.fit_transform([x.value for  x in claims_list])



    def __error_analisis(self, df_preds):

        df_FNF = df_preds[(df_preds['valeur'] == 'False') & (df_preds['predicted'] != 'False')]
        df_FNT = df_preds[(df_preds['valeur'] == 'True') & (df_preds['predicted'] != 'True')]
        df_FNM = df_preds[(df_preds['valeur'] == 'Mixed') & (df_preds['predicted'] != 'Mixed')]

        df_FNF.to_csv('claims_pred_FNF.csv')
        df_FNT.to_csv('claims_pred_FNT.csv')
        df_FNM.to_csv('claims_pred_FNM.csv')


    def __prediction(self):


        for train_i, test_i in self.skf.split(np.array(self.X), self.y):
            self.X_train, self.X_test = self.X[train_i], self.X[test_i]
            self.y_train, self.y_test = self.y[train_i], self.y[test_i]

        self.model.fit(self.X_train, self.y_train)

        print(self.label_encoder.inverse_transform(self.y_test))

        y_pred = self.model.predict(self.X_test)

        res_metrics = metrics.classification_report(self.y_test, y_pred, output_dict=True)

        with open('claimspredicted.csv', 'w') as f:

            f.write("claims|valeur|predicted\n")

            for claim, val, pred in zip(self.X_test, self.label_encoder.inverse_transform(self.y_test), self.label_encoder.inverse_transform(y_pred)):
                f.write(f"{claim}|{val}|{pred}" + "\n")


        df_preds = pd.read_csv('claimspredicted.csv', sep='|')

        self.__error_analisis(df_preds)

        df_metrics = pd.DataFrame(res_metrics).T

        df_metrics['support'] = df_metrics.support.apply(int)

        sns.heatmap(df_metrics, annot=True, fmt='g')
        plt.savefig('SGD_repport.jpg')
        plt.clf()

        tick_labels = list(self.label_encoder.classes_)
        labels = list(self.label_encoder.transform(tick_labels))

        sns.heatmap(confusion_matrix(self.y_test,y_pred,labels=labels), annot=True, xticklabels=tick_labels, yticklabels=tick_labels, fmt='g')

        plt.savefig('heatmap_SGD.jpg')


    def __prediction_balance(self):


        for train_i, test_i in self.skf.split(np.array(self.X), self.y):
            self.X_train, self.X_test = self.X[train_i], self.X[test_i]
            self.y_train, self.y_test = self.y[train_i], self.y[test_i]

        self.model.fit(self.X_balanced, self.y_balanced)

        print(self.label_encoder.inverse_transform(self.y_test))

        y_pred = self.model.predict(self.X_test)

        res_metrics = metrics.classification_report(self.y_test, y_pred, output_dict=True)

        with open('claimspredicted_balanced.csv', 'w') as f:

            f.write("claims|valeur|predicted\n")

            for claim, val, pred in zip(self.X_test, self.label_encoder.inverse_transform(self.y_test), self.label_encoder.inverse_transform(y_pred)):
                f.write(f"{claim}|{val}|{pred}" + "\n")


        df_preds = pd.read_csv('claimspredicted_balanced.csv', sep='|')

        self.__error_analisis(df_preds)

        df_metrics = pd.DataFrame(res_metrics).T

        df_metrics['support'] = df_metrics.support.apply(int)


        plt.clf()

        sns.heatmap(df_metrics, annot=True, fmt='g')
        plt.savefig('SGD_repport_balanced.jpg')
        plt.clf()

        tick_labels = list(self.label_encoder.classes_)
        labels = list(self.label_encoder.transform(tick_labels))

        sns.heatmap(confusion_matrix(self.y_test,y_pred,labels=labels), annot=True, xticklabels=tick_labels, yticklabels=tick_labels, fmt='g')

        plt.savefig('heatmap_SGD_balanced.jpg')


    def __grid_exec_SGD(self):

        
        # self.grid.fit(self.X, self.y)
        # resdf = pd.DataFrame(self.grid.cv_results_)
        # resdf.to_csv('scores_final_model.csv')

        df = cross_val_score(self.grid, self.X, self.y)

        # nl = "\n"
        # print(f"""Best Estimator: {self.grid.best_estimator_}{nl}
        #         Best parametres: {self.grid.best_params_}{nl}
        #         Best score: {self.grid.best_score_}""")





Final_Model('output_got_complete.csv', balanced_input='output_balanced.csv')









        
