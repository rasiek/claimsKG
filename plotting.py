# scikit imports
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
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

# other imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Value:
    """
    Truth Value enumeration class 
    """

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


class Claims_Processor:


    def __init__(self, classifier, *args) -> None:


        #Variables
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.CM = None

        # Classifiers

        self.classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gaussian NB': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'K Neighboors': KNeighborsClassifier(),
        'System Vector': SVC(kernel='linear')
        }

        self.linear = SVC(kernel='linear')
        self.neighbors = KNeighborsClassifier(n_neighbors=3)


        # Tools
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.df_list = [pd.read_csv(x) for x in args]
        # self.dataframe = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.skf = StratifiedKFold()



        # Methods Call

        self.X, self.y = self.__text_minning_tokenization()

        if classifier == 'linear':
            self.__linear_classification()
        elif classifier == 'neighbor':
            self.__neighbor_clasification()
        elif classifier == 'comparison':
            self.clf_comparison()


    
    def __tokenizer(self):
        pass

        


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
        

        return self.vectorizer.fit_transform([x.body for x in claims_list]), self.label_encoder.fit_transform([x.value for  x in claims_list])


    
    def __neighbor_clasification(self):
        """

        """


        scores = cross_val_score(self.neighbors, self.X, self.y)
        print(scores)
        y_pred = cross_val_predict(self.neighbors, self.X, self.y)


        labels = [0,1,2]
        tick_labels = ["True","False", "Mixed"]

        sns.heatmap(confusion_matrix(self.y, y_pred, labels=labels), annot=True, xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.ylabel('valeur cible')
        plt.xlabel('Prediction ')
        plt.tick_params()
        plt.show()

    def clf_comparison(self):

        for train_index, test_index in self.skf.split(self.X.toarray(), self.y):

            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]

        for key, clf in self.classifiers.items():
            
            clf.fit(self.X_train.toarray(), self.y_train)
            score = clf.score(self.X_test.toarray(), self.y_test)

            print(f'Classifier: {key}')

            if hasattr(clf, 'predict'):
                y_pred = clf.predict(self.X_test.toarray())
            
            # if hasattr(clf, 'predict_proba'):
            #     probability = clf.predict_proba(self.X_test.toarray())
            #     for prob, value, pred in zip(probability, self.y_test, y_pred):
            #         print(f'probability: {prob}, pred: {pred} value: {value}')

            accu = accuracy_score(self.y_test, y_pred)
            accu_not_normalized = accuracy_score(self.y_test, y_pred, normalize=False)

            print(f"""
                accuracy score: {accu},
                accuracy score not normalized: {accu_not_normalized},
                """)




    
    def __linear_classification(self):

        for train_index, test_index in self.skf.split(self.X.toarray(), self.y):

            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]

            

        


        self.linear.fit(self.X_train, self.y_train)

        y_pred = cross_val_predict(self.linear, self.X_test, self.y_test)


        labels = [0,1,2]
        tick_labels = ["True","False", "Mixed"]
        
        linear_accu =accuracy_score(self.y_test, y_pred)
        linear_accu_not_normalized =accuracy_score(self.y_test, y_pred, normalize=False)
        linear_scores = cross_val_score(self.linear, self.X_test, self.y_test)

        lin_sco_str = "cross validation scores:"
        count = 1
        nl = "\n"
        for score in linear_scores:
            lin_sco_str += f"{nl}Run {count}: {score}"
            count += 1


        print(f"""
        accuracy score: {linear_accu},
        accuracy score not normalized: {linear_accu_not_normalized},
        {lin_sco_str}
        """)
        


        self.CM = confusion_matrix(self.y_test, y_pred, labels=labels)
        sns.heatmap(self.CM, annot=True, xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.ylabel('Valeur cible')
        plt.xlabel('Prediction')
        plt.show()



# filename = 'output_got.csv'
# filename2 = 'output_got_2.csv'
# filename3 = 'output_got_3.csv'
# Claims_Processor('comparison', filename, filename2)
# Claims_Processor(filename1, 'comparison')
# Claims_Processor(filename2, 'comparison')


