

# scikit imports
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# other imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


class Claims_Processor:

    def __init__(self, csv_file, classifier) -> None:


        #Variables
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Classifiers

        self.linear = SVC(kernel= 'linear')
        self.neighbors = KNeighborsClassifier(n_neighbors=3)


        # Tools
        self.dataframe = pd.read_csv(csv_file)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.label_encoder = LabelEncoder()
        self.skf = StratifiedKFold()



        # Methods Call

        self.X, self.y = self.__text_minning_tokenization()

        if classifier == 'linear':
            self.__linear_classification()
        elif classifier == 'neighbor':
            self.__neighbor_clasification()
        


    def __text_minning_tokenization(self):
        """
        Returns two lists of the tokenized the body of the Claim and its truth value adjusted to the claimsKG scale 
        """

        claims_list = []

        claims_body = self.dataframe['claimReview_claimReviewed'].values.tolist()
        claims_value = self.dataframe['rating_alternateName'].values.tolist()

        for (body, value) in zip(claims_body, claims_value):
            claims_list.append(Claim(body, value))


        return self.vectorizer.fit_transform([x.body for x in claims_list]), self.label_encoder.fit_transform([x.value for  x in claims_list])


    
    def __neighbor_clasification(self):
        """

        """


        scores = cross_val_score(self.neighbors, self.X, self.y)
        print(scores)
        
        predicted = cross_val_predict(self.neighbors, self.X, self.y)
        labels = [0,1,2]
        tick_labels = ["True","False", "Mixed"]
        sns.heatmap(confusion_matrix(self.y,predicted, labels=labels), annot=True, xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.ylabel('valeur cible')
        plt.xlabel('Prediction ')
        plt.tick_params()
        plt.show()

    
    def __linear_classification(self):

        for train_index, test_index in self.skf.split(self.X.toarray(), self.y):

            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]


        self.linear.fit(self.X_train, self.y_train)

        prediction = cross_val_predict(self.linear, self.X_test, self.y_test)
        labels = [0,1,2]
        tick_labels = ["True","False", "Mixed"]
        sns.heatmap(confusion_matrix(self.y_test, prediction, labels=labels), annot=True, xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.ylabel('valeur cible')
        plt.xlabel('Prediction ')
        plt.tick_params()
        plt.show()



filename = 'output_got.csv'
Claims_Processor(filename, 'linear')


