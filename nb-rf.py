from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class NBDecisionTreeClassifier(DecisionTreeClassifier):
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.min_samples_leaf = max(self.min_samples_leaf, X.shape[1])
        super(NBDecisionTreeClassifier, self).fit(X, y, sample_weight, check_input, X_idx_sorted)

        data = pd.DataFrame(X)
        data['label'] = y
        data['leaf'] = self.apply(X)
        self.leaves_model = dict()
        for n, g in data.groupby(['leaf']):
            self.leaves_model[n] = GaussianNB()
            self.leaves_model[n].partial_fit(g.iloc[:, :-2].values, g['label'].values, classes=self.classes_)

        return self

    def predict(self, X, check_input=True):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X, check_input=True):
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        log_proba = np.empty((len(X), self.n_classes_))
        for i, x_leaf in enumerate(self.apply(X)):
            log_proba[i] = self.leaves_model[x_leaf].predict_log_proba([X[i]])
        return log_proba


def evalute_model(path):
    df = pd.read_csv(path)
    x, y = df.iloc[:, :-1].values,  df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1337)

    rf = RandomForestClassifier(n_estimators=100, random_state=1337)
    print_result("Without NaiveBayes:", rf, x_train, x_test, y_train, y_test)

    rf_nb = RandomForestClassifier(n_estimators=100, random_state=1337)
    rf_nb.base_estimator = NBDecisionTreeClassifier()
    print_result("With NaiveBayes:", rf_nb, x_train, x_test, y_train, y_test)


def print_result(header, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    res = model.predict(x_test)

    ac = accuracy_score(y_test, res)
    pr, re, fs, _ = precision_recall_fscore_support(y_test, res)
    print(header)
    print("\t\t\t%s" % "\t\t|\t".join([str(x) for x in model.classes_]))
    print("Precision\t%s" % "\t|\t".join([("%.2f" % x) for x in pr]))
    print("Recall\t\t%s" % "\t|\t".join([("%.2f" % x) for x in re]))
    print("F1-score\t%s" % "\t|\t".join([("%.2f" % x) for x in fs]))
    print("Accuracy\t%.2f\n" % ac)


if __name__ == '__main__':
    files = [
        'iris',
        'immuno',
        'phishing',
        'wine',
        'brcancer',
        'yeast',
        'occupancy',
        'magic04',
        'banknote',
        'seeds',
    ]

    for fname in files:
        print("------- %s -------" % fname)
        evalute_model("data\\%s.csv" % fname)
