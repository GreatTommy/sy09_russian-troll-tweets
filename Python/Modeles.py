import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from itertools import combinations

# sklearn usefull
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

# discriminant analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# knn
from sklearn.neighbors import KNeighborsClassifier

# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------
#base_path = "C:\\Users\\ACER\\Desktop\\UTC\\SY09\\PROJET\\sy09_russian-troll-tweets"
base_path = "C:\\Python\\Russian-troll-tweets"
# ---------------------------------------------------


class Modeles:
    train_data = None
    test_data = None
    _X_train = None
    _y_train = None
    _X_test = None
    _y_test = None
    _accounts_train = None
    _accounts_test = None

    def __init__(self, data_cat="tweets", data=None):
        if data_cat == "tweets":
            self._load_data()
        else:
            self._load_data_accounts()

    def _load_data(self):
        self.train_data = pd.read_csv(
            os.path.join(base_path, "Data\data_with_scores\\train_csv.csv")
        )
        self.test_data = pd.read_csv(
            os.path.join(base_path, "Data\data_with_scores\\test_csv.csv")
        )

    def _load_data_accounts(self):
        self.train_data = pd.read_csv(
            os.path.join(base_path, "Data\data_with_scores\\accounts_train_csv.csv")
        )
        self.test_data = pd.read_csv(
            os.path.join(base_path, "Data\data_with_scores\\accounts_test_csv.csv")
        )

    def prepare_dataset(self, variables, acp=False, n_dim=2):
        X_train = self.train_data[variables]
        y_train = self.train_data["account_category"]
        X_test = self.test_data[variables]
        y_test = self.test_data["account_category"]
        if acp:
            X_train = self._perform_acp(X_train, n_dim)
            X_test = self._perform_acp(X_test, n_dim)
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

    def prepare_dataset_bonus(self, variables, acp=False, n_dim=2):
        X_train = self.train_data[variables]
        y_train = self.train_data["account_category"]
        X_test = self.test_data[variables]
        y_test = self.test_data["account_category"]
        accounts_train = self.train_data["author"]
        accounts_test = self.test_data["author"]
        if acp:
            X_train = self._perform_acp(X_train, n_dim)
            X_test = self._perform_acp(X_test, n_dim)
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._accounts_train = accounts_train
        self._accounts_test = accounts_test

    def _perform_acp(self, X_train, n_dim):
        pca = PCA(n_components=n_dim)
        pca.fit(X_train)
        # print(pca.explained_variance_ratio_)
        p_compo = pca.transform(X_train)
        return p_compo

    def test_model(self, model, model_name="", msg=True):
        y_pred = model.predict(self._X_test)
        accuracy = accuracy_score(self._y_test, y_pred)
        cm = confusion_matrix(self._y_test, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        if msg:
            print(f"Accuracy with {model_name} model : {accuracy}")
            print(f"Accuracies for each class : (mean = {np.mean(cm.diagonal())})")
            print(cm.diagonal())
        return accuracy

    def test_model_bonus(self, model, model_name="", msg=True):
        y_pred = model.predict(self._X_test)

        df_pred = pd.DataFrame(y_pred, columns=["y_pred"])
        df_pred["account"] = self._accounts_test
        df_pred = df_pred.groupby(["account", "y_pred"]).size().unstack(fill_value=0)
        df_pred["elected_cat"] = df_pred[
            ["HashtagGamer", "LeftTroll", "NewsFeed", "RightTroll"]
        ].idxmax(axis=1)

        df_test = self._y_test.to_frame()
        df_test["account"] = self._accounts_test
        df_test = df_test.drop_duplicates(subset=["account"]).sort_values("account")

        accuracy = accuracy_score(df_test["account_category"], df_pred["elected_cat"])
        cm = confusion_matrix(df_test["account_category"], df_pred["elected_cat"])
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        if msg:
            print(f"Accuracy with {model_name} model : {accuracy}")
            print(f"Accuracies for each class : (mean = {np.mean(cm.diagonal())})")
            print(cm.diagonal())
        return accuracy

    """
    /////////////////// TRAIN AND TEST ///////////////////
    """
    # //////////////////// k plus proche voisins
    def train_KNN_model(self, n_neighbors):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self._X_train, self._y_train)
        return model

    # /////////////////// quadratic

    def train_quadratic_model(self):
        model = QuadraticDiscriminantAnalysis()
        model.fit(self._X_train, self._y_train)
        return model

    # /////////////////// decision tree

    def train_decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self._X_train, self._y_train)
        return model

    def train_decision_tree_bagging(self, n_estimators=50, max_leaf_nodes=50):
        base_estimator = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)

        model = BaggingClassifier(
            base_estimator=base_estimator, n_estimators=n_estimators
        )
        model.fit(self._X_train, self._y_train)
        return model

    def train_random_forest(self, n_estimators=50, max_leaf_nodes=50):
        model = clf = RandomForestClassifier(
            n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes
        )
        model.fit(self._X_train, self._y_train)
        return model


def script_KNN(data_cat="tweets"):
    mds = Modeles(data_cat)
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        # "score_hashtag_HashtagGamer",
        # "score_hashtag_LeftTroll",
        # "score_hashtag_NewsFeed",
        # "score_hashtag_RightTroll",
    ]

    # sans ACP
    mds.prepare_dataset(variables)
    knn = mds.train_KNN_model(172)
    mds.test_model(knn, "knn")

    # avec ACP
    # mds.prepare_dataset(variables, acp=True, n_dim=4)
    # knn = mds.train_KNN_model(23)
    # mds.test_model(knn, "knn with ACP")


def script_quadratic(data_cat="tweets"):
    mds = Modeles(data_cat)
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        # "score_hashtag_HashtagGamer",
        # "score_hashtag_LeftTroll",
        # "score_hashtag_NewsFeed",
        # "score_hashtag_RightTroll",
    ]

    # sans ACP
    mds.prepare_dataset(variables)
    quadratic_model = mds.train_quadratic_model()
    mds.test_model(quadratic_model, "quadratic")

    # avec ACP
    # mds.prepare_dataset(variables, acp=True, n_dim=4)
    # quadratic_model = mds.train_quadratic_model()
    # mds.test_model(quadratic_model, "quadratic with ACP")


def script_decision_tree():
    mds = Modeles()
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        "score_hashtag_HashtagGamer",
        "score_hashtag_LeftTroll",
        "score_hashtag_NewsFeed",
        "score_hashtag_RightTroll",
    ]

    # sans ACP
    # mds.prepare_dataset(variables)

    # decision tree
    # model = mds.train_decision_tree()
    # mds.test_model(model, "dec. tree")

    # bagging
    # model = mds.train_decision_tree_bagging()
    # mds.test_model(model, "bagg. dec. tree")

    # random forest
    # model = mds.train_random_forest()
    # mds.test_model(model, "rd forest")

    # avec ACP
    # mds.prepare_dataset(variables, acp=True)

    # decision tree
    # model = mds.train_decision_tree()
    # mds.test_model(model, "dec. tree with ACP")

    variables = combinations(variables, 2)

    for combination in variables:
        mds.prepare_dataset(list(combination))
        model = mds.train_random_forest(n_estimators=15)
        mds.test_model(model, f"{combination} rd forest")


def partiesliste(seq):
    p = []
    i, imax = 0, 2 ** len(seq) - 1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq) - 1
        while j <= jmax:
            if (i >> j) & 1 == 1:
                s.append(seq[j])
            j += 1
        p.append(s)
        i += 1
    return p


def script_variables_test():
    """
    teste toutes les combinaisons de scores
    """
    mds = Modeles()
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        "score_hashtag_HashtagGamer",
        "score_hashtag_LeftTroll",
        "score_hashtag_NewsFeed",
        "score_hashtag_RightTroll",
    ]

    variables = partiesliste(variables)[1:]
    # variables=[var for var in variables if len(var)>1]
    print(variables)

    accuracies = pd.DataFrame(columns=["variables", "accuracy"])

    for combination in variables:
        mds.prepare_dataset(list(combination))
        model = mds.train_random_forest(n_estimators=15)
        accuracy = mds.test_model(model, f"{combination} rd forest")
        accuracies.append(combination, accuracy)
    return accuracies


def script_highest_score():
    mds = Modeles()
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        # "score_hashtag_HashtagGamer",
        # "score_hashtag_LeftTroll",
        # "score_hashtag_NewsFeed",
        # "score_hashtag_RightTroll",
    ]
    mds.prepare_dataset(variables)
    score_max = mds._X_test[variables].idxmax(axis=1).apply(lambda x: x.split("_")[2])
    accuracy = accuracy_score(mds._y_test, score_max)
    cm = confusion_matrix(mds._y_test, score_max)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print(f"Accuracy with highest score model : {accuracy}")
    print(f"Accuracies for each class : (mean = {np.mean(cm.diagonal())})")
    print(cm.diagonal())


def script_quadratic_bonus(data_cat="tweets"):
    mds = Modeles(data_cat)
    variables = [
        "score_word_HashtagGamer",
        "score_word_LeftTroll",
        "score_word_NewsFeed",
        "score_word_RightTroll",
        "score_hashtag_HashtagGamer",
        "score_hashtag_LeftTroll",
        "score_hashtag_NewsFeed",
        "score_hashtag_RightTroll",
        'nb_liens'
    ]

    # sans ACP
    mds.prepare_dataset_bonus(variables)
    model = mds.train_random_forest(n_estimators=100, max_leaf_nodes=100)
    mds.test_model_bonus(model, "rd forest")


# script_KNN()
# script_quadratic()
# script_highest_score()
# script_highest_score(data_cat="accounts")

#script_quadratic_bonus()
