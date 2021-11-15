import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

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


def scatterplot_pca(
    columns=None, hue=None, style=None, data=None, pc1=1, pc2=2, **kwargs
):
    """
    Utilise `sns.scatterplot` en appliquant d'abord une ACP si besoin
    pour réduire la dimension.
    """

    # Select columns (should be numeric)
    data_quant = data if columns is None else data[columns]
    data_quant = data_quant.drop(
        columns=[e for e in [hue, style] if e is not None], errors="ignore"
    )

    # Reduce to two dimensions
    if data_quant.shape[1] == 2:
        data_pca = data_quant
        pca = None
    else:
        n_components = max(pc1, pc2)
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_quant)
        data_pca = pd.DataFrame(
            data_pca[:, [pc1 - 1, pc2 - 1]], columns=[f"PC{pc1}", f"PC{pc2}"]
        )

    # Keep name, force categorical data for hue and steal index to
    # avoid unwanted alignment
    if isinstance(hue, pd.Series):
        if not hue.name:
            hue.name = "hue"
        hue_name = hue.name
    elif isinstance(hue, str):
        hue_name = hue
        hue = data[hue]
    elif isinstance(hue, np.ndarray):
        hue = pd.Series(hue, name="class")
        hue_name = "class"

    hue = hue.astype("category")
    hue.index = data_pca.index
    hue.name = hue_name

    if isinstance(style, pd.Series):
        if not style.name:
            style.name = "style"
        style_name = style.name
    elif isinstance(style, str):
        style_name = style
        style = data[style]
    elif isinstance(style, np.ndarray):
        style = pd.Series(style, name="style")
        style_name = "style"

    sp_kwargs = {}
    full_data = data_pca
    if hue is not None:
        full_data = pd.concat((full_data, hue), axis=1)
        sp_kwargs["hue"] = hue_name
    if style is not None:
        full_data = pd.concat((full_data, style), axis=1)
        sp_kwargs["style"] = style_name

    x, y = data_pca.columns
    ax = sns.scatterplot(x=x, y=y, data=full_data, **sp_kwargs)

    return ax, pca


def plot_clustering(data, clus1, clus2=None, ax=None, **kwargs):
    """Affiche les données `data` dans le premier plan principal.
    """

    if ax is None:
        ax = plt.gca()

    other_kwargs = {e: kwargs.pop(e) for e in ["centers", "covars"] if e in kwargs}

    ax, pca = scatterplot_pca(data=data, hue=clus1, style=clus2, ax=ax, **kwargs)

    if "centers" in other_kwargs and "covars" in other_kwargs:
        # Hack to get colors
        # TODO use legend_out = True
        levels = [str(l) for l in np.unique(clus1)]
        hdls, labels = ax.get_legend_handles_labels()
        colors = [
            artist.get_facecolor().ravel()
            for artist, label in zip(hdls, labels)
            if label in levels
        ]
        colors = colors[: len(levels)]

        if data.shape[1] == 2:
            centers_2D = other_kwargs["centers"]
            covars_2D = other_kwargs["covars"]
        else:
            centers_2D = pca.transform(other_kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T for c in other_kwargs["covars"]
            ]

        p = 0.9
        sig = norm.ppf(p ** (1 / 2))

        for covar_2D, center_2D, color in zip(covars_2D, centers_2D, colors):
            v, w = linalg.eigh(covar_2D)
            v = 2.0 * sig * np.sqrt(v)

            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])

            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    return ax, pca


def add_decision_boundary(
    model, levels=None, resolution=100, ax=None, label=None, color=None, region=True
):
    """Trace une frontière et des régions de décision sur une figure existante.

    La fonction requiert un modèle scikit-learn `model` pour prédire
    un score ou une classe. La discrétisation utilisée est fixée par
    l'argument `resolution`. Une (ou plusieurs frontières) sont
    ensuite tracées d'après le paramètre `levels` qui fixe la valeur
    des lignes de niveaux recherchées.

    """

    if ax is None:
        ax = plt.gca()

    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    cat2num = {cat: num for num, cat in enumerate(model.classes_)}
    num2cat = {num: cat for num, cat in enumerate(model.classes_)}
    vcat2num = np.vectorize(lambda x: cat2num[x])
    Z_num = vcat2num(Z)

    # Add decision boundary to legend
    color = "red" if color is None else color
    sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")

    mask = np.zeros_like(Z_num, dtype=bool)
    for k in range(len(model.classes_) - 1):
        mask |= Z_num == k - 1
        Z_num_mask = np.ma.array(Z_num, mask=mask)
        ax.contour(
            XX,
            YY,
            Z_num_mask,
            levels=[k + 0.5],
            linestyles="dashed",
            corner_mask=True,
            colors=["red"],
            antialiased=True,
        )

    if region:
        # Hack to get colors
        # TODO use legend_out = True
        slabels = [str(l) for l in model.classes_]
        hdls, hlabels = ax.get_legend_handles_labels()
        hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}

        color_dict = {}
        for label in model.classes_:
            if str(label) in hlabels_hdls:
                hdl = hlabels_hdls[str(label)]
                color = hdl.get_facecolor().ravel()
                color_dict[label] = color
            else:
                raise Exception("No corresponding label found for ", label)

        colors = [color_dict[num2cat[i]] for i in range(len(model.classes_))]
        cmap = mpl.colors.ListedColormap(colors)

        ax.imshow(
            Z_num,
            interpolation="nearest",
            extent=ax.get_xlim() + ax.get_ylim(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.2,
        )

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
    ///////////////////////////// TRAIN /////////////////////////////
    """
    def get_i(self):
        r=random.randint(0,100)
        if r<37:
            return 0
        if r<58:
            return 1
        if r<88:
            return 2
        return 3

    def naif_model(self):
        preds=["RightTroll","LeftTroll","NewsFeed", "HashtagGamer"]
        y_pred = pd.Series([preds[self.get_i()] for i in range(self._X_test.shape[0])], name="account_category")
        print(y_pred)
        print(self._y_test)
        accuracy = accuracy_score(self._y_test, y_pred)
        cm = confusion_matrix(self._y_test, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        print(f"Accuracy with naif model : {accuracy}")
        print(f"Accuracies for each class : (mean = {np.mean(cm.diagonal())})")
        print(cm.diagonal())
        return accuracy

    def convert_category(self,a):
        if a=='score_word_HashtagGamer': return 'HashtagGamer'
        if a=='score_word_LeftTroll': return 'LeftTroll'
        if a=='score_word_NewsFeed': return 'NewsFeed'
        if a=='score_word_RightTroll': return 'RightTroll'
        if a=='score_hashtag_HashtagGamer': return 'HashtagGamer'
        if a=='score_hashtag_LeftTroll': return 'LeftTroll'
        if a=='score_hashtag_NewsFeed': return 'NewsFeed'
        else: return 'RightTroll'

    def naif_model_scores(self):
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

        preds=["RightTroll","LeftTroll","NewsFeed", "HashtagGamer"]
        allnulindex=self._X_test[np.isin(self._X_test.index, np.where(~self._X_test[variables].any(axis=1))[0])].index
        y_pred=self._X_test.idxmax(axis = 1)
        y_pred=y_pred.apply(lambda x: self.convert_category(x))
        for i in allnulindex:
            y_pred.iloc[i]=preds[self.get_i()]

        accuracy = accuracy_score(self._y_test, y_pred)
        cm = confusion_matrix(self._y_test, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        print(f"Accuracy with naif model : {accuracy}")
        print(f"Accuracies for each class : (mean = {np.mean(cm.diagonal())})")
        print(cm.diagonal())
        return accuracy

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
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes
        )
        model.fit(self._X_train, self._y_train)
        print(f'n_estimators={n_estimators} max_leaf_nodes={max_leaf_nodes}')
        return model

    def print_ACP_3d_express(self, data, hue=None, symbol=None, n_components=3):
        """
        Requires plotly.express package
        """
        if n_components<3:
            print(f"n_components {n_components}<3. Illegal: -->n_components=3")
            n_components=3
        import plotly.express as px

        # ACP
        cls=PCA(n_components=n_components)
        proj=cls.fit_transform(data)
        dft=pd.DataFrame(proj,columns=[f"CP{i}" for i in range(1,n_components+1)])

        # titre
        total_var = cls.explained_variance_ratio_.sum() * 100
        titre=f'Total Explained Variance: {total_var:.2f}%\n{cls.explained_variance_ratio_}'

        # affichage
        fig = px.scatter_3d(dft, x="CP1", y="CP2", z="CP3",
                                color=hue,
                                symbol=symbol,
                                size=list(map(lambda x: 20,hue)),
                                opacity=0.8,
                                title=titre)
        fig.show()
        return dft

    def print_3d_express(self, data, hue=None, symbol=None):
        """
        Requires plotly.express package
        """
        import plotly.express as px


        # affichage
        fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2],
                                color=hue,
                                symbol=symbol,
                                size=list(map(lambda x: 20,hue)),
                                opacity=0.8)
        fig.show()




    """
    ///////////////////////////// SCRIPTS /////////////////////////////
    """

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
        #'post_type',
        'nb_liens'
    ]

    # sans ACP
    mds.prepare_dataset(variables, acp=True)

    # decision tree
    # model = mds.train_decision_tree()
    # mds.test_model(model, "dec. tree")

    # bagging
    # model = mds.train_decision_tree_bagging()
    # mds.test_model(model, "bagg. dec. tree")

    # random forest
    model = mds.train_random_forest(n_estimators=100, max_leaf_nodes=100)
    mds.test_model(model, "rd forest")

    return mds, model

    # avec ACP
    # mds.prepare_dataset(variables, acp=True)

    # decision tree
    # model = mds.train_decision_tree()
    # mds.test_model(model, "dec. tree with ACP")

    # variables = combinations(variables, 2)
    #
    # for combination in variables:
    #     mds.prepare_dataset(list(combination))
    #     model = mds.train_random_forest(n_estimators=15)
    #     mds.test_model(model, f"{combination} rd forest")


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
    ]

    # sans ACP
    mds.prepare_dataset_bonus(variables)
    quadratic_model = mds.train_quadratic_model()
    mds.test_model_bonus(quadratic_model, "quadratic")


# script_KNN()
# script_quadratic()
# script_highest_score()
# script_highest_score(data_cat="accounts")

#script_quadratic_bonus()


sns.scatterplot(x=mds._X_train.columns[0], y=mds._X_train.columns[0], hue=mds._y_train.name, data=pd.concat((mds._X_train, mds._y_train),axis=1))
add_decision_boundary(model)
plt.show()
