import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import norm
import matplotlib as mpl
from sklearn.base import BaseEstimator
import scipy.linalg as linalg

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
from sklearn.model_selection import GridSearchCV


from nearest_prototypes import NearestPrototypes


#base_path = "sy09_russian-troll-tweets"
base_path = "D:\\Users\\Merwan_Bouvier\\Desktop\\Merwan\\SY09\\Projet\\russian-troll-tweets-master"
#base_path = "C:\\Python\\Russian-troll-tweets"


class AnalyseKNN:
    
    """
    ///////////////////////////////////// DATAS /////////////////////////////////////
    """

    _original_data=None
    _data_scores=None
    
    def get_original_data(self, nrows=None, nrows_readen=None, renew=False):
        if nrows_readen and (nrows_readen<nrows):
            print("nrows_readen<nrows. Illegal: -->nrows_readen=None")
            nrows_readen=None

        if type(self._original_data)!=pd.DataFrame or renew or (not nrows) or self._original_data.shape[0]<nrows:
            self._original_data=pd.read_csv(os.path.join(base_path, "Data/combined_csv.csv"), low_memory=False, nrows=nrows_readen)
            self._original_data=self._original_data[[#'external_author_id',
                                                    #'author',
                                                    'content',
                                                    #'region',       # y'a presque que USA
                                                    'language',     #on prend que English
                                                    #'publish_date',
                                                    #'harvested_date',
                                                    'following',
                                                    'followers',
                                                    'updates',
                                                    'post_type',    # à modifier
                                                    'account_type',# as coded by Linvill and Warren
                                                    'retweet',
                                                    'account_category',# as coded by Linvill and Warren
                                                    #'new_june_2018',
                                                    #'alt_external_id',
                                                    'tweet_id',# from article_url
                                                    #'article_url',
                                                    #'tco1_step1',
                                                    #'tco2_step1',
                                                    #'tco3_step1'
                                                    ]]
            self._original_data=self._original_data[self._original_data['language']=='English']
            if nrows and (self._original_data.shape[0]<nrows):
                if nrows_readen:
                    print("nrows_readen too little. Illegal: -->nrows_readen=None")
                    return self.get_original_data(nrows=nrows, nrows_readen=None, renew=renew)
                else:
                    print("nrows_readen too big. Illegal: -->nrows=None")
            else:
                return self._original_data[self._original_data['language']=='English'][:nrows]
        return self._original_data[:nrows]

    def get_data_scores(self, nrows=None, renew=False):
        if type(self._data_scores)!=pd.DataFrame or renew or (not nrows) or self._data_scores.shape[0]<nrows:
            self._data_scores=pd.read_csv(os.path.join(base_path, "Data/data_with_scores/data_with_scores.csv"), low_memory=False, nrows=nrows)
            self._data_scores.drop(columns=self._data_scores.columns[0], axis=1, inplace=True)
            return self._data_scores
        return self._data_scores[:nrows]

    
    
    """
    ///////////////////////////////////// K-PLUS-PROCHE-VOISINS /////////////////////////////////////
    """
    def process_KNN_return_accuracy(self, X_train, y_train, X_val, y_val, n_neighbors):
        """
        Précision d'un modèle Knn pour un jeu de données
        d'apprentissage et de validation fournis.
        """
    
        # Définition, apprentissage et prédiction 
        cls = KNeighborsClassifier(n_neighbors=n_neighbors)
        cls.fit(X_train, y_train)
        pred = cls.predict(X_val)
    
        # Calcul de la précision avec `accuracy_score`
        acc = accuracy_score(pred, y_val)
    
        return acc
    
    def knn_multiple_validation(self,X, y, n_splits, train_size, n_neighbors_list, debug=True):
        """
        Génère les couples nombre de voisins et précisions correspondantes.
        """
    
        # Conversion en tableau numpy si on fournit des DataFrame par exemple
        X, y = check_X_y(X, y)
    
        def models_accuracies(train_index, val_index, n_neighbors_list, debug=debug):
            """Précision de tous les modèles pour un jeu de données fixé."""
    
            # Création de `X_train`, `y_train`, `X_val` et `y_val`
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_val = X[val_index, :]
            y_val = y[val_index]
    
            # Calcul des précisions pour chaque nombre de voisins présent
            # dans `n_neighbors`
            n = len(train_index)
            for n_neighbors in n_neighbors_list:
                accuracy = self.process_KNN_return_accuracy(X_train, y_train, X_val, y_val, n_neighbors)
                if debug:
                    print(n_neighbors,accuracy)
                yield (
                    n_neighbors,
                    accuracy,
                    n / n_neighbors
                )
    
        # Définition de `n_splits` jeu de données avec `ShuffleSplit`
        ms = ShuffleSplit(n_splits=n_splits, train_size=train_size).split(X)
    
        # Calcul et retour des précisions avec `models_accuracies` pour
        # chaque jeu de données défini par `ShuffleSplit`.
        for train_index, test_index in ms:
            yield from models_accuracies(train_index, test_index, n_neighbors_list)
    
    
    
    
    
    
    """
    ///////////////////////////////////// AFFICHAGES
    """
    def add_decision_boundary(self, model,
                                resolution=100,
                                ax=None,
                                levels=None,
                                label=None,
                                color=None,
                                region=True,
                                model_classes=None):
        """Trace une frontière et des régions de décision sur une figure existante.
    
        :param model: Un modèle scikit-learn ou une fonction `predict`
        :param resolution: La discrétisation en nombre de points par abcisses/ordonnées à utiliser
        :param ax: Les axes sur lesquels dessiner
        :param label: Le nom de la frontière dans la légende
        :param color: La couleur de la frontière
        :param region: Colorer les régions ou pas
        :param model_classes: Les étiquettes des classes dans le cas où `model` est une fonction
    
        """
    
        # Set axes
        if ax is None:
            ax = plt.gca()
    
        # Add decision boundary to legend
        color = "red" if color is None else color
        sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")
    
        # Create grid to evaluate model
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], resolution)
        yy = np.linspace(ylim[0], ylim[1], resolution)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
        def draw_boundaries(XX, YY, Z_num, color):
            # Boundaries
            mask = np.zeros_like(Z_num, dtype=bool)
            for k in range(len(model_classes) - 1):
                mask |= Z_num == k - 1
                Z_num_mask = np.ma.array(Z_num, mask=mask)
                ax.contour(
                    XX,
                    YY,
                    Z_num_mask,
                    levels=[k + 0.5],
                    linestyles="dashed",
                    corner_mask=True,
                    colors=[color],
                    antialiased=True,
                )
    
        def get_regions(predict_fun, xy, shape, model_classes):
            Z_pred = predict_fun(xy).reshape(shape)
            cat2num = {cat: num for num, cat in enumerate(model_classes)}
            num2cat = {num: cat for num, cat in enumerate(model_classes)}
            vcat2num = np.vectorize(lambda x: cat2num[x])
            Z_num = vcat2num(Z_pred)
            return Z_num, num2cat
    
        def draw_regions(ax, model_classes, num2cat, Z_num):
            # Hack to get colors
            # TODO use legend_out = True
            slabels = [str(l) for l in model_classes]
            hdls, hlabels = ax.get_legend_handles_labels()
            hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}
    
            color_dict = {}
            for label in model_classes:
                if str(label) in hlabels_hdls:
                    hdl = hlabels_hdls[str(label)]
                    color = hdl.get_facecolor().ravel()
                    color_dict[label] = color
                else:
                    raise Exception("No corresponding label found for ", label)
    
            colors = [color_dict[num2cat[i]] for i in range(len(model_classes))]
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
    
        if isinstance(model, BaseEstimator):
            if model_classes is None:
                model_classes = model.classes_
    
            if levels is not None:
                if len(model.classes_) != 2:
                    raise Exception("Lignes de niveaux supportées avec seulement deux classes")
    
                # Scikit-learn model, 2 classes + levels
                Z = model.predict_proba(xy)[:, 0].reshape(XX.shape)
                Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)
    
                # Only 2 classes, simple contour
                ax.contour(
                    XX,
                    YY,
                    Z,
                    levels=levels,
                    colors=[color]
                )
    
                draw_regions(ax, model_classes, num2cat, Z_num)
            else:
                # Scikit-learn model + no levels
                Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)
    
                draw_boundaries(XX, YY, Z_num, color)
                if region:
                    draw_regions(ax, model_classes, num2cat, Z_num)
        else:
            if model_classes is None:
                raise Exception("Il faut spécifier le nom des classes")
            if levels is not None:
                raise Exception("Lignes de niveaux avec fonction non supporté")
    
            # Model is a predict function, no levels
            Z_num, num2cat = get_regions(model, xy, XX.shape, model_classes)
            draw_boundaries(XX, YY, Z_num, color)
            if region:
                draw_regions(ax, model_classes, num2cat, Z_num)

                
    def scatterplot_pca(self, columns=None, hue=None, style=None, data=None, acp_abs=1, acp_ord=2, **kwargs):
        """
        Utilise `sns.scatterplot` en appliquant d'abord une ACP si besoin pour réduire la dimension.
        
        @param acp_abs: numéro composante principale en abscisse
        @param acp_ord: numéro composante principale en ordinnee
        
        @return ax, pca: Axis and ACP
        
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
            n_components = max(acp_abs, acp_ord)
            pca = PCA(n_components=n_components)
            data_pca = pca.fit_transform(data_quant)
            data_pca = pd.DataFrame(
                data_pca[:, [acp_abs - 1, acp_ord - 1]], columns=[f"PC{acp_abs}", f"PC{acp_ord}"]
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
    
    def plot_clustering(self, data, clus1, clus2=None, ax=None, **kwargs):
        """Affiche les données `data` dans le premier plan principal.
        """
    
        if ax is None:
            ax = plt.gca()
    
        other_kwargs = {e: kwargs.pop(e) for e in ["centers", "covars"] if e in kwargs}
    
        ax, pca = self.scatterplot_pca(data=data, hue=clus1, style=clus2, ax=ax, **kwargs)
    
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
    
    

    """
    ///////////////////////////////////// SCRIPTS
    """     
    
    def script_knn_multiple_validation_with_ACP(self, nrows=None, train_size = 0.65, n_splits = 10, n_neighbors_list=None, n_components=3, debug=True):
        """
        @param nrows
        @param train_size: part de données d'entrainement
        @param n_splits: séparation des données d'entrainement
        @param n_neighbors_list: liste nombres de voisins à tester
        @param n_components: nombre de composantes pour l'ACP
        """
        
        # DATA
        
        if n_neighbors_list == None:
            n_neighbors_list = np.unique(np.round(np.geomspace(1, 500,100)).astype(int))
        
        if nrows:
            original_data=self.get_original_data(nrows=nrows, nrows_readen=2*nrows)
        else:
            original_data=self.get_original_data()
        data_scores=self.get_data_scores(nrows=nrows)

        entrainement_cat=['HashtagGamer', 'LeftTroll', 'NewsFeed', 'RightTroll']
        entrainement=list(map(lambda x: x in entrainement_cat, original_data["account_category"]))

        data_scores_entrainement=data_scores[entrainement]
        original_data_entrainement=original_data[entrainement]
        assert data_scores_entrainement.shape[0]==original_data_entrainement.shape[0]
        
        data_scores_entrainement=data_scores_entrainement[['score_hashtag_RightTroll',
                                        'score_word_LeftTroll',
                                        'score_word_NewsFeed',
                                        'score_word_RightTroll']]
        
        # ACP
        
        cls=PCA(n_components=n_components)
        proj=cls.fit_transform(data_scores_entrainement)
        dft=pd.DataFrame(proj,columns=[f"CP{i}" for i in range(1,n_components+1)])
        dft["y"] = original_data_entrainement["account_category"]
        
        # knn_multiple_validation
        
        X = dft.iloc[:, :-1]
        y = original_data_entrainement["account_category"]
        if debug:
            print(X)
            print(y.unique())
        
        
        param_grid = {
            "n_neighbors": [5],
            "n_prototypes_list": [[100 + i, 100 + i, 100 - i, 100 - i] for i in range(-2, 2, 2)],
        }
        cls = NearestPrototypes(n_prototypes_list=[100, 100, 100, 100], n_neighbors=5)
        #knp.fit(X, y)
        search = GridSearchCV(cls, param_grid, scoring="accuracy", cv=10)
        search.fit(X, y)
        '''
        ax = sns.scatterplot(x="CP1", y="CP2", hue="y", data=dft, alpha=0.3,legend=False)
        df = pd.DataFrame(
        dict(X1=knp.prototypes_[:, 0], X2=knp.prototypes_[:, 1], y=knp.labels_)
        )
        sns.scatterplot(x="CP1", y="CP2", hue="y", marker="s", data=df)
        self.add_decision_boundary(knp)
        plt.show()
        '''
        if debug:
            print(search.best_params_)
        
        df = pd.DataFrame(
            (   
                dict(n_prototypes0=d["n_prototypes_list"][0], error=e, std=s)
                for d, e, s in zip(
                    search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
                )
            )
        )
        plt.errorbar(df["n_prototypes0"], df["error"], yerr=df["std"])
        plt.show()
        
        '''
        gen = self.knn_multiple_validation(X, y, n_splits, train_size, n_neighbors_list, debug=False)
        if debug:
            print(gen)
        df = pd.DataFrame(gen, columns=["# neighbors", "accuracy", "degrés de liberté"])
        
        # K optimal
        
        Kopt = df.groupby("# neighbors").mean().accuracy.idxmax() 
        
        # plot accuracy(liberté)
        
        sp = sns.lineplot(x="degrés de liberté", y="accuracy", err_style="bars", ci="sd", data=df)
        sp.set(xscale="log")
        plt.show()
        
        # plot optimal clustering with decision boundary
        
        cls = KNeighborsClassifier(n_neighbors=Kopt)
        cls.fit(X, y)
        self.plot_clustering(X, y)
        self.add_decision_boundary(cls)
        plt.show()
        '''
    def script_k_proche_voisins_scores(self, nrows=None, train_size = 0.90, n_splits = 10, n_neighbors_list=None):
        if n_neighbors_list == None:
            n_neighbors_list = np.unique(np.round(np.geomspace(1, 500,100)).astype(int))
        
        if nrows:
            original_data=self.get_original_data(nrows=nrows, nrows_readen=2*nrows)
        else:
            original_data=self.get_original_data()
        data_scores=self.get_data_scores(nrows=nrows)

        entrainement_cat=['HashtagGamer', 'LeftTroll', 'NewsFeed', 'RightTroll']
        entrainement=list(map(lambda x: x in entrainement_cat, original_data["account_category"]))

        data_scores_entrainement=data_scores[entrainement]
        original_data_entrainement=original_data[entrainement]
        assert data_scores_entrainement.shape[0]==original_data_entrainement.shape[0]
        
        X = data_scores_entrainement
        y = original_data_entrainement["account_category"]
        print(X.info())
        print(X)
        print(y.unique())
        gen = self.knn_multiple_validation(X, y, n_splits, train_size, n_neighbors_list)
        print(gen)
        df = pd.DataFrame(gen, columns=["# neighbors", "accuracy", "degrés de liberté"])
        Kopt = df.groupby("# neighbors").mean().accuracy.idxmax()
        print(Kopt) 

        sp = sns.lineplot(x="degrés de liberté", y="accuracy", err_style="bars", ci="sd", data=df)
        sp.set(xscale="log")
        plt.show()
        
    


    
    
    
    
    




def __main__():
    a=AnalyseKNN()
    
    a.script_knn_multiple_validation_with_ACP(nrows=100000)
