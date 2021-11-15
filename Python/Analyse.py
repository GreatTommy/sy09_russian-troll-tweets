import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA


#base_path = "sy09_russian-troll-tweets"
base_path = "C:\\Python\\Russian-troll-tweets"



class Analyse:

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
    /////////////// PRINT ACP ///////////////
    """


    def print_ACP_2d(self, data, hue=None, save_name=None, n_components=2):

        cls=PCA(n_components=n_components)
        proj=cls.fit_transform(data)
        dft=pd.DataFrame(proj,columns=[f"CP{i}" for i in range(1,n_components+1)])

        # titre
        total_var = cls.explained_variance_ratio_.sum() * 100
        titre=f'Total Explained Variance: {total_var:.2f}%\n{cls.explained_variance_ratio_}'
        plt.title(titre)

        # affichage
        if type(hue)==pd.Series:
            sns.scatterplot(x="CP1",y="CP2",data=dft,
                            hue=hue,
                            style=hue)
        else:
            sns.scatterplot(x="CP1",y="CP2",data=dft, markers='X')
        plt.show()

        # sauvegarde
        if save_name:
            print(f"figure saved: {os.path.join(base_path, save_name)}")
            plt.savefig(os.path.join(base_path, save_name))

        return dft

    def print_ACP_3d(self, data, hue=None, symbol=None, save_name=None, n_components=3):

        cls=PCA(n_components=n_components)
        proj=cls.fit_transform(data)
        dft=pd.DataFrame(proj,columns=[f"CP{i}" for i in range(1,n_components+1)])

        # titre
        total_var = cls.explained_variance_ratio_.sum() * 100
        titre=f'Total Explained Variance: {total_var:.2f}%\n{cls.explained_variance_ratio_}'
        plt.title(titre)

        # couleur
        color_labels = hue.unique()
        rgb_values = sns.color_palette("Set1", len(color_labels))
        color_map = dict(zip(color_labels, rgb_values))

        # affichage
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.scatter3D(xs=dft["CP1"], ys=dft["CP2"], zs=dft["CP3"],
                    c=hue.map(color_map),
                    s=10,
                    alpha=0.2)

        plt.show()
        return dft

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









    """
    /////////////// SCRIPTS ///////////////
    """

    def script_ACP_2d(self, nrows=None, n_components=2):
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

        self.print_ACP_2d(data_scores_entrainement, hue=original_data_entrainement.account_category, save_name="ACP.png", n_components=n_components)

    def script_ACP_3d(self, nrows=None, express=False, n_components=3):
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

        # data=data_scores_entrainement[['score_hashtag_HashtagGamer',
        #                                 'score_hashtag_LeftTroll',
        #                                 'score_word_HashtagGamer',
        #                                 'score_hashtag_NewsFeed']]
        data=data_scores_entrainement[['score_hashtag_RightTroll',
                                        'score_word_LeftTroll',
                                        'score_word_NewsFeed',
                                        'score_word_RightTroll']]

        if express:
            self.print_ACP_3d_express(data,
                                hue=original_data_entrainement.account_category,
                                symbol=None,
                                n_components=n_components)
        else:
            self.print_ACP_3d(data,
                            hue=original_data_entrainement.account_category,
                            symbol=original_data_entrainement.account_category,
                            save_name="ACP.png")

    def script_test(self, nrows=1000000):
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

        data=data_scores_entrainement[['score_hashtag_HashtagGamer',
                                            'score_hashtag_LeftTroll',
                                            'score_hashtag_RightTroll',
                                            'score_word_NewsFeed']]

        colors=['green','purple','red','blue']
        for i, cat in enumerate(entrainement_cat):
            plt.scatter(data=data[list(map(lambda x: x == cat, original_data_entrainement["account_category"]))],
                x="score_hashtag_LeftTroll",
                y="score_hashtag_RightTroll",
                c=colors[i],
                marker='+',
                alpha=0.9,
                s=1)

        plt.show()


def __main__():
    '''
    Ne pas utiliser les affichages 3d avec tous les points...
    '''
    a=Analyse()

    # exemples:
    a.script_ACP_2d(nrows=None, n_components=2)
    #a.script_ACP_3d(nrows=200000, express=False)
    #a.script_ACP_3d(nrows=200000, express=True, n_components=4)






"""
"""
"""
    TESTS
"""
"""
"""
#%%     Chargement données/selection des colonnes


# a=Analyse()
# nrows=None
#
# if nrows:
#     original_data=a.get_original_data(nrows=nrows, nrows_readen=2*nrows)
# else:
#     original_data=a.get_original_data()
# data_scores=a.get_data_scores(nrows=nrows)
#
# entrainement_cat=['HashtagGamer', 'NewsFeed', 'LeftTroll', 'RightTroll']
# entrainement=list(map(lambda x: x in entrainement_cat, original_data["account_category"]))
#
# data_scores_entrainement=data_scores[entrainement]
# original_data_entrainement=original_data[entrainement]
# assert data_scores_entrainement.shape[0]==original_data_entrainement.shape[0]
#
# data=data_scores_entrainement[['score_hashtag_HashtagGamer',
#                                 'score_hashtag_LeftTroll',
#                                 'score_hashtag_RightTroll',
#                                 'score_word_NewsFeed']]
#
# n_components=3

#%%         Scatterplot ACP

# cls=PCA(n_components=n_components)
# proj=cls.fit_transform(data)
# dft=pd.DataFrame(proj,columns=[f"CP{i}" for i in range(1,n_components+1)])
#
# # titre
# total_var = cls.explained_variance_ratio_.sum() * 100
# titre=f'Total Explained Variance: {total_var:.2f}%\n{cls.explained_variance_ratio_}'
# plt.title(titre)
#
# hue=original_data_entrainement.account_category
#
# # affichage
# if type(hue)==pd.Series:
#     sns.scatterplot(x="CP2",y="CP3",data=dft,
#                     hue=hue)
# else:
#     sns.scatterplot(x="CP2",y="CP3",data=dft, markers='X')
# plt.show()


#%%
#%%
#%%
#%%
#%%
#%%

# data=pd.read_csv(os.path.join(base_path, "Data/combined_csv.csv"), low_memory=False, nrows=None)
#
# #%%
#
# def liste_liens_tweet(row):
#     l=[row.tco1_step1, row.tco2_step1, row.tco3_step1]
#     l=[x for x in l if not pd.isna(x)]
#     return len(l)
#
# liens_G=data.apply(liste_liens_tweet, axis=1)
#
# #%%
#
# lG=pd.DataFrame(np.expand_dims(liens_G,axis=1))


















