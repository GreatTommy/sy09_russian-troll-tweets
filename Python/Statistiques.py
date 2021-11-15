import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_PATH = "C:\\Python\\Russian-troll-tweets\\Data\\"
TARGET_PATH=BASE_PATH+"most_common_words_categories\\"

class Statistiques:

    """
    ///////////////////////////// MATRICE DE CORRESPONDANCES /////////////////////////////
    """
    _corr_matrix=None
    _csv_files=None

    def generer_matrice_correspondances(self, csv_files, base_path=TARGET_PATH, debug=False):
        n=len(csv_files)
        corr_matrix = np.zeros((n, n))
        for idxx, filex in enumerate(csv_files):
            if debug:
                print(idxx, filex)
            dfx = pd.read_csv(os.path.join(base_path, f"{filex}"), names=["word", "occurences"], header=None)
            dfx["occurences"] = dfx["occurences"] / dfx["occurences"].sum()
            for idxy, filey in enumerate(csv_files):
                if debug:
                    print("   ", idxy, filey)
                dfy = pd.read_csv(os.path.join(base_path, f"{filey}"), names=["word", "occurences"], header=None)
                dfy["occurences"] = dfy["occurences"] / dfy["occurences"].sum()
                val_sum = 0
                for index, row in dfx.iterrows():
                    val1 = row["occurences"]
                    val2 = dfy.loc[dfy['word'] == row["word"]]["occurences"].values
                    if val2.size != 0:
                        val_sum += (val1 + val2) / 2
                if debug:
                    print(f"{val_sum}")
                corr_matrix[idxx][idxy] = val_sum

        self._corr_matrix=corr_matrix
        self._csv_files=csv_files
        if debug:
            print(corr_matrix)
        return corr_matrix

    def print_matrice_correspondances(self, save=True, base_path=TARGET_PATH, file_name="correspondances"):
        """
        Affiche une matrice déjà générée
        """
        if self._csv_files:
            axis_labels = list(map(lambda x: x[:-4], self._csv_files))
            sns.heatmap(self._corr_matrix,
                        xticklabels=axis_labels,
                        yticklabels=axis_labels,
                        vmin=0, vmax=1,
                        square=True,
                        )
        # SAVE
        if save:
            plt.savefig(os.path.join(base_path, file_name+".svg"))
            plt.savefig(os.path.join(base_path, file_name+".png"))
        self.plt_config_1()
        plt.show()

    """
    ///////////////////////////// PLT /////////////////////////////
    """
    _current_title=None

    def plt_config_1(self, title=None):
        plt.xticks(rotation=45)
        if title:
            plt.title(title)
        else:
            plt.title(self._current_title)

    """
    ///////////////////////////// SCRIPTS /////////////////////////////
    """
    def script_generer_matrice_correspondances(self, base_path=TARGET_PATH):
        #csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

        csv_files=['HashtagGamer.csv', 'LeftTroll.csv', 'NewsFeed.csv', 'RightTroll.csv']
        return self.generer_matrice_correspondances(csv_files, base_path=base_path)

    def script_print_matrice_correspondances(self):
        self.script_generer_matrice_correspondances()
        self.print_matrice_correspondances()

    def script_print_words_correspondances(self):
        self.script_generer_matrice_correspondances(base_path=BASE_PATH+"most_common_words_categories\\")
        self._current_title="Correlation des mots"
        self.print_matrice_correspondances(base_path=BASE_PATH+"most_common_words_categories\\",file_name="words_correspondances")

    def script_print_hashtags_correspondances(self):
        self.script_generer_matrice_correspondances(base_path=BASE_PATH+"most_common_hashtags_categories\\")
        self._current_title="Correlation des hashtags"
        self.print_matrice_correspondances(base_path=BASE_PATH+"most_common_hashtags_categories\\", file_name="hashtags_correspondances")

    def script_correlations(self):
        self.script_print_words_correspondances()
        self.script_print_hashtags_correspondances()

    def script_correlations_joined(self, save=True, base_path="C:\\Python\\Russian-troll-tweets\\figures\\", file_name="Corr_concatenate"):
        m1=self.script_generer_matrice_correspondances(base_path=BASE_PATH+"most_common_hashtags_categories\\")
        m2=self.script_generer_matrice_correspondances(base_path=BASE_PATH+"most_common_words_categories\\")
        m=np.concatenate((m1,m2),axis=1)

        if self._csv_files:
            axis_labels1 = list(map(lambda x: "#"+x[:-4], self._csv_files))
            axis_labels2 = list(map(lambda x: x[:-4], self._csv_files))
            axis_labelsy = list(map(lambda x: "(#)"+x[:-4], self._csv_files))
            axis_labels=np.concatenate((axis_labels1,axis_labels2))
            sns.heatmap(m,
                        xticklabels=axis_labels,
                        yticklabels=axis_labelsy,
                        vmin=0, vmax=1,
                        square=True,
                        )

        # SAVE
        if save:
            #plt.savefig(os.path.join(base_path, file_name+".svg"))
            plt.savefig(os.path.join(base_path, file_name+".png"))
        self._current_title="Correspondances des hashtags/mots"
        self.plt_config_1()
        plt.show()



    """
    ////////////////////////////// MATRCIE DE CORRELATION ////////////////////////
    method= pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            spearman : Spearman rank correlation
            callable
    """
    _correlation=None

    def script_print_correlation_scores(self, nrows=None):
        data=pd.read_csv(
            os.path.join("C:\\Python\\Russian-troll-tweets", "Data\data_with_scores\\train_csv.csv")
        )
        data=data[['score_hashtag_HashtagGamer',
                    'score_hashtag_NewsFeed',
                    'score_word_RightTroll',
                    'score_hashtag_LeftTroll',
                    'score_word_HashtagGamer',
                    'score_word_NewsFeed',
                    'score_word_LeftTroll',
                    'score_hashtag_RightTroll']]
        print(data.shape)
        labels=['# HG',
                '# NF',
                '# R',
                '# L',
                'w HG',
                'w NF',
                'w R',
                'w L',]

        self._correlation = np.abs(data.corr(method='Kendall'))
        sns.heatmap(self._correlation,
                    xticklabels=labels,
                    yticklabels=labels,
                    vmin=0, vmax=1,
                    square=True)
        plt.show()


    def ponderate_scores_correlation(self, nrows=None):
        """
        On pondère les scores grâce à la matrice de correlation
        pour accentuer l'importance des scores très correlés
        aux autres.
        """


        data=pd.read_csv(os.path.join(BASE_PATH, "data_with_scores/data_with_scores.csv"), nrows=nrows)
        data=data.drop(data.columns[0], axis=1)

        corr_df = data.corr(method='pearson')

        pond=corr_df.apply(lambda x: np.abs(x).sum()-1, axis=0)
        pond/pond.max()

        return data.apply(lambda x: x*pond[x.index], axis=1)























