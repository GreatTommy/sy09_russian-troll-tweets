import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
import typing
import array

from Util import Progress

base_path = "C:\\Python\\Russian-troll-tweets"

class Pretraitements:
    """
    Regroupe les méthodes d'enrichissement de données.
    Rattaché à un DataFrame initial
    """
    _data=None
    def set_data(self, data):
        self._data=data

    _most_common_words=None
    _words_to_ban = None

    def __init__(self,data=None):
        self._data=data
        self._most_common_words=self._read_most_common_words(os.path.join(base_path, "Data\most_common_words.txt"), 100)
        self._words_to_ban = ['http', 'https']

    def IsStringOrBytesLike(self,obj):
        return isinstance(obj, str) or isinstance(obj, typing.ByteString) or isinstance(obj, array.array)

    _original_data=None
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
                                                    #'following',
                                                    #'followers',
                                                    #'updates',
                                                    'post_type',    # à modifier
                                                    'account_type',# as coded by Linvill and Warren
                                                    #'retweet',
                                                    'account_category',# as coded by Linvill and Warren
                                                    #'new_june_2018',
                                                    #'alt_external_id',
                                                    #'tweet_id',# from article_url
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


    """
    ///////////////////////////// HASHTAGS /////////////////////////////
    """
    _EXP_REG_HASHTAG="\B#\w*\S+\w"

    def _find_hashtags(self,text):
        """
        @return list of # from the string.

            Expression régulière du hashtag:
                "\S*#(?:\[[^\]]+\]|\S+)"
                "#\w"
                "\B#\w*\S+\w"
        """
        if not self.IsStringOrBytesLike(text): #string or bytes-like object
            return None
        lst=re.findall(self._EXP_REG_HASHTAG,text)
        if lst:
            return lst
        return None

    def get_hashtags(self,data=None, columns=["content"]):
        """
        @return pd.Series:
            - index: index d'origine
            - value: liste des # du tweet
        """
        if type(data)==pd.DataFrame:
            df=data[columns].astype(str)
        else:
            df=self._data[columns].astype(str)

        return df.apply(lambda x: self._find_hashtags(str(x)), axis=1).dropna()

    def count_hashtags(self, hashtags, n=100):
        """
        @param hashtags: DataFrame
        @param n: renvoie les n # les plus présents.
        @return pd.Series:
            - index: string du #
            - value: nombre d'apparitions du #
        """
        return hashtags.explode('content').apply(lambda x: x.lower()).value_counts().sort_values(ascending = False)[:n]

    def create_most_freq_hashtags(self,data=None, columns=["content"], n=100, debug=True, csv=True, csv_name=None, add_directory="Data\\most_common_hashtags"):
        if type(data)!=pd.DataFrame:
            data=self._data

        hashtags=self.get_hashtags(data=data,columns=columns)
        hashtags_count=self.count_hashtags(hashtags, n=n)

        if csv_name:
            mfw_base_path=os.path.join(base_path, add_directory)
            print(os.path.join(mfw_base_path, csv_name), flush=True)
            hashtags_count.to_csv(os.path.join(mfw_base_path, csv_name),index=True, header=False)
        else:
            if debug:
                print(hashtags_count.to_csv(index=True, header=False), flush=True)

        return hashtags_count

    def print_most_freq_hashtags(self, data=None, columns=["content"], n=50, q=0.75):
        """
        @param n: parmis les n # les plus présents.
        @param q: les # affichés sont ceux suffisamment présents, selon le quantile du dénombrement des #.
        @return pd.Series: série des # séléctionnés pour l'affichage.
        """
        if data==None:
            data=self._data
        htgs=self.get_hashtags(data=data,columns=columns)
        htgs_count=self.count_hashtags(htgs, n=n)

        print(htgs_count.to_csv(index=True))
        selection=htgs_count
        # seuil=htgs_count.drop_duplicates().quantile(q)
        # selection=htgs_count[htgs_count >= seuil]

        sns.scatterplot(data=selection,x=selection.index, y=selection.values)
        self.plt_config_1()
        plt.show()

    """
    /////////////// SCORES CALCUL ///////////////
    """

    def calculate_score_hashtags(self, hashtags_with_occurences, tweet_hashtags, debug=False):
        if tweet_hashtags==None:
            return 0

        # intersection (join outer left)
        tweet_df=pd.DataFrame(tweet_hashtags,columns=['hashtag'])
        hashtags_with_occurences_light=pd.merge(hashtags_with_occurences, tweet_df, how='inner', on='hashtag')
        score=hashtags_with_occurences_light["occurences"].sum()
        if debug:
            print("score = ",score)
        return score

    """
    ///////////////////////////// COMMON WORDS /////////////////////////////
    """
    def _read_most_common_words(self, path, length):
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content[:length]


    def _clean_words(self, mot):
        """
        @return:
            - None si le mot est trop petits (<4), courants ou à banir
            - le mot en minuscule sinon.
        """
        mot = mot.lower()
        if len(mot) < 4:
            return None
        if mot in self._most_common_words or mot in self._words_to_ban:
            return None
        return mot

    def _find_and_clean_words(self, text, keep_hashtags=False):
        """
        @return la liste des mots nettoyés d'un texte.
        @param keep_hashtags:
            - True: considère les hashtags comme des mots (les garde)
            - False: retire les hashtags.
        """
        if not self.IsStringOrBytesLike(text): #string or bytes-like object
            return None
        if keep_hashtags==False:
            text=re.sub(self._EXP_REG_HASHTAG,'',text)
        lst=re.findall("\w+",text)
        if lst:
            return list(filter(None,list(map(self._clean_words, lst))))
        return None

    def get_words(self,data=None, columns=["content"]):
        """
        @return pd.Series:
            - index: index d'origine
            - value: liste des mots du tweet
        """
        if type(data)==pd.DataFrame:
            df=data[columns] #.astype(str)
        else:
            df=self._data[columns] #.astype(str)
        return df.apply(lambda x: self._find_and_clean_words(str(x.values)), axis=1).dropna()

    def count_words(self, words, n=50):
        """
        @param n: renvoie les n mots les plus présents.
        @return pd.Series:
            - index: string du mot
            - value: nombre d'apparitions du mot
        """
        return words.explode('content').value_counts().sort_values(ascending = False)[:n]

    def create_most_freq_words(self,data=None, columns=["content"], n=100, debug=True, csv=True, csv_name=None, add_directory="Data\most_common_words_categories"):
        if type(data)!=pd.DataFrame:
            data=self._data

        words=self.get_words(data=data,columns=columns)
        words_count=self.count_words(words, n=n)

        if csv_name:
            mfw_base_path=os.path.join(base_path, add_directory)
            print(os.path.join(mfw_base_path, csv_name), flush=True)
            words_count.to_csv(os.path.join(mfw_base_path, csv_name),index=True, header=False)
        else:
            if debug:
                print(words_count.to_csv(index=True, header=False), flush=True)

        return words_count

    def print_most_freq_words(self,data=None, columns=["content"], n=15):
        if data==None:
            data=self._data

        words=self.get_words(data=data,columns=columns)
        words_count=self.count_words(words, n=n)

        print(words_count.to_csv(index=True, header=False))

        sns.scatterplot(data=words_count,x=words_count.index, y=words_count.values)

        self.plt_config_1()
        plt.show()


    """
    /////////////// SCORES CALCUL ///////////////
    """

    def calculate_score_words(self, words_with_occurences, tweet_words, debug=False):
        if tweet_words==None:
            return 0

        # réduction de la liste des mots words_with_occurences
        tweet_df=pd.DataFrame(tweet_words,columns=['word'])
        words_with_occurences_light=pd.merge(words_with_occurences,tweet_df, how='inner',on='word')
        score=words_with_occurences_light["occurences"].sum()
        if debug:
            print("score = ",score)
        return score

    """
    ///////////////////////////// PLT /////////////////////////////
    """
    _current_title=None

    def plt_config_1(self, title=None):
        plt.xticks(rotation=70)
        if title:
            plt.title(title)
        else:
            plt.title(self._current_title)


    """
    ///////////////////////////// SCRIPTS /////////////////////////////
    """


    def variable_first_selection(self):
        original_data=pd.read_csv(os.path.join(base_path, "Data\combined_csv.csv"), low_memory=False, nrows=None)

        data=original_data[[#'external_author_id',
                            #'author',
                            'content',
                            #'region',       # y'a presque que USA
                            #'language',     #on prend que English
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

        data=data[original_data['language']=='English']#.reset_index(inplace=True, drop=True)
        return data

    def script_print_most_freq_hashtags(self, nrows=None):
        self._data=self.get_original_data(nrows=nrows)
        self._current_title="Hashtags les plus fréquents dans les tweets en anglais"
        self.print_most_freq_hashtags()

    def script_print_most_freq_words(self, nrows=None):
        self._data=self.get_original_data(nrows=nrows)
        self._current_title="Mots les plus fréquents dans les tweets en anglais"
        self.print_most_freq_words()

    def script_create_most_freq_hashtags_categories_csv(self, new_dir="most_common_hashtags_categories", n=100, debug=True, nrows=None):
        original_data=self.get_original_data(nrows=nrows)

        account_category=['HashtagGamer', 'LeftTroll', 'NewsFeed', 'RightTroll']

        if debug:
            progress=Progress(begin=0, end=len(account_category), name="#")
            for category in account_category:
                self._data=original_data[original_data['account_category']==category]
                self.create_most_freq_hashtags(csv_name=category+".csv", add_directory="Data\\most_common_hashtags_categories", n=n)
                progress.print_incremental_progress_percent()
            progress=None
        else:
            for category in account_category:
                self._data=original_data[original_data['account_category']==category]
                self.create_most_freq_hashtags(csv_name=category+".csv", add_directory="Data\\most_common_hashtags_categories", n=n)

    def script_create_most_freq_words_categories_csv(self, new_dir="most_common_words_categories", n=100, debug=True, nrows=None):
        original_data=self.get_original_data(nrows=nrows)

        account_category=['HashtagGamer', 'LeftTroll', 'NewsFeed', 'RightTroll']


        if debug:
            progress=Progress(begin=0, end=len(account_category), name="w")
            for category in account_category:
                self._data=original_data[original_data['account_category']==category]
                self.create_most_freq_words(csv_name=category+".csv", add_directory="Data\\most_common_words_categories", n=n)
                progress.print_incremental_progress_percent()
            progress=None
        else:
            for category in account_category:
                self._data=original_data[original_data['account_category']==category]
                self.create_most_freq_words(csv_name=category+".csv", add_directory="Data\\most_common_words_categories", n=n)

    """
    /////////////// VARIABLES SCORES ///////////////
    """
    def script_score(self, debug=True, nrows=None):
        data=self.get_original_data(nrows=nrows)
        csv_files=['HashtagGamer.csv', 'LeftTroll.csv', 'NewsFeed.csv', 'RightTroll.csv']

        # SCORE WORDS

        for idx, file in enumerate(csv_files):
            print(idx, file)
            df = pd.read_csv(os.path.join(base_path, f"Data\most_common_words_categories\{file}"), names=["word", "occurences"], header=None)
            df["occurences"] = df["occurences"] / df["occurences"].sum()

            if debug:
                progress=Progress(begin=0,end=data.shape[0])
                data[f"score_word_{file[:-4]}"] = data.apply(lambda row : self.calculate_score_words(df, self._find_and_clean_words(progress.print_incremental_progress_percent_ghost(row["content"]))), axis = 1)
                progress=None
            else:
                data[f"score_word_{file[:-4]}"] = data.apply(lambda row : self.calculate_score_words(df, self._find_and_clean_words(row["content"])), axis = 1)
        self._data=data
        data.to_csv("data_with_score_words.csv",index=True)

        #SCORE HASHTAGS

        for idx, file in enumerate(csv_files):
            print(idx, file)
            df = pd.read_csv(os.path.join(base_path, f"Data\most_common_hashtags_categories\{file}"), names=["hashtag", "occurences"], header=None)
            df["occurences"] = df["occurences"] / df["occurences"].sum()

            if debug:
                progress=Progress(begin=0,end=data.shape[0])
                data[f"score_hashtag_{file[:-4]}"] = data.apply(lambda row : self.calculate_score_hashtags(df, self._find_hashtags(progress.print_incremental_progress_percent_ghost(row["content"]))), axis = 1)
                progress=None
            else:
                data[f"score_hashtag_{file[:-4]}"] = data.apply(lambda row : self.calculate_score_hashtags(df, self._find_and_clean_words(row["content"])), axis = 1)
        self._data=data
        data.to_csv("data_with_scores.csv",index=True)

    def script_combine_scores_csv(self, new_file_name="data_with_scores.csv", print_progress=True):
        csv_files=['HashtagGamer_scores.csv', 'LeftTroll_scores.csv', 'NewsFeed_scores.csv', 'RightTroll_scores.csv']

        path=os.path.join(base_path, "Data\data_with_scores")

        data=None
        progress=Progress(begin=0,end=len(csv_files), name="concat", i=1)
        for file in csv_files:
            df=pd.read_csv(os.path.join(path, file), low_memory=False)
            df=df[["score_word_"+file[:-11], "score_hashtag_"+file[:-11]]]
            if type(data)==pd.DataFrame:
                data=pd.concat([data, df], axis=1)
            else:
                data=df
            progress.print_incremental_progress_percent()
        progress=None
        data.to_csv(os.path.join(path, new_file_name), index=True, header=True)
        return data





#%%

def __main__(): # Ca tourne !!!
    pt=Pretraitements()
    #pt.script_create_most_freq_hashtags_categories_csv(n=100)
    #pt.script_create_most_freq_words_categories_csv(n=100)
    pt.script_score()





































