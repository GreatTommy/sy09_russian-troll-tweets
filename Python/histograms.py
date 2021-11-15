import numpy as np
import pandas as pd
import glob
import os
import datetime

path= r"D:\Users\Merwan_Bouvier\Desktop\Merwan\SY09\Projet\russian-troll-tweets-master"
base_path = "C:\\Python\\Russian-troll-tweets\\"
path=base_path

custom_date_parser = lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %H:%M")

all_files = glob.glob(os.path.join(path,"*.csv"))
all_df = []
for f in all_files:
    df = pd.read_csv(f,parse_dates=['publish_date','harvested_date'],  date_parser=custom_date_parser)
    df['file'] = f.split('/')[-1]
    all_df.append(df)

df_merged = pd.concat(all_df, ignore_index=True, sort=True)
df_merged.to_csv( "merged.csv")
'''
tweets1 = pd.read_csv("IRAhandle_tweets_1.csv",parse_dates=['publish_date','harvested_date'],  date_parser=custom_date_parser)
tweets2 = pd.read_csv("IRAhandle_tweets_2.csv",parse_dates=['publish_date','harvested_date'],  date_parser=custom_date_parser)
tweets3 = pd.read_csv("IRAhandle_tweets_3.csv")
tweets4 = pd.read_csv("IRAhandle_tweets_4.csv")
tweets5 = pd.read_csv("IRAhandle_tweets_5.csv") #0,15,20
tweets6 = pd.read_csv("IRAhandle_tweets_6.csv")
tweets7 = pd.read_csv("IRAhandle_tweets_7.csv")
tweets8 = pd.read_csv("IRAhandle_tweets_8.csv")
tweets9 = pd.read_csv("IRAhandle_tweets_9.csv")
tweets10 = pd.read_csv("IRAhandle_tweets_10.csv") #20
tweets11 = pd.read_csv("IRAhandle_tweets_11.csv")
tweets12 = pd.read_csv("IRAhandle_tweets_12.csv") #10,20
tweets13 = pd.read_csv("IRAhandle_tweets_13.csv")


tweets1["date"] = tweets1["publish_date"].dt.date
tweets1["hour"] = tweets1["publish_date"].dt.hour


tweets1[["publish_date","account_category"]].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count().plot(kind="bar")
'''
X = df_merged[["publish_date","account_category"]]

'''
x1 = X[X['account_category'] == "RightTroll"].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count()
x2 = X[X['account_category'] == "LeftTroll"].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count()
x3 = X[X['account_category'] == "NewsFeed"].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count()
x4 = X[X['account_category'] == "HashtagGamer"].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count()
x5 = X[X['account_category'] == "Fearmonger"].groupby([tweets1["publish_date"].dt.year, tweets1["publish_date"].dt.month]).count()
'''

X2012 = X[X['publish_date'].dt.year == 2012]
X2013 = X[X['publish_date'].dt.year == 2013]
X2014 = X[X['publish_date'].dt.year == 2014]
X2015 = X[X['publish_date'].dt.year == 2015]
X2016 = X[X['publish_date'].dt.year == 2016]
X2017 = X[X['publish_date'].dt.year == 2017]
X2018 = X[X['publish_date'].dt.year == 2018]

#2012
x1 = X2012.groupby([X2012["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2012")

#2013
x1 = X2013.groupby([X2013["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2013")

#2014
x1 = X2014.groupby([X2014["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2014")

#2015
x1 = X2015.groupby([X2015["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2015")

#2016
x1 = X2016.groupby([X2016["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2016")

#2017
x1 = X2017.groupby([X2017["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2017")

#2018
x1 = X2018.groupby([X2018["publish_date"].dt.month, 'account_category']).count().unstack()
x1.columns = x1.columns.droplevel()
x2 = x1.plot(kind='bar')
x2.set_xlabel("months of 2018")

#Region par categorie
X6 = df_merged[df_merged['region'] != "United States"]
X6 = X6[["region","account_category"]]
x1 = X6.groupby([X6["region"], 'account_category']).size().unstack()
x2 = x1.plot(kind='bar')
x2.set_xlabel("region per account")

#Region par categorie
X6 = df_merged[["updates","account_category"]]
x1 = X6.groupby([X6["updates"], 'account_category']).size().unstack()
x2 = x1.plot(kind='bar')
x2.set_xlabel("region per account")

#publication en fonction de l'heure
x2 = X.groupby([X["publish_date"].dt.hour, 'account_category']).count().unstack()
x2.columns = x2.columns.droplevel()
x2.plot(kind='bar')

#publication en fonction des mois (endemble des données)
x2 = X.groupby([X["publish_date"].dt.year,X["publish_date"].dt.month, 'account_category']).count().unstack()
x2.columns = x2.columns.droplevel()
x2.plot(kind='bar')

#
X2 = df_merged[["publish_date","account_category","content"]]
X2015 = X2[X2['publish_date'].dt.year == 2015]
X2015_06 = X2015[X2015['publish_date'].dt.month == 6]
X2015_06 = X2015_06[X2015_06['account_category'] == "NonEnglish"]
X2015_06["content"]

'''
#nombre de 3eme lien pour les tweets
X3 = df_merged[df_merged['tco3_step1'].isnull() == False]
X3 = X3[["publish_date","account_category"]]
x1 = X3.groupby([X3["publish_date"].dt.year, 'account_category']).count().unstack()
# Droping unnecessary column level
x1.columns = x1.columns.droplevel()
x1.plot(kind='bar')

#nombre de 2eme lien pour les tweets
X4 = df_merged[df_merged['tco2_step1'].isnull() == False]
X4 = X4[["publish_date","account_category"]]
x2 = X4.groupby([X4["publish_date"].dt.year, 'account_category']).count().unstack()
x2.columns = x2.columns.droplevel()
x2.plot(kind='bar')

#nombre de 1er lien pour les tweets
X5 = df_merged[df_merged['tco1_step1'].isnull() == False]
X5 = X5[["publish_date","account_category"]]
x3 = X5.groupby([X5["publish_date"].dt.year, 'account_category']).count().unstack()
x3.columns = x3.columns.droplevel()
x3.plot(kind='bar')
'''

#Création de ligne link : nombre de liens du tweet
conditions = [
    (df_merged['tco3_step1'].isnull() == False),
    (df_merged['tco3_step1'].isnull()) & (df_merged['tco2_step1'].isnull() == False),
    (df_merged['tco3_step1'].isnull()) & (df_merged['tco2_step1'].isnull()) & (df_merged['tco1_step1'].isnull() == False),
    (df_merged['tco3_step1'].isnull()) & (df_merged['tco2_step1'].isnull()) & (df_merged['tco1_step1'].isnull())
] 
values = [3,2,1,0]
df_merged["link"] = np.select(conditions, values)

#Nombre de liens max par category 
X3 = df_merged[["link","account_category"]]
x3 = X3.groupby(["link", 'account_category']).size().unstack()
x3.plot(kind='bar')

#Type des tweets
conditions = [
    (df_merged['post_type'].isnull()),
    (df_merged['post_type'] == "RETWEET"),
    (df_merged['post_type'] == "QUOTE_TWEET")
] 
values = ["TWEET", "RETWEET", "QUOTE_TWEET"]
df_merged["post_type"] = np.select(conditions, values)

X4 = df_merged[["post_type","account_category"]]
x4 = X4.groupby(["post_type", 'account_category']).size().unstack()
x4.plot(kind='bar')

#Grand nombre de publication de Rigth Troll sur cette période
X5 = df_merged[["publish_date","account_category","content","post_type"]]
X2017 = X5[X5['publish_date'].dt.year == 2017]
X2017_08 = X2017[X2017['publish_date'].dt.month == 8]
X2017_08_right = X2017_08[X2017_08['account_category'] == "RightTroll"]
X2017_08_right.to_csv( "2017_08_rightTroll.csv")


#type des tweets sur la période de forte affluence
X4 = X2017_08_right[["post_type","account_category"]]
x4 = X4.groupby(["post_type", 'account_category']).size().unstack()
x4.plot(kind='bar')

#Contenu des tweets comerciaux
X6 = df_merged[df_merged['account_category'] == "Commercial"]
X6 = X6["content"]
X6.to_csv( "commercial.csv")


df_merged.groupby(['account_category']).agg({'link': ['min', 'max', 'mean', 'count']})



