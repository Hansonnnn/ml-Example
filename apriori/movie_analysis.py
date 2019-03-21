import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def get_data():
    df = pd.read_csv('/Users/hanzhao/PycharmProjects/ml-example/apriori/data/movies.csv', sep=',')
    return df


'''get_dummies is way to one-hot encode'''


def apriori_movies():
    df = get_data()
    df_oneH = df.drop('genres', 1).join(df.genres.str.get_dummies())
    df_oneH.set_index(['movieId', 'title'], inplace=True)
    frequent_df = apriori(df_oneH, min_support=0.03, use_colnames=True)
    return frequent_df


def association_movies():
    frequent_df = apriori_movies()
    association_df = association_rules(frequent_df, metric='lift', min_threshold=1)
    print(association_df)


association_movies()
