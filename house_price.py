import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class HousePrice(object):
    def explore_data(self):
        df_train = pd.read_csv('/Users/hanzhao/PycharmProjects/ml-example/file/data/train.csv')
        # var = 'YearBuilt'
        # data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
        # # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
        # f,ax = plt.subplots(figsize=(16,8))
        # fg = sns.boxplot(x=var,y='SalePrice',data=data)
        # fg.axis(ymin=0,ymax=800000)
        # plt.xticks(rotation=90)
        # plt.show()
        # sns.distplot(df_train['SalePrice'])
        # plt.show()
        corrmat = df_train.corr()
        # f,ax = plt.subplots(figsize=(12,9))
        # sns.heatmap(corrmat,vmax=.8,square=True)
        # # plt.show()
        # k = 10  # number of variables for heatmap
        # cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        # cm = np.corrcoef(df_train[cols].values.T)
        # sns.set(font_scale=1.25)
        # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
        #                  yticklabels=cols.values, xticklabels=cols.values)
        # plt.show()
        sns.set()
        cols =['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        sns.pairplot(df_train[cols])
        plt.show()

    def missing_data(self):
        df_train = pd.read_csv('/Users/hanzhao/PycharmProjects/ml-example/file/data/train.csv')
        total = df_train.isnull().sum().sort_values(ascending=True)
        percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=True)
        missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
        df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
        df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
        print(df_train.isnull().sum().max())


    def out_liars(self):
        """TODO handle something that out liars"""

        pass





hp = HousePrice()
hp.missing_data()