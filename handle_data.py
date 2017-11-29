""" @description this python file shows how to slice data with train and test"""

import pandas as pd
from random import shuffle
from common_util import get_path
import warnings

warnings.filterwarnings('ignore')


def get_data():
    model_data_path = get_path('section1', 'model_data_path')
    model_data = pd.read_excel(model_data_path)
    model_matrix = model_data.as_matrix()
    shuffle(model_matrix)
    p = 0.8  # ratio of training
    train_data = model_matrix[:int(len(model_matrix) * p), :]
    test_data = model_matrix[int(len(model_matrix) * p):, :]
    print(train_data, test_data)
    return train_data, test_data


def explore_data():
    source_data_path = get_path('section1', 'source_data_path')
    result_data_path = get_path('section1', 'result_data_path')
    data = pd.read_csv(source_data_path, encoding='utf-8')
    explore = data.describe(percentiles=[], include='all').T  # T is transposition
    explore['null'] = len(data) - explore['count']
    explore = explore[['null', 'max', 'min']]
    explore.columns = [u'空值数', u'最大值', u'最小值']
    explore.describe()
    explore.to_excel(result_data_path)


def clean_data():
    source_data_path = get_path('section1', 'source_data_path')
    clean_data_path = get_path('section1', 'clean_data_path')
    data = pd.read_csv(source_data_path, encoding='utf-8')
    data = data[data['SUM_YR_1'].notnull() * data['SUM_YR_2'].notnull()]
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)
    data = data[index1 | index2 | index3]
    data.to_excel(clean_data_path)


def standrand_data():
    lrmfc_data_path = get_path('section1', 'LRFMC_data_path')
    data = pd.read_excel(lrmfc_data_path)
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]
    zscored_data_path = get_path('section1', 'zscored_data_path')
    data.to_excel(zscored_data_path, index=False)




