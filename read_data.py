""" @description this python file shows how to slice data with train and test"""

import pandas as pd

from random import shuffle

model_data = pd.read_excel('/Users/hanzhao/PycharmProjects/ml-example/file/data/model.xls')


def get_data():
    model_matrix = model_data.as_matrix()

    shuffle(model_matrix)

    p = 0.8 # ratio of training

    train_data = model_matrix[:int(len(model_matrix)*p),:]

    test_data = model_matrix[int(len(model_matrix)*p):,:]

    return train_data, test_data


if  __name__ =="__main__" :
   print(get_data())