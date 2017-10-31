"""@description Largrange interpolation solution """

import pandas as pd
from scipy.interpolate import lagrange

inputfile = '/Users/hanzhao/PycharmProjects/ml-example/file/data/missing_data.xls'  # source data
outputfile = '/Users/hanzhao/PycharmProjects/ml-example/file/tmp/missing_data_processed.xls'  # output data (after interpolation)

data = pd.read_excel(inputfile, header=None)

def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]

    return lagrange(y.index, list(y))(n)


def largrange_interpolation():
    # judge data if interpolation
    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = ployinterp_column(data[i], j)

    data.to_excel(outputfile, header=None, index=False)
    return data
