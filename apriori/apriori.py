import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def get_data_frame():
    data = {'ID': [1, 2, 3, 4, 5, 6],
            'Onion': [1, 0, 0, 1, 1, 1],
            'Potato': [1, 1, 0, 1, 1, 1],
            'Burger': [1, 1, 0, 0, 1, 1],
            'Milk': [0, 1, 1, 1, 0, 1],
            'Beer': [0, 0, 1, 0, 1, 0]}

    df = pd.DataFrame(data)
    df = df[['ID', 'Onion', 'Potato', 'Burger', 'Milk', 'Beer']]
    return df


'''
Lift(X→Y)

When Lift=1, X makes no impact on Y

When Lift>1, there is a relationship between X & Y
'''


def mlxtend_association_rules():
    freqence_result = mlxtend_apriori()
    # res = association_rules(freqence_result, min_threshold=1)
    res = association_rules(freqence_result, metric='lift', min_threshold=1)
    print("association:{0}".format(res))


def mlxtend_apriori():
    df = get_data_frame()
    res = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer']], min_support=0.60, use_colnames=True)
    print("support:{0}".format(res))
    return res


mlxtend_association_rules()
