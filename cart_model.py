from sklearn.tree import DecisionTreeClassifier

from read_data import get_data
from sklearn.externals import joblib

from cm_plot import cm_plot


def cart_model():
    treefile = '/Users/hanzhao/PycharmProjects/ml-example/file/tmp/tree.pkl'
    tree = DecisionTreeClassifier()

    train = get_data()[0]
    tree.fit(train[:, :3], train[:, 3])

    joblib.dump(tree, treefile)  # save training model by joblib

    cm_plot(train[:, 3], tree.predict(train[:, :3])).show()


cart_model()
