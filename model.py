"""@description test lm model & cart model"""

from handle_data import get_data
from cm_plot import cm_plot
from keras.models import load_model
from sklearn.externals import joblib
from common_util import get_path


def lm_model():
    net_file = get_path('section2', 'net_file')
    train = get_data()[0]
    test = get_data()[1]
    net = load_model(net_file)
    predict_result = net.predict_classes(test[:, :3]).reshape(len(test))  ## transform result
    cm_plot(test[:, 3], predict_result).show()


def cart_model():
    tree_file = get_path('section2', 'tree_file')
    train = get_data()[0]
    test = get_data()[1]
    cart = joblib.load(tree_file)
    cm_plot(train[:, 3], cart.predict(train[:, :3])).show()




