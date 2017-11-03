from keras.models import Sequential

from read_data import get_data
from cm_plot import cm_plot
from keras.models import load_model

net_file = '/Users/hanzhao/PycharmProjects/ml-example/file/tmp/net.h5'


def lm_model():
    train = get_data()[0]
    test = get_data()[1]
    net = load_model(net_file)
    predict_result = net.predict_classes(test[:, :3]).reshape(len(test))  ## transform result
    cm_plot(test[:, 3], predict_result).show()


