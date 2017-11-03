"""@description training with LM Classifier and cart classifier"""
from keras.models import Sequential

from keras.layers.core import Dense, Activation  # neural network ,activation function

from read_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

netfile = '/Users/hanzhao/PycharmProjects/ml-example/file/tmp/net.h5'  # path with LM model
treefile = '/Users/hanzhao/PycharmProjects/ml-example/file/tmp/tree.pkl'  # path with cart model


def train_lm_classification():
    # init net
    net = Sequential()

    net.add(Dense(input_dim=3, output_dim=10))  # input to hide
    net.add(Activation('relu'))  # relu function between their
    net.add(Dense(input_dim=10, output_dim=1))  # hide to output
    net.add(Activation('sigmoid'))  # sigmoid's function between their
    net.compile(loss='binary_crossentropy', optimizer='adam')  ## use adam
    train = get_data()[0]

    net.fit(train[:, :3], train[:, 3], nb_epoch=1000, batch_size=1)  # train model ,1000's loop

    net.save(netfile)


def train_cart_classification():
    tree = DecisionTreeClassifier()
    train = get_data()[0]
    tree.fit(train[:, :3], train[:, 3])
    joblib.dump(tree, treefile)  # save training model by joblib
