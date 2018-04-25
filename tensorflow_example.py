import tensorflow as tf
from numpy.random import RandomState


class TensorExample:
    def __init__(self):
        self.batch_size = 8
        self.dataset_size = 128
        self.STEPS = 5000
        self.w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        self.w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
        self.x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
        self.y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    def def_net_structure(self):
        a = tf.matmul(self.x, self.w1)
        y = tf.matmul(a, self.w2)

        y = tf.sigmoid(y)
        cross_entropy = -tf.reduce_mean(
            self.y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

        train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
        rdm = RandomState(1)

        X = rdm.rand(self.dataset_size, 2)

        Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            #
            # print(sess.run(self.w1))
            # print(sess.run(self.w2))

            for i in range(self.STEPS):
                start = (i * self.batch_size) % self.dataset_size
                end = min(start + self.batch_size, self.dataset_size)

                sess.run(train_step, feed_dict={self.x: X[start:end], self.y_: Y[start:end]})
                if i % 1000 == 0:
                    total_cross_entropy = sess.run(cross_entropy, feed_dict={self.x: X, self.y_: Y})
                print("After  %d train step ,cross entropy on all data is %g" % (i, total_cross_entropy))

                print(sess.run(self.w1))
                print(sess.run(self.w2))


if __name__ == '__main__':
    te = TensorExample()
    te.def_net_structure()
