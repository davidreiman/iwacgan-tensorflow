import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, x_dim, y_dim, name='mlp-discriminator'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.name = name

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tc.layers.fully_connected(
                x, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(
                fc1, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)
            fc3 = tc.layers.fully_connected(
                fc2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc3 = leaky_relu(fc3)
            class_logits = tc.layers.fully_connected(
                fc3, self.y_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc4 = tc.layers.fully_connected(fc3, 1, activation_fn=tf.identity)
            return fc4, class_logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def loss(self, prediction, target):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target))


class Generator(object):
    def __init__(self, x_dim, y_dim, z_dim, name='mlp-generator'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.name = name

    def __call__(self, z, y):
        with tf.variable_scope(self.name) as vs:

            fc = tf.concat([z, y], 1)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tcl.fully_connected(
                fc, 512,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tcl.batch_norm
            )
            fc = leaky_relu(fc)
            fc = tc.layers.fully_connected(
                fc, self.x_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Classifier(object):
    def __init__(self, x_dim, y_dim, name='mlp-classifier'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:

            fc1 = tc.layers.fully_connected(
                x, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tcl.batch_norm
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(
                fc1, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tcl.batch_norm
            )
            fc2 = leaky_relu(fc2)
            fc3 = tc.layers.fully_connected(
                fc2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tcl.batch_norm
            )
            fc3 = leaky_relu(fc3)
            class_logits = tc.layers.fully_connected(
                fc3, self.y_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tcl.batch_norm
            )

        return class_logits
