import os
import time
import argparse
import importlib
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from visualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, y_sampler, z_sampler, data, model, scale=10.0):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.y_dim = self.g_net.y_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y') # Real
        self.y_ = tf.placeholder(tf.float32, [None, self.y_dim], name='y_') # Fake
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z, self.y_)

        self.d, self.dlogits = self.d_net(self.x, reuse=False)
        self.d_, self.d_logits= self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)
        self.c_loss_r = tf.losses.softmax_cross_entropy(self.y, self.dlogits)
        self.c_loss_f = tf.losses.softmax_cross_entropy(self.y_, self.d_logits)

        self.c_loss = 0.5*(self.c_loss_r + self.c_loss_f)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        # print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx + self.c_loss
        self.g_loss = self.g_loss + self.c_loss_f

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in tqdm(range(0, num_batches)):
            d_iters = 5

            if t % 500 == 0 or t < 25:
                d_iters = 100

            for _ in range(0, d_iters):
                bx, by = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                by_ = self.y_sampler(batch_size, self.y_dim)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.y: by, self.y_: by_, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            by_ = self.y_sampler(batch_size, self.y_dim)
            self.sess.run(self.g_adam, feed_dict={self.z: bz, self.y: by, self.y_: by_, self.x: bx})

            if t % 100 == 0 and t != 0:
                bx, by = self.x_sampler(batch_size)
                by_ = self.y_sampler(batch_size, self.y_dim)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.y: by, self.y_: by_, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.y_: by, self.z: bz}
                )
                c_loss = self.sess.run(
                    self.c_loss, feed_dict={self.x: bx, self.y: by, self.y_: by_, self.z: bz}
                )


                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss, g_loss))

                print('Compare Wasserstein loss ({}) to categorical loss ({}).'.format(d_loss, c_loss))

            if t % 100 == 0:
                bz = self.z_sampler(self.y_dim, self.z_dim)
                by = np.zeros((self.y_dim, self.y_dim))
                by[range(self.y_dim), range(self.y_dim)] = 1
                bx = self.sess.run(self.x_, feed_dict={self.z: bz, self.y_: by})
                bx = xs.data2img(bx)
                bx = bx.reshape((-1, 28, 28))
                fig, axes = plt.subplots(1, self.y_dim, sharey=True)

                for i in range(self.y_dim):
                    axes[i].imshow(bx[i])
                    axes[i].axis('off')

                fig.savefig('logs/images/{}.png'.format(t/100))
                plt.close()


if __name__ == '__main__':
    X_DIM = 784
    Y_DIM = 10
    Z_DIM = 100

    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='models')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    ys = data.LabelSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator(X_DIM, Y_DIM)
    g_net = model.Generator(X_DIM, Y_DIM, Z_DIM)
    wgan = WassersteinGAN(g_net, d_net, xs, ys, zs, args.data, args.model)
    wgan.train()
