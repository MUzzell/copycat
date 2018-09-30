
import tensorflow as tf
from  tensorflow import     keras
import numpy as np


class DQNNetwork(object):

    def __init__(self, input_dim, n_actions):
        self.input_dim = input_dim
        self.n_actions = n_actions

    def build_network_keras(self):

        nel = np.prod(self.input_dim)

        model = keras.Sequential([
                keras.layers.Reshape((4, 84, 84), input_shape=(4,84,84)),
                keras.layers.Conv2D(32, (8,8), strides=4,
                                    padding='same',
                                    activation='relu'),
                keras.layers.Conv2D(64, (4,4), strides=2,
                                    padding='same',
                                    activation='relu'),
                keras.layers.Conv2D(64, (3,3), strides=1,
                                    padding='same',
                                    activation='relu'),
                # keras.layers.Reshape(nel),
                keras.layers.Dense(512),
                keras.layers.Dense(6)
            ])

        return model

    def build_network(self):

        self.input = tf.placeholder(tf.float32, shape=self.input_dim)

        self.in_reshape = tf.reshape(self.input, [4, 84, 84])

        self.conv1 = tf.layers.conv2d(
            self.in_reshape, filters=32, stride=4,
            kernel_size=[8, 8], padding='same', name="conv1")
        self.conv1_relu = tf.nn.relu(self.conv1)

        self.conv2 = tf.layers.conv2d(
            self.conv1_relu, filters=64, stride=2,
            kernel_size=[4, 4], padding='same', name="conv2")
        self.conv2_relu = tf.nn.relu(self.conv2)

        self.conv3 = tf.layers.conv2d(
            self.conv2_relu, filters=64, stride=1,
            kernel_size=[3, 3], padding='same', name="conv3")
        self.conv3_relu = tf.nn.relu(self.conv3)

        nel = np.prod(self.input_dim)
        self.nel_reshape = tf.reshape(self.conv3_relu, [nel])

        self.fc = tf.layers.dense(
            self.nel_reshape, 512, name="fc")
        self.fc_relu = tf.nn.relu(self.fc)

        self.out = tf.layers.dense(
            self.fc_relu, self.n_actions, name="out")


    def forward(self, state):
        pass