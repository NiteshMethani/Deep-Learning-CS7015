import tensorflow as tf


class MNIST_CNN:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('layer0'):
                X_img = tf.reshape(X, [-1, 28, 28, 1])

            # Convolutional Layer #1 and Pooling Layer #1
            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=64, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool1 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[2, 2], strides=2, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Convolutional Layer #2 and Pooling Layer #2
            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool2 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[2, 2], strides=2, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Convolutional Layer #3 and Pooling Layer #3
            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                
            with tf.variable_scope('layer4'):
                conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool4 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[2, 2], strides=2, padding="SAME", activation=tf.nn.relu, use_bias=True)

            # Logits (no activation) Layer
            with tf.variable_scope('layer5'):
                dense1 = tf.layers.conv2d(inputs=pool4, filters=1024, kernel_size=[1, 1], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                dense2 = tf.layers.conv2d(inputs=dense1, filters=1024, kernel_size=[1, 1], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                pool5 = tf.layers.conv2d(inputs=dense2, filters=10, kernel_size=[1, 1], strides=1, padding="SAME", activation=tf.nn.relu, use_bias=True)
                global_avg = tf.layers.average_pooling2d(inputs=pool5, pool_size=[4, 4], strides=1, padding="VALID")
                logits = tf.reshape(global_avg, [-1, 10])
                prediction = tf.nn.softmax(logits)

        return [X_img, conv1, pool1, conv2, pool2, conv3, conv4, pool4, dense1, dense2, pool5, prediction], logits

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
