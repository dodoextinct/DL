from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import hy_params
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

checkpoint_file = tf.train.latest_checkpoint(os.path.join(hy_params.checkpoint_dir, 'checkpoints'))
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

test_data = np.array([mnist.test.images[100]])

input_x = tf.get_default_graph().get_operation_by_name("input_x").outputs[0]

prediction = tf.get_default_graph().get_operation_by_name("prediction").outputs[0]

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)
    data = sess.run(prediction, feed_dict={input_x: test_data})
    print("Prediction", data.argmax())

img=mnist.test.images[100].reshape([28,28])
plt.gray()
plt.imshow(img)
plt.show()

