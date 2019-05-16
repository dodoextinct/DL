from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import hy_params
import model
import os

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X = model.X
Y = model.Y

checkpoint_dir = os.path.abspath(os.path.join(hy_params.checkpoint_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

init = tf.global_variables_initializer()

batch_x, batch_y = mnist.train.next_batch(hy_params.batch_size)

print(batch_x.shape)
print(batch_y.shape)

all_loss=[]
with tf.Session() as sess:
    writer_1 = tf.summary.FileWriter("./runs/summary/",sess.graph)
    sum_var = tf.summary.scalar("loss", model.accuracy)
    write_op = tf.summary.merge_all()
    sess.run(init)

    for step in range(1, hy_params.num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(hy_params.batch_size)
        sess.run(model.train_op, feed_dict={X: batch_x, Y: batch_y})

        if step%hy_params.display_step == 0 or step == 1:
            loss,acc,summary = sess.run([model.loss_op, model.accuracy, write_op],feed_dict = {X: batch_x, Y: batch_y})
            all_loss.append(loss)
            writer_1.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +  "{:.3f}".format(acc))
        
        if step % hy_params.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step = step)
            print("Saved model checkpoint to {}\n".format(path))
    print("Optimization Finished")
    print("Testing Accuracy:", \
sess.run(model.accuracy, feed_dict={X: mnist.test.images,
Y: mnist.test.labels}))

import matplotlib.pyplot as plt

plt.plot(range(1,101), all_loss)
plt.show()


