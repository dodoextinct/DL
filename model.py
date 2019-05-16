import tensorflow as tf
import hy_params

X = tf.placeholder("float", [None, hy_params.num_input], name="input_x")
Y = tf.placeholder("float", [None, hy_params.num_classes], name="input_y")

weights = {'h1' : tf.Variable(tf.random_normal([hy_params.num_input, hy_params.n_hidden_1])),
            'h2' : tf.Variable(tf.random_normal([hy_params.n_hidden_1, hy_params.n_hidden_2])),
            'out' : tf.Variable(tf.random_normal([hy_params.n_hidden_2, hy_params.num_classes]))}

biases = {'b1': tf.Variable(tf.random_normal([hy_params.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([hy_params.n_hidden_2])),
                'out' : tf.Variable(tf.random_normal([hy_params.num_classes]))}

layer_1 = tf.add(tf.matmul(X, weights['h1']),biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
logits = tf.add(tf.matmul(layer_2, weights['out']),biases['out'])

prediction = tf.nn.softmax(logits, name='prediction')

loss_op =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=hy_params.learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)
,name='accuracy')

