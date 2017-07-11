from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helpers


beginTime = time.time()

# parameter definitions
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# prepare data
data_sets = data_helpers.load_data()

# define input placeholders
images_placeholders = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholders = tf.placeholder(tf.int64, shape=[None])

# define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

# classifier's result
logits = tf.matmul(images_placeholders, weights) + biases

# loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholders))

# training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholders)

# operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Repeat max_steps times
    for i in range(max_steps):
        # Generate input data batch
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        # Periodically print out the model's current accuracy
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                images_placeholders: images_batch, labels_placeholders: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

        # Perform a single training step
        sess.run(train_step, feed_dict={images_placeholders: images_batch,
                                        labels_placeholders: labels_batch})




    # After finishing the training, evaluate on the test set
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholders: data_sets['images_test'],
        labels_placeholders: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))



endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
