# Artificial Neural Network
# Only available on Tensorflow v1.x

import tensorflow as tf
import numpy as np

'''
Datasets
'''
# [tail, barking, legs, length]
x_data = np.array([
    [1, 1, 4, 50],
    [1, 0, 4, 5],
    [1, 0, 4, 20],
    [1, 0, 4, 10],
    [1, 1, 4, 100],
    [1, 0, 4, 12]
])

# [dog, mouse, cat]
y_data = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1]
])


'''
Neural network
'''
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

W1 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([4]))
L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

W2 = tf.Variable(tf.random_uniform([4, 5]))
b2 = tf.Variable(tf.zeros([5]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_uniform([5, 6]))
b3 = tf.Variable(tf.zeros([6]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

W4 = tf.Variable(tf.random_uniform([6, 4]))
b4 = tf.Variable(tf.zeros([4]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

W5 = tf.Variable(tf.random_uniform([4, 3]))
b5 = tf.Variable(tf.zeros([3]))
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))

W6 = tf.Variable(tf.random_uniform([3, 3]))
b6 = tf.Variable(tf.zeros([3]))
model = tf.nn.relu(tf.add(tf.matmul(L5, W6), b6))

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
trainer = optimizer.minimize(cost)


'''
Learning
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for loop in range(1000):
    sess.run(trainer, feed_dict={x: x_data, y: y_data})

    if (loop + 1) % 10 == 0:
        print(loop + 1, sess.run(cost, feed_dict={x: x_data, y: y_data}))


'''
Output
'''
prediction = tf.argmax(model, 1)
target = tf.argmax(y, 1)
print('Prediction: ', sess.run(prediction, feed_dict={x: x_data}))
print('Real value: ', sess.run(target, feed_dict={y: y_data}))

correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print('Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={x: x_data, y: y_data}))

sess.close()