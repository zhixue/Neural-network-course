import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

tf.set_random_seed(1)

# load data train/test data 60000 [28*28] ; label 0~9
mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)

# parameter
learning_rate = 0.0001
training_epochs = 20000
batch_size = 50
# network parameter

# input/output
n_input = 28 * 28
n_classes = 10

# x , y
X=tf.placeholder("float",[None,n_input])
Y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(X, [-1,28,28,1])

# model

W_conv1 = weight_variable([5, 5, 1, 6])#32
b_conv1 = bias_variable([6])
conv1 = conv2d(x_image, W_conv1)

relu1 = tf.nn.relu(conv1 + b_conv1)

pool1 = max_pool_2x2(relu1)

W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
conv2 = conv2d(pool1, W_conv2)

relu2 = tf.nn.relu(conv2 + b_conv2)

pool2 = max_pool_2x2(relu2)

W_fc1 = weight_variable([7 * 7 * 16, 84])
b_fc1 = bias_variable([84])
pool2_flat = tf.reshape(pool2, [-1, 7*7*16])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
fc1_drop = tf.nn.dropout(fc1, keep_prob)
# output
W_fc2 = weight_variable([84, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2)



cross_entropy = -tf.reduce_sum(Y_*tf.log(y_conv+0.0000000001))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# init
init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)
start_time = time.time()


for i in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    if i%2000 == 0:
        train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})

print("(LeNet5_CNN) Cost trainning time = "+ str(time.time() - start_time) +" seconds;" ,end=' ')
print("test accuracy = %g"%accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0}))
#(LeNet5_CNN) Cost trainning time = 559.8334238529205 seconds; test accuracy = 0.9837


############## feature plot
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# imput image
fig2,ax2 = plt.subplots(figsize=(2,2))
ax2.imshow(np.reshape(mnist.train.images[0], (28, 28)))
plt.show()

# conv1
input_image = mnist.train.images[0:1]
conv1_6 = sess.run(conv1, feed_dict={X:input_image})     # [1, 28, 28 ,8]
conv1_transpose = sess.run(tf.transpose(conv1_6, [3, 0, 1, 2]))
fig3,ax3 = plt.subplots(nrows=1, ncols=6, figsize = (6,1))
for i in range(6):
    ax3[i].imshow(conv1_transpose[i][0])
plt.title('Conv1 6x28x28')
plt.show()

# pool1
pool1_6 = sess.run(pool1, feed_dict={X:input_image})     # [1, 14, 14, 8]
pool1_transpose = sess.run(tf.transpose(pool1_6, [3, 0, 1, 2]))
fig4,ax4 = plt.subplots(nrows=1, ncols=6, figsize=(6,1))
for i in range(6):
    ax4[i].imshow(pool1_transpose[i][0])
plt.title('Pool1 6x14x14')
plt.show()

# conv2
conv2_16 = sess.run(conv2, feed_dict={X:input_image})          # [1, 14, 14, 16]
conv2_transpose = sess.run(tf.transpose(conv2_16, [3, 0, 1, 2]))
fig5,ax5 = plt.subplots(nrows=1, ncols=16, figsize = (16, 1))
for i in range(16):
    ax5[i].imshow(conv2_transpose[i][0])
plt.title('Conv2 16x14x14')
plt.show()

# pool2
pool2_16 = sess.run(pool2, feed_dict={X:input_image})         #[1, 7, 7, 16]
pool2_transpose = sess.run(tf.transpose(pool2_16, [3, 0, 1, 2]))
fig6,ax6 = plt.subplots(nrows=1, ncols=16, figsize = (16, 1))
plt.title('Pool2 16x7x7')
for i in range(16):
    ax6[i].imshow(pool2_transpose[i][0])
plt.show()

