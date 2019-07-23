import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.set_random_seed(1)

# load data train/test data 60000 [28*28] ; label 0~9
mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)

# parameter
learning_rate = 0.001
training_epochs = 1000
batch_size = 100
# network parameter
n_hidden_1 = 3000
# input/output
n_input = 28 * 28
n_classes = 10

# x , y
X=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

# network model: input * 1, hidden layer * 1, output *1
W1=tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev=0.1))
b1=tf.Variable(tf.zeros([n_hidden_1]))
hidden1=tf.nn.relu(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.truncated_normal([n_hidden_1,n_classes],stddev=0.1))
b2=tf.Variable(tf.zeros([n_classes]))
y_hat=tf.nn.softmax(tf.matmul(hidden1,W2)+b2)

# aviod log0
loss=-tf.reduce_sum(y*tf.log(y_hat+0.0000000001))
train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# init
init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)
start_time = time.time()

# train
for i in range(training_epochs):
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    session.run(train,feed_dict={X:batch_xs,y:batch_ys})

prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1))
accuracy=tf.reduce_mean(tf.cast(prediction,"float"))
print("(3 layer feed-forward) Cost trainning time = "+ str(time.time() - start_time) +" seconds;" ,end=' ')
print("Test accurary = ",end='')
print(session.run(accuracy,feed_dict={X:mnist.test.images,y:mnist.test.labels}))
# (3 layer feed-forward) Cost trainning time = 15.84966516494751 seconds; Test accurary = 0.9655