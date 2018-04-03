import tensorflow as tf
import shutil
import os.path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io

#这里是输出训练好的pb网络文件
export_dir = './tmp/expert-export'


if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#这里是在读取数据
mnist = scipy.io.loadmat('mnist-original.mat')
mnist['data'] = mnist['data'].astype(np.float32)
mnist['target'] = mnist['target'].astype(np.int32)
pic=mnist['data'].reshape((128*128,2747))
pic = pic.reshape(pic.shape[1],128, 128)
#print(pic[5])
#plt.imshow(pic[5])
#plt.show()
#print(mnist['target'][0,:])
#print(mnist['data'][0,:])
mnistt = scipy.io.loadmat('mnist-original-test.mat')
mnistt['data'] = mnist['data'].astype(np.float32)
mnistt['target'] = mnist['target'].astype(np.int32)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'): #创建summaries对象名称作用域
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

#构建网络
g = tf.Graph()
with g.as_default():
    x = tf.placeholder("float", shape=[None, 16384])
    y_ = tf.placeholder("float", shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    x_image = tf.reshape(x, [-1, 128, 128, 1])
    x_image1 = tf.random_crop(x_image,[-1, 88, 88, 1])
    #tf.summary.image('input', x_image, 10)
    h_conv1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1)

    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(h_pool2)

    W_conv3 = weight_variable([6, 6, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    print(h_pool3)

    W_conv4 = weight_variable([5, 5, 64, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    keep_prob = tf.placeholder("float")
    h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)
    print(h_conv4_drop)

    W_conv5 = weight_variable([3, 3, 128, 10])
    b_conv5 = bias_variable([10])
    h_pool5 = conv2d(h_conv4_drop, W_conv5) + b_conv5
    print(h_pool5)
    h_pool5 = tf.reshape(h_pool5,[-1,10])
    y_conv = tf.nn.softmax(h_pool5)
    #y_conv = np.reshape(y_conv,(20,10))
    print(y_conv)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 损失函数是交叉熵损失
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # tf.train.AdamOptimizer 使用Adam 算法的Optimizer
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # 这里 tf.argmax 就是返回最大的那个数值所在的下标 tf.equal是用来判别的，
    # 如果两者相等就输出True，如果不相等就输出False，是bool数据，因此之后要转化成float

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    tf.summary.scalar('accuracy', accuracy)

    # 首先tf.cast是把原来的correct_prediction数据类型为bool转化成float，然后tf.reduce_mean是计算平均值

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    #下面是训练过程的准确率
    for i in range(230):  #这里设置的训练的次数，由于
        if i%10 == 0:
           print(i)
           train_accuracy = accuracy.eval(
                {x:mnist['data'][i:i+99,:], y_:mnist['target'][i:i+99,:], keep_prob:1.0}, sess)
           #train_writer.add_summary(train_accuracy)
           #值得注意，用于检测训练准确率是随机时候因子设为全部失活
           print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(
            {x:mnist['data'][i:i+99,:], y_:mnist['target'][i:i+99,:], keep_prob: 0.5}, sess)  #在训练的过程是设置每次失活一半的神经元

    #下面是测试过程的准确率
    print("test accuracy %g" % accuracy.eval(
        {x: mnist['data'][1:20,:], y_: mnist['target'][1:20,:], keep_prob:1.0}, sess))



# Store variable
_W_conv1 = W_conv1.eval(sess)
_b_conv1 = b_conv1.eval(sess)
_W_conv2 = W_conv2.eval(sess)
_b_conv2 = b_conv3.eval(sess)
_W_conv3 = W_conv3.eval(sess)
_b_conv3 = b_conv3.eval(sess)
_W_conv4 = W_conv4.eval(sess)
_b_conv4 = b_conv4.eval(sess)
_W_conv5 = W_conv5.eval(sess)
_b_conv5 = b_conv5.eval(sess)

train_writer = tf.summary.FileWriter(export_dir + '/train', sess.graph)
sess.close()

# 创建一个新的图用来输出Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    x_2 = tf.placeholder("float", shape=[None, 16384], name="input")

    W_conv1_2 = tf.constant(_W_conv1, name="constant_W_conv1")
    b_conv1_2 = tf.constant(_b_conv1, name="constant_b_conv1")
    x_image_2 = tf.reshape(x_2, [-1, 128, 128, 1])
    x_image_22 = tf.random_crop(x_image_2,[-1, 88, 88, 1])
    h_conv1_2 = tf.nn.relu(conv2d(x_image_22, W_conv1_2) + b_conv1_2)
    h_pool1_2 = max_pool_2x2(h_conv1_2)

    W_conv2_2 = tf.constant(_W_conv2, name="constant_W_conv2")
    b_conv2_2 = tf.constant(_b_conv2, name="constant_b_conv2")
    h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
    h_pool2_2 = max_pool_2x2(h_conv2_2)

    W_conv3_2 = tf.constant(_W_conv3, name="constant_W_conv3")
    b_conv3_2 = tf.constant(_b_conv3, name="constant_b_conv3")
    h_conv3_2 = tf.nn.relu(conv2d(h_pool2_2, W_conv3_2) + b_conv3_2)
    h_pool3_2 = max_pool_2x2(h_conv3_2)

    W_conv4_2 = tf.constant(_W_conv4, name="constant_W_conv4")
    b_conv4_2 = tf.constant(_b_conv4, name="constant_b_conv4")
    h_conv4_2 = tf.nn.relu(conv2d(h_pool3_2, W_conv4_2) + b_conv4_2)
    keep_prob = tf.placeholder("float")
    h_conv4_2_drop = tf.nn.dropout(h_conv4_2, keep_prob)

    W_conv5_2 = tf.constant(_W_conv5, name="constant_W_conv5")
    b_conv5_2 = tf.constant(_b_conv5, name="constant_b_conv5")
    h_pool5_2 = conv2d(h_conv4_2_drop, W_conv5_2) + b_conv5_2
    h_pool5_2 = tf.reshape(h_pool5_2,[-1,10])
    y_conv_2 = tf.nn.softmax(h_pool5_2, name="output")


    sess_2 = tf.Session()
    init_2 = tf.initialize_all_variables();
    sess_2.run(init_2)

    graph_def = g_2.as_graph_def()
    tf.train.write_graph(graph_def, export_dir, 'expert-graph.pb', as_text=False)

    # Test trained model
    y__2 = tf.placeholder("float", [None, 10])
    correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

    print("check accuracy %g" % accuracy_2.eval(
        {x_2: mnistt['data'][1:10], y__2: mnistt['target'][1:10]}, sess_2))
