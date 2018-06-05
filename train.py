import tensorflow as tf
import os.path
import random
import numpy as np
import glob

BOTTLENECK_TENSOR_SIZE = 1536
CACHE_DIR = './tmp/tensor_3/'
save_path = "./classify/model_3.ckpt-50000"
train = False
n_classes = 3

def read_data(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    bottlenecks = []
    labels = []
    for idx, folder in enumerate(cate):
        for bottleneck_path in glob.glob(folder + '/*.txt'):
            bottleneck_file = open(bottleneck_path, 'r')
            bottleneck_string = bottleneck_file.read()
            bottleneck = [float(x) for x in bottleneck_string.split(',')]
            ground_truth = np.zeros(n_classes, dtype = np.float32)
            ground_truth[idx] = 1.0
            bottlenecks.append(bottleneck)
            labels.append(ground_truth)
    return np.asarray(bottlenecks, np.float32), np.asarray(labels, np.int32)

np.random.seed(65663)
data, label = read_data(CACHE_DIR)
num_example = data.shape[0]
print(num_example)
arr = np.arange(num_example)
np.random.shuffle(arr)
print(len(arr))
data = data[arr]
label = label[arr]
batch_size = 128
# 将所有数据分为训练集和验证集
k = 5
ratio = 0.1
s = int(num_example*ratio)
print(s)
x_train = data[:-2*s]
y_train = label[:-2*s]
x_val = data[-2*s:-s]
y_val = label[-2*s:-s]
x_test = data[-s:]
y_test = label[-s:]
index_train = [i for i in range(len(x_train))]
index_val = [i for i in range(len(x_val))]
print(len(x_train), len(x_val), len(x_test))



bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
with tf.name_scope('training_ops'):
    weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
    biases = tf.Variable(tf.zeros([n_classes]))
    logits = tf.add(tf.matmul(bottleneck_input, weights), biases)
    final_tensor = tf.nn.softmax(logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)
correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 50000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    if train:
        max_val_acc = 0
        for j in range(epochs):
            # train
            loss, _ = sess.run([cross_entropy_mean, train_step], feed_dict={bottleneck_input: np.array(x_train), ground_truth_input: np.array(y_train)})
            if j%200 == 0 or j+1 == epochs:
                z = random.sample(index_val, batch_size)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: np.array(x_val[z]),
                                                                                ground_truth_input: np.array(y_val[z])})
                print(j, loss, 'validation accuracy = %.1f%%' % (validation_accuracy * 100))
                if validation_accuracy > max_val_acc:
                    max_val_acc = validation_accuracy
                    saver.save(sess=sess, save_path=save_path)
                    print('-------saving----------')
                    test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: np.array(x_test),
                                                                         ground_truth_input: np.array(y_test)})
                    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
                if j+1 == epochs:
                    saver.save(sess=sess, save_path=save_path+'-50000')

            if j%1000 == 0 or j+1 == epochs:
                test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: np.array(x_test),
                                                                                ground_truth_input: np.array(y_test)})
                print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
        print('----------------------done ---------------------')
    else:
        saver.restore(sess=sess, save_path=save_path)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: np.array(x_test),
                                                             ground_truth_input: np.array(y_test)})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    sess.close()