import tensorflow as tf
import pandas as pd
import numpy as np

class CabbageModel:
    def __init__(self):
        pass

    def create_model(self):
        model = tf.global_variables_initializer()
        data = pd.read_csv('./data/price_data.csv', sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:,1:-1] #feature
        y_data = xy[:,[-1]] # 가격
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # 러닝
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
            if step % 500 == 0:
                print('# %d 손실비용 : %d'%(step, cost_))
                print("- 배추가격 : %d" % (hypo_[0]))
                
        # 저장
        saver = tf.train.Saver()
        saver.save(sess, 'saved_model/saved.ckpt')
        print('저장완료')


