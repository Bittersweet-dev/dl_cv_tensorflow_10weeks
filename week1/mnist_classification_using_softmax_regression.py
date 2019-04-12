# -*- coding: utf-8 -*-

import tensorflow as tf

# MNIST 데이터를 다운로드 합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 입력값과 출력값을 받기 위한 플레이스홀더를 정의합니다.
#
# x: 입력 데이터,
# 784: 28 x 28, MNIST 의 손글씨 이미지 크기
# 
# y: 출력 데이터,
# 10: 숫자 0~9를 인식하기 위한 크기
#
# shape=[row, col]
#
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 변수들을 설정하고 Softmax Regression 모델을 정의합니다.
#
# W: 이미지(784)를 숫자(10)로 대응하기 위한 가중치(Weight)
# b: 바이어스(bias)
#
# softmax: n개의 변수 x에 대해 e^x / SUM(all e^x) 로 치환하는 함수.
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
# reduce_sum: 더하는 대상을 차원(행, 열, ...)을 지정하여 수행하는 함수.
# softmax regression: 일반화된 버전의 logistic regression.
#
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(logits)

# cross-entropy 손실 함수와 옵티마이저를 정의합니다.
#
# cross entropy: 특정 이벤트를 식별(identify)하기 위한 평균 비트 수를 계산.
#
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) # tf.nn.softmax_cross_entropy_with_logits API를 이용한 구현
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 세션을 열고 변수들에 초기값을 할당합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 1, 3, 5, 10, 100, 1000번 씩 반복을 각각 수행하면서 최적화를 수행합니다.
for j in [1, 3, 5, 10, 100, 1000]:
	for i in range(j):
	  batch_xs, batch_ys = mnist.train.next_batch(100)
	  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

	# 학습이 끝나면 학습된 모델의 정확도를 출력합니다.
	correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("정확도(Accuracy): %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})) # 정확도 : 약 91%