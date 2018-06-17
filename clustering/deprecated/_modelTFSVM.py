# -*- coding: utf-8 -*-


import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load, matrixify


# disable GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# session
# sess = tf.Session() 

# macros
batch_size = 100
feature_num = 14
species_num = 36

# load data
leaves = load(2, ...)
x_vals = np.array(leaves.data)
y_vals = matrixify(leaves.target)

# print(x_vals.shape)
# print(y_vals.shape)

# sperate dataset for traning and prediction
# x_length = len(x_vals)
# train_indices = np.random.choice(x_length, round(x_length*0.8), replace=False)
# test_indices = np.array(list(set(range(x_length)) - set(train_indices)))
# x_vals_train = x_vals[train_indices]
# x_vals_test = x_vals[test_indices]
# y_vals_train = y_vals[train_indices]
# y_vals_test = y_vals[test_indices]

example_id = np.array([ int(i) for i in range(len(y_vals)) ])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_vals, 'id': example_id},
    y=y_vals,
    num_epochs=None,
    shuffle=True)

svm = tf.contrib.learn.SVM(
    example_id_column='id',
    feature_columns=(tf.contrib.layers.real_valued_column(
                        column_name='x', dimension=128),),
    l2_regularization=0.1)

svm.fit(input_fn=train_input_fn, steps=10)

# # initialisation
# x_data = tf.placeholder(shape=[None, feature_num], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, species_num], dtype=tf.float32)

# # 创建变量
# A = tf.Variable(tf.random_normal(shape=[feature_num, 1]))
# b = tf.Variable(tf.random_normal(shape=[species_num, 1]))

# # linear model
# model_output = tf.subtract(tf.matmul(x_data, A), b)

# # declare vector L2 'norm' function squared
# l2_norm = tf.reduce_sum(tf.square(A))

# # loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# alpha = tf.constant([0.01])
# classification_term = tf.reduce_mean(tf.maximum(0.0, tf.subtract(1.0, tf.multiply(model_output, y_target))))
# loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# # start traning
# my_opt = tf.train.GradientDescentOptimizer(0.01)
# train_step = my_opt.minimize(loss)

# init = tf.global_variables_initializer()
# sess.run(init)

# # Training loop
# loss_vec = []
# train_accuracy = []
# test_accuracy = []
# for _ in range(20000):
#     rand_index = np.random.choice(len(x_vals_train), size=batch_size)
#     rand_x = x_vals_train[rand_index]
#     rand_y = np.transpose([y_vals_train[rand_index]])
#     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
