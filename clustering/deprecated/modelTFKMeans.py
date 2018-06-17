# -*- coding: utf-8 -*-


import os
import random

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load


# ignore GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# # load data
# leaves = load(2, ...)
# x_vals = np.array(leaves.data)
# y_vals = np.array(leaves.target)

# macros
num_steps = 50      # total steps to train
batch_size = 1024   # number of samples per batch
k = 25              # number of clusters
num_classes = 36    # 36 species
num_features = 14   # each data has 14 features

def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """
 
    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)
 
    # Find out the dimensionality
    dim = len(vectors[0])
    print(dim)
 
    # Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    random.shuffle(vector_indices)
 
    # GRAPH OF COMPUTATION
    # We initialize a new graph and set it as the default during each run
    # of this algorithm. This ensures that as this function is called
    # multiple times, the default graph doesn't keep getting crowded with
    # unused ops and Variables from previous function calls.
 
    graph = tf.Graph()
 
    with graph.as_default():
 
        # SESSION OF COMPUTATION
 
        sess = tf.Session()
 
        ## CONSTRUCTING THE ELEMENTS OF COMPUTATION
 
        ## First lets ensure we have a Variable vector for each centroid,
        ## initialized to one of the vectors from the available data points
        centroids = [ tf.Variable((vectors[vector_indices[i]]))
                        for i in range(noofclusters) ]
        ## These nodes will assign the centroid Variables the appropriate values
        centroid_value = tf.placeholder('float64', [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
 
        ## Variables for cluster assignments of individual vectors
        ## (initialized to 0 at first)
        assignments = [ tf.Variable(0) for i in range(len(vectors)) ]
        ## These nodes will assign an assignment Variable the appropriate value
        assignment_value = tf.placeholder('int32')
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment, assignment_value))
 
        ## Now let's construct the node that will compute the mean
        # The placeholder for the input
        mean_input = tf.placeholder('float', [None, dim])
        # The Node/op takes the input and computes a mean along the 0th
        # dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)
 
        ## Node for computing Euclidean distances
        # Placeholders for input
        v1 = tf.placeholder('float', [dim])
        v2 = tf.placeholder('float', [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
 
        ## This node will figure out which cluster to assign a vector to,
        ## based on Euclidean distances of the vector from the centroids.
        # Placeholder for input
        centroid_distances = tf.placeholder('float', [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)
 
        ## INITIALIZING STATE VARIABLES
 
        ## This will help initialization of all Variables defined with respect
        ## to the graph. The Variable-initializer should be defined after
        ## all the Variables have been constructed, so that each of them
        ## will be included in the initialization.
        init_op = tf.global_variables_initializer()
 
        # Initialize all variables
        sess.run(init_op)
 
        ## CLUSTERING ITERATIONS
 
        # Now perform the Expectation-Maximization steps of K-Means clustering
        # iterations. To keep things simple, we will only do a set number of
        # iterations, instead of using a Stopping Criterion.
        noofiterations = 100
        for iteration_n in range(noofiterations):
 
            ## EXPECTATION STEP
            ## Based on the centroid locations till last iteration, compute
            ## the _expected_ centroid assignments.
            # Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # Compute Euclidean distance between this vector and each
                # centroid. Remember that this list cannot be named
                # 'centroid_distances', since that is the input to the
                # cluster assignment node.
                distances = [ sess.run(euclid_dist,
                                feed_dict={v1: vect, v2: sess.run(centroid)})
                                for centroid in centroids ]
                # Now use the cluster assignment node, with the distances
                # as the input
                assignment = sess.run(cluster_assignment, feed_dict={
                                        centroid_distances: distances})
                # Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                            assignment_value: assignment})
 
            ## MAXIMIZATION STEP
            # Based on the expected state computed from the Expectation Step,
            # compute the locations of the centroids so as to maximize the
            # overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                # Collect all the vectors assigned to this cluster
                assigned_vects = [ vectors[i] for i in range(len(vectors))
                                    if sess.run(assignments[i]) == cluster_n ]
                # Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                                mean_input: np.array(assigned_vects)})
                # Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                            centroid_value: new_location})
 
        # Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

# center,result = TFKMeansCluster(x_vals, 36)
# print(center)
############生成测试数据###############
sampleNo = 10000#数据数量
mu = 3
# 二维正态分布
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = np.linalg.cholesky(Sigma)
srcdata = np.dot(np.random.randn(sampleNo, 2), R) + mu
# srcdata = load(0, 2, ...).data
print(srcdata)
plt.plot(srcdata[:,0],srcdata[:,1],'bo')
############kmeans算法计算###############
k = 36
center, result = TFKMeansCluster(srcdata, k)
print(center)
############利用seaborn画图###############
res={"x":[],"y":[],"kmeans_res":[]}
for i in range(len(result)):
    res["x"].append(srcdata[i][0])
    res["y"].append(srcdata[i][1])
    res["kmeans_res"].append(result[i])
pd_res = pd.DataFrame(res)
sns.lmplot("x","y",data=pd_res,fit_reg=False,size=5,hue="kmeans_res")
plt.show()

# # input images
# X = tf.placeholder(tf.float32, shape=[None, num_features])
# # labels (for assigning a label to a centroid and testing)
# Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# # K-Means Parameters
# kmeans = tf.contrib.factorization.KMeans(
#             inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

# # build K-Means graph
# (all_scores, cluster_idx, scores, cluster_centers_initialized,
#     init_op, train_op) = kmeans.training_graph()
# cluster_idx = cluster_idx[0]            # fix for cluster_idx being a tuple
# avg_distance = tf.reduce_mean(scores)

# # initialize the variables (i.e. assign their default value)
# init_vars = tf.global_variables_initializer()

# # start TensorFlow session
# sess = tf.Session()

# # run the initializer
# sess.run(init_vars, feed_dict={X: x_vals})
# sess.run(init_op, feed_dict={X: x_vals})

# # training
# for i in range(1, num_steps+1):
#     _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
#                             feed_dict={X: x_vals})
#     if i % 10 == 0 or i == 1:
#         print(f'Step {i}, Avg Distance: {d}')

# # assign a label to each centroid
# # count total number of labels per centroid, using the label of each training
# # sample to their closest centroid (given by 'idx')
# counts = np.zeros(shape=(k, num_classes))
# for i, x in enumerate(idx):
#     counts[x] += y_vals[i]
# # assign the most frequent label to the centroid
# labels_map = [ np.argmax(c) for c in counts ]
# labels_map = tf.convert_to_tensor(labels_map)

# # evaluation ops
# # lookup: centroid_id -> label
# cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# # compute accuracy
# correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
# accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # test model
# test_x, test_y = x_vals, y_vals
# print('Test Accuracy:', sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
