from __future__ import division
from __future__ import print_function
from time import gmtime, strftime

import time
import sys
import os
from pylab import *
from scipy import sparse
import numpy as np
import pandas as pd
import tensorflow as tf
import feather
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

# Given a set of ratings, 2 matrix factors that include one or more
# trainable variables, and a regularizer, uses gradient descent to
# learn the best values of the trainable variables.
def mf(ratings_train, ratings_val, W, H, regularizer, mean_rating, max_iter, lr = 0.01, decay_lr = False, log_summaries = False):
    # Extract info from training and validation data
    rating_values_tr, num_ratings_tr, user_indices_tr, item_indices_tr = extract_rating_info(ratings_train)
    rating_values_val, num_ratings_val, user_indices_val, item_indices_val = extract_rating_info(ratings_val)

    # Multiply the factors to get our result as a dense matrix
    result = tf.matmul(W, H)

    # Now we just want the values represented by the pairs of user and item
    # indices for which we had known ratings.
    result_values_tr = tf.gather(tf.reshape(result, [-1]), user_indices_tr * tf.shape(result)[1] + item_indices_tr, name="extract_training_ratings")
    result_values_val = tf.gather(tf.reshape(result, [-1]), user_indices_val * tf.shape(result)[1] + item_indices_val, name="extract_validation_ratings")

    # Calculate the difference between the predicted ratings and the actual
    # ratings. The predicted ratings are the values obtained form the matrix
    # multiplication with the mean rating added on.
    diff_op = tf.sub(tf.add(result_values_tr, mean_rating, name="add_mean"), rating_values_tr, name="raw_training_error")
    diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val, name="raw_validation_error")

    with tf.name_scope("training_cost") as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")

        cost = tf.div(tf.add(base_cost, regularizer), num_ratings_tr * 2, name="average_error")

    with tf.name_scope("validation_cost") as scope:
        cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), num_ratings_val * 2, name="average_error")

    with tf.name_scope("train") as scope:
        if decay_lr:
            # Use an exponentially decaying learning rate.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Passing global_step to minimize() will increment it at each step 
            # so that the learning rate will be decayed at the specified 
            # intervals.
            train_step = optimizer.minimize(cost, global_step=global_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_step = optimizer.minimize(cost)

    with tf.name_scope("training_rmse") as scope:
      rmse_tr = tf.sqrt(tf.reduce_sum(tf.square(diff_op)) / num_ratings_tr)

    with tf.name_scope("validation_rmse") as scope:
      # Validation set rmse:
      rmse_val = tf.sqrt(tf.reduce_sum(tf.square(diff_op_val)) / num_ratings_val)

    # Create a TensorFlow session and initialize variables.
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    if log_summaries:
        # Make sure summaries get written to the logs.
        accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)
        accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph_def)
    # Keep track of cost difference.
    last_cost = 0
    diff = 1
    # Run the graph and see how we're doing on every 1000th iteration.
    for i in range(max_iter):
        if i > 0 and i % 1000 == 0:
            if diff < 0.000001:
                print("Converged at iteration %s" % (i))
                break;
            if log_summaries:
                res = sess.run([rmse_tr, rmse_val, cost, summary_op])
                summary_str = res[3]
                writer.add_summary(summary_str, i)
            else:
                res = sess.run([rmse_tr, rmse_val, cost])
            acc_tr = res[0]
            acc_val = res[1]
            cost_ev = res[2]
            print("Training RMSE at step %s: %s" % (i, acc_tr))
            print("Validation RMSE at step %s: %s" % (i, acc_val))
            diff = abs(cost_ev - last_cost)
            last_cost = cost_ev
        else:
            sess.run(train_step)

    finalTrain = rmse_tr.eval(session=sess)
    finalVal = rmse_val.eval(session=sess)
    finalW = W.eval(session=sess)
    finalH = H.eval(session=sess)
    sess.close()
    return finalTrain, finalVal, finalW, finalH

# Extracts user indices, item indices, rating values and number
# of ratings from the ratings triplets.
def extract_rating_info(ratings):
    rating_values = np.array(ratings[:,2], dtype=float32)
    user_indices = ratings[:,0]
    item_indices = ratings[:,1]
    num_ratings = len(item_indices)
    return rating_values, num_ratings, user_indices, item_indices

# Creates a trainable tensor representing either user or item bias,
# and a corresponding tensor of 1's for the other.
def create_factors_for_bias(num_users, num_items, lda, user_bias = True):
    if user_bias:
        # Random normal intialized column for users
        W = tf.Variable(tf.truncated_normal([num_users, 1], stddev=0.02, mean=0), name="users")
        # Row of 1's for items
        H = tf.ones((1, num_items), name="items")
        # Add regularization.
        regularizer = tf.mul(tf.reduce_sum(tf.square(W)), lda, name="regularize")
    else:
        # Column of 1's for users
        W = tf.ones((num_users, 1), name="users")
        # Random normal intialized row for items
        H = tf.Variable(tf.truncated_normal([1, num_items], stddev=0.02, mean=0), name="items")
        # Add regularization.
        regularizer = tf.mul(tf.reduce_sum(tf.square(H)), lda, name="regularize")
    return W, H, regularizer


# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn item bias on top of provided user bias.
def learn_item_bias_from_fixed_user_bias(ratings_tr, ratings_val, user_bias, num_items, lda, global_mean, max_iter):
    W = tf.concat(1, [tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((user_bias.shape[0],1), dtype=float32, name="item_bias_ones")])
    H = tf.Variable(tf.truncated_normal([1, num_items], stddev=0.02, mean=0), name="items")
    H_with_user_bias = tf.concat(0, [tf.ones((1, num_items), name="user_bias_ones", dtype=float32), H])
    regularizer = tf.mul(tf.reduce_sum(tf.square(H)), lda, name="regularize")
    return mf(ratings_tr, ratings_val, W, H_with_user_bias, regularizer, global_mean, max_iter, 0.8)

# Learns factors of the given rank with specified regularization parameter.
def create_factors_without_biases(num_users, num_items, rank, lda):
    # Initialize the matrix factors from random normals with mean 0. W will
    # represent users and H will represent items.
    W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=0), name="users")
    H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=0), name="items")
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    return W, H, regularizer

# Given previously learned user bias and item bias vectors, creates
# tensors to learn factors of the given rank (excluding the bias vectors)
# and a regularizer.
def create_factors_with_biases(user_bias, item_bias, rank, lda):
    num_users = user_bias.shape[0]
    num_items = item_bias.shape[1]
    # Initialize the matrix factors from random normals with mean 0. W will
    # represent users and H will represent items.
    W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=0), name="users")
    H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=0), name="items")

    # To the user matrix we add a bias column holding the bias of each user,
    # and another column of 1s to multiply the item bias by.
    W_plus_bias = tf.concat(1, [W, tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((num_users,1), dtype=float32, name="item_bias_ones")])
    # To the item matrix we add a row of 1s to multiply the user bias by, and
    # a bias row holding the bias of each item.
    H_plus_bias = tf.concat(0, [H, tf.ones((1, num_items), name="user_bias_ones", dtype=float32), tf.convert_to_tensor(item_bias, dtype=float32, name="item_bias")])
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    return W_plus_bias, H_plus_bias, regularizer


# Uses k-fold cross-validation to learn the best regularization
# parameter to use for either user or item bias.
def learn_bias_lda(ratings, num_folds, ldas, num_users, num_items, global_mean, max_iter, user_bias = True):
    labels = ratings[:,2]
    skf = StratifiedKFold(labels, num_folds)
    min_lda = None
    min_rmse = 0
    for lda in ldas:
        sum_rmses = 0
        W, H, reg = create_factors_for_bias(num_users, num_items, lda, user_bias)
        for train, test in skf:
            tr, val, finalw, finalh = mf(ratings[train,:], ratings[test,:], W, H, reg, global_mean, max_iter, 0.8)
            sum_rmses += val
            print("Training rmse: %s, val rmse: %s, lda: %s" % (tr, val, lda))
        avg_rmse = sum_rmses / num_folds
        if min_lda == None:
            # This is our first lambda.
            min_lda = lda
            min_rmse = avg_rmse
        elif avg_rmse < min_rmse:
            # We did better than the last lambda.
            min_rmse = avg_rmse
            min_lda = lda
        else:
            # It's not going to get any better with the next lambda.
            break
    return min_lda

# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn user bias from the training set.
def get_user_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter):
    W, H, reg = create_factors_for_bias(num_users, num_items, lda, True)
    tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 0.8)
    return finalw

# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn item bias from the training set.
def get_item_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter):
    W, H, reg = create_factors_for_bias(num_users, num_items, lda, False)
    tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 0.8)
    return finalh

def main():
    path = os.path.expanduser(sys.argv[1])
    ratings_df = feather.read_dataframe(path)
    num_ratings = ratings_df.shape[0]
    ratings = np.concatenate((np.array(ratings_df['user_id'], dtype=pd.Series).reshape(num_ratings, 1), np.array(ratings_df['item_id'], dtype=pd.Series).reshape(num_ratings, 1), np.array(ratings_df['rating'], dtype=pd.Series).reshape(num_ratings, 1)), axis=1)
    global_mean = mean(ratings[:,2])
    np.random.seed(12)
    ratings_tr, ratings_val = train_test_split(ratings, train_size=.7)
    max_iter = int(sys.argv[2])
    to_learn = sys.argv[3]
    num_users = np.unique(ratings[:,0]).shape[0]
    num_items = np.unique(ratings[:,1]).shape[0]
    if to_learn == "user_bias_lda":
        lda = learn_bias_lda(ratings_tr, 4, [2,4,6,8,10], num_users, num_items, global_mean, max_iter)
        print("Best lambda for user bias is %s" %(lda))
    elif to_learn == "item_bias_lda":
        lda = learn_bias_lda(ratings_tr, 4, [2,4,6,8,10], num_users, num_items, global_mean, max_iter, False)
        print("Best lambda for item bias is %s" %(lda))
    elif to_learn == "user_bias":
        lda = float(sys.argv[4])
        user_bias = get_user_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter)
        np.save("user_bias", user_bias)
    elif to_learn == "item_bias":
        lda = float(sys.argv[4])
        item_bias = get_item_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter)
        np.save("item_bias", item_bias)
    elif to_learn == "item_bias_fixed_user":
        lda = float(sys.argv[4])
        user_bias = np.load("user_bias.npy")
        tr, val, finalw, finalh = learn_item_bias_from_fixed_user_bias(ratings_tr, ratings_val, np.load("user_bias.npy"), num_items, lda, global_mean, max_iter)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("item_bias_fixed_user", finalh[1,:].reshape(num_items,))
    elif to_learn == "features":
        lda = float(sys.argv[4])
        rank = int(sys.argv[5])
        user_bias = np.load("user_bias.npy").reshape(num_users, 1)
        item_bias = np.load("item_bias.npy").reshape(1, num_items)
        W, H, reg = create_factors_with_biases(user_bias, item_bias, rank, lda)
        tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 1.0, True)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("final_w", finalw)
        np.save("final_h", finalh)
    elif to_learn == "features-only":
        lda = float(sys.argv[4])
        rank = int(sys.argv[5])
        W, H, reg = create_factors_without_biases(num_users, num_items, rank, lda)
        tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 1.0, True)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("final_w", finalw)
        np.save("final_h", finalh)

if __name__ == '__main__':
    main()