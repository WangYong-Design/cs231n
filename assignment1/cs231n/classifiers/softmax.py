from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train,D = X.shape
    _,C = W.shape
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    scores = X.dot(W)

    scores = (scores.T - np.max(scores,axis = 1)).T
    exp_normalized_scores = np.exp(scores)
    sum = np.sum(exp_normalized_scores, axis=1)

    # Compute the loss
    for i in range(num_train):
        s_yi = exp_normalized_scores[i, y[i]]
        loss -= np.log(s_yi * 1.0 / sum[i])

    loss /= num_train
    loss += reg * np.sum(np.square(W))

    # Compute the dW
    for i in range(num_train):
        sum_i = sum[i]
        for j in range(C):
            dW[:, j] += (exp_normalized_scores[i][j] * 1.0 / sum_i) * X[i].T
            if j == y[i]:
                dW[:, y[i]] -= X[i]

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]

    scores = X.dot(W)
    norm_scores  = (scores.T - scores.max(axis = 1)).T
    exp_norm_scores = np.exp(norm_scores)
    sum = np.sum(exp_norm_scores,axis = 1)

    # Compute the loss
    probs = ((exp_norm_scores *1.0).T /sum).T
    loss = np.sum(-np.log(probs[range(N),y]))
    loss /= N
    loss += reg * np.sum(np.square(W))

    # Compute the dW
    dz = probs
    dz[range(N),y] -= 1
    dW = X.T.dot(dz)
    dW /=N
    dW += 2*reg*W

    return loss, dW
