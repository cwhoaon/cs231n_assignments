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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        score = X[i].dot(W)
        exp_score = np.exp(score)
        normalize_score = exp_score[y[i]] / np.sum(exp_score)
        cur_loss = -np.log(normalize_score)
        loss += cur_loss

        log_grad = -1/(normalize_score)
        normalize_grad = np.full((num_class), - exp_score[y[i]] / (np.sum(exp_score)**2)) 
        normalize_grad[y[i]] = (np.sum(exp_score)-exp_score[y[i]]) / (np.sum(exp_score)**2)
        normalize_grad *= log_grad
        exp_grad = np.exp(score) * normalize_grad
        dot_grad = X[i].reshape(-1, 1).dot(exp_grad.reshape(1,-1))
        dW += dot_grad

    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    score = X.dot(W)
    exp_score = np.exp(score)
    normalize_score = exp_score[np.arange(num_train), y] / np.sum(exp_score, axis=1)
    log_score = - np.log(normalize_score)
    loss += np.mean(log_score)
    loss += reg * np.sum(W*W)

    log_grad = - 1 / normalize_score
    one_hot_encoded = np.eye(10)[y]*np.sum(exp_score, axis=1, keepdims=True)
    normalize_grad = (one_hot_encoded - exp_score[np.arange(num_train), y].reshape(-1, 1)) /  (np.sum(exp_score, axis=1, keepdims=True)**2)
    normalize_grad *= log_grad.reshape(-1, 1)
    exp_grad = np.exp(score) * normalize_grad
    dot_grad = X.T.dot(exp_grad)
    dW += dot_grad/num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
