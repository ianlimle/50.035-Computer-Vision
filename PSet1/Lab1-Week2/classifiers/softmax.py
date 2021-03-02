import numpy as np
from random import shuffle

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
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        # avoid numerical instability
        f = X.dot(W)[i] - np.max(X.dot(W)[i])
        norm_exp_f = np.exp(f)/sum(np.exp(f))
        loss_i = -np.log(norm_exp_f[y[i]])
        loss += loss_i
        
        for j in range(num_classes):
            # increment gradient for all classes for a training example 
            dW[:, j] += X[i] * norm_exp_f[j]
        # subtract value of all dimensions from the ground-truth class for a training example
        dW[:, y[i]] -= X[i]
        
    loss /= num_train
    # apply regularization to loss
    loss += reg * np.sum(W*W)
    dW /= num_train
    # apply regularization to the gradient
    dW += reg * 2 * W    
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
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
    num_train = X.shape[0]
    scores = X.dot(W) - np.max(X.dot(W), axis=1, keepdims=True)

    # softmax loss
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )

    # weight gradient
    softmax_matrix[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax_matrix)
    
    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW += reg * 2 * W 

    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

