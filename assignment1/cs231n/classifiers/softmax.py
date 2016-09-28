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
    num_class = W.shape[1]
    for i in xrange(num_train):
        scores = np.exp(X[i].dot(W))
        correct_score = scores[y[i]]
        scores_sum = np.sum(scores)

        loss += -np.log(correct_score / scores_sum)
        dw = -np.matmul(scores.reshape((-1, 1)), X[i].reshape((1, -1))) / (scores_sum * correct_score)
        for j in xrange(num_class):
            if j == y[i]:
                dw[j, :] *= (scores_sum - correct_score)
            else:
                dw[j, :] *= (0. - correct_score)

        dW += dw.T

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

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
    num_class = W.shape[1]
    mask = (np.arange(num_train), y)

    scores = np.exp(X.dot(W))
    scores_sum = np.sum(scores, axis=1).reshape((-1, 1))
    scores_correct = scores[mask].reshape((-1, 1))

    dsoft = -scores_sum / scores_correct
    tmp = np.zeros_like(scores)
    tmp[mask] = scores_sum[:, 0]
    dh1 = (tmp - scores_correct) / (scores_sum * scores_sum) * scores * dsoft

    dW = np.matmul(X.T, dh1) / num_train + reg * W

    loss = -np.mean(np.log(scores_correct / scores_sum)) + 0.5 * reg * np.sum(W * W)

    # params = np.zeros_like(S)
    # params[mask] = S_sum
    # params -= S_correct.reshape((-1, 1))
    # params /= np.reshape((S_sum * S_correct), (-1, 1))
    #
    # dW = -np.matmul(X.T, S * params) / num_train
    # dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
