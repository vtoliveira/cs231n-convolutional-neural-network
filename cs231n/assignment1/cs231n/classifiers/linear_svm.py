import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) # it calculates all scores for ith training point, since the result has N x C, in this case 1 x C for each i.
    correct_class_score = scores[y[i]] # it will save the score value for the correct class, lets say the indice 1 has score 0.35
    for j in range(num_classes): # it iterates for all classes C, and if j is not the correct class, computes the loss
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score+1 # note delta = 1 # delta is the margin, how certain we want to be
      if margin > 0: # we only add to loss function if margin > 0, because if it is less than zero, means that we have the correct class
        loss += margin
        dW[:, j] += X[i]  # for jth class, we add X[i]
        dW[:, y[i]] -= X[i] # for y[i] class, we subtract X[i] => this happens since if margin is zero, the row y[i] is the one with template
                            # for the correct class, then for the gradient if margin is higher than zero, we have to decrease by X[i] times the
                            # number of wrong classes accordingly to the gradient

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X @ W
  correct_class_scores = y
  margin = scores - scores[np.arange(scores.shape[0]),correct_class_scores][:, np.newaxis] + 1
  margin = margin[np.arange(margin.shape[0]), correct_class_scores] = 0
  margin = margin[margin>0]
  loss += np.sum(margin)
  
  loss /= num_train
  loss += reg * np.sum(W*W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
