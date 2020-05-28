"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
from .linear_classifier import LinearClassifier



def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    #set parameters
    N = X.shape[0]      # # N examples
    C = W.shape[1]      # # C classes 
    D = X.shape[1]      # # D features(Domension)
    W = W.T
    X = X.T
    y = y.T
    
    # Initialization 
    y_hat = np.zeros((C,N))
    z = np.zeros((C,N))
    dz = np.zeros(z.shape)
    dW = np.zeros(W.shape)
    z_sum = 0
    L =  np.zeros((N,))
    J = 0
    
    # set input for softmax
    for c in range(C):
        for n in range(N):
            for d in range(D):
                z[c,n] += W[c,d]* X[d,n]        # z = input of activation(softmax)
#   print("naive = " + str(z[1,1:20]))     
    #softmax
    z_exp = np.zeros(z.shape)
    ovfl = np.max(z)     # overflow-prevent term
    for n in range(N):
        z_sum = 0
        for c in range(C):        
            z_exp[c,n] = np.exp(z[c,n]-ovfl)
            z_sum = z_sum +  z_exp[c,n]
        for c in range(C): 
            y_hat[c,n] = z_exp[c,n] /  z_sum       
    #loss fuction L
#     print(y_hat[1,1:20])
    for n in range(N):
        for c in range(C):       
            if(c == y[n,]):
                L[n] = -np.log(y_hat[c,n])  # N sample's L
    #cost function J --sum of L of N samples
    for n in range(N):
        J += L[n]
    J = J/N
    loss = J + reg
        
    #gradient dw
    for n in range(N):
        for c in range(C):
            if(c == y[n]):
                dz[c,n] = y_hat[c,n]-1
            else:
                dz[c,n] = y_hat[c,n]
                    
    for c in range(C):
        for d in range(D):
            for n in range(N):
                dW[c,d] += dz[c,n] * X.T[n,d]/N
  
    for c in range(C):
        for d in range(D):
             dW[c,d] += reg * W[c,d]
    
    dW = dW.T

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
#     set parameters
    N = X.shape[0]      # # N examples
    C = W.shape[1]      # # C classes 
    D = X.shape[1]      # # D features(Domension)
    X = X.T
    y = y.T
    
    # Initialization 
    dZ = np.zeros((C,N))    
    # set input for softmax
    z = np.dot(W.T,X)                             
    #softmax
    ovfl = np.max(z, axis=0, keepdims=True)     # overflow-prevent term
#     print("ovfl =" +str(ovfl.shape))
    z_exp = np.exp(z-ovfl)
    z_sum = np.sum(z_exp, axis = 0, keepdims = True)
#     print("z_exp =" +str(z_exp.shape))
#     print("z_sum.shape =" +str(z_sum.shape))
    y_hat = z_exp/z_sum      # y_hat = output of softmax
#     print("y_hat =" +str(y_hat.shape))
#     print(y_hat[1,1:20])

    #cross entropy/cost func batch  
    J = (-1 / N ) * np.sum(np.log(y_hat[y,np.arange(N)])) 
#     print("J =" +str(J))
    loss = J + 0.5 * reg * np.sum(W*W)
#     print(" loss =" +str( loss))
                        
    #gradient
    dZ[:] = y_hat
    dZ[y,np.arange(N)] = y_hat[y,np.arange(N)] - 1
    dW = (1/N) * ( np.dot(X,dZ.T))
    dW += reg*W
                
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    #This is inheritance
    #SoftmaxClassifier is child class. LinearClassifier is parent class
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4, 6e3]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    for lr in learning_rates:
        for rs in regularization_strengths:
            softmax = SoftmaxClassifier()
            loss = softmax.train(X_train, y_train, learning_rate=lr, reg=rs,
                          num_iters=1500, verbose=True)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            val_accuracy = np.mean(y_val == y_val_pred)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_softmax = softmax
            results[(lr, rs)] = (np.mean(y_train == y_train_pred), val_accuracy)
            all_classifiers.append(softmax)
            
            
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
