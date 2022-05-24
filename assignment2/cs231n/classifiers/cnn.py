from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #

        C = input_dim[0]

        self.params["W1"] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
        self.params["b1"] = np.zeros(num_filters)
        W2_size = num_filters * np.prod(input_dim) / (self.pool_param["pool_height"]**2 * input_dim[0])
        self.params["W2"] = weight_scale * np.random.randn(int(W2_size),hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params["b3"] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = self.pool_param

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #

        layer_caches = {}
        Conv1,cache = conv_relu_pool_forward(X, self.params["W1"],self.params["b1"],conv_param,pool_param)
        layer_caches["Conv1"] = cache
        Affine1,cache = affine_relu_forward(Conv1,self.params["W2"],self.params["b2"])
        layer_caches["Affine1"] = cache
        scores,cache = affine_forward(Affine1,self.params["W3"],self.params["b3"])
        layer_caches["Affine2"] = cache

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #

        loss,dscores = softmax_loss(scores,y)
        loss += 0.5 * self.reg * (np.sum(self.params["W1"]**2) + np.sum(self.params["W2"]**2) +
                                  np.sum(self.params["W3"]**2))
        cache = layer_caches["Affine2"]
        dX,dW,db = affine_backward(dscores,cache)
        dW += self.reg * self.params["W3"]
        grads["W3"], grads["b3"] = dW,db
        del layer_caches["Affine2"]

        cache = layer_caches["Affine1"]
        dX,dW,db = affine_relu_backward(dX,cache)
        dW += self.reg * self.params["W2"]
        grads["W2"],grads["b2"] = dW,db
        del layer_caches["Affine1"]

        cache = layer_caches["Conv1"]
        dX,dW,db = conv_relu_pool_backward(dX,cache)
        dW += self.reg * self.params["W1"]
        grads["W1"], grads["b1"] = dW, db
        del layer_caches["Conv1"]

        return loss, grads
