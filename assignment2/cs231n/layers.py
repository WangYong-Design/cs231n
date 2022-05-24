from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    X = np.reshape(x,(x.shape[0],-1))
    scores = X.dot(w) + b

    cache = (x, w, b)
    return scores, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #

    N = x.shape[0]
    x_strech = np.reshape(x,(x.shape[0],-1))

    # XWpb = XW + b
    db = np.ones(N).dot(dout)
    dxw = dout

    # XW = X*W
    dw = x_strech.T.dot(dxw)
    dx = dxw.dot(w.T).reshape(x.shape)

    return dx, dw, db

def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    mask = x > 0
    dx = dout * mask
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    N = x.shape[0]
    norm_scores = (x.T - x.max(axis = 1)).T
    exp_norm_socres = np.exp(norm_scores)
    probs = ((exp_norm_socres.T * 1.0)/exp_norm_socres.sum(axis = 1)).T

    loss = np.mean(-np.log(probs[range(N),y]))

    dx = probs
    dx[range(N),y] -= 1
    dx /= N

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #

        sample_mean = x.mean(axis = 0)
        sample_var = x.var(axis = 0)
        x_center = x - sample_mean
        x_norm = x_center / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x_center,x_norm,sample_var,gamma,eps)

    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_norm = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #

    # Unpack the cache
    x_center, x_norm, var_batch, gamma, eps = cache
    N, D = dout.shape

    # NOTE:
    # Get diagonal() for grad(sigma_{B}^{2}), grad(mu_{B}) and grad(gamma)  !!!
    # They are different from grad(beta)
    #   The reason may be that beta is just a bias connected to a add gate but not multiply gate in computation flow ?
    dx_norm = dout * gamma
    dvar_batch = -0.5 * ((x_norm / (var_batch + eps)).T.dot(dx_norm)).diagonal()
    dx_center = (2.0 / N) * x_center * dvar_batch + dx_norm / np.sqrt(var_batch + eps)
    dmean_batch = np.ones((D, N)).dot(-1.0 * dx_center).diagonal()

    dx = dx_center + (1.0 / N) * dmean_batch
    dgamma = dout.T.dot(x_norm).diagonal()
    dbeta = np.ones(N).dot(dout)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#

    # I cannot simplify my original version... orz
    # so I just try to make the code more neat
    # Unpack the cache
    x_center, x_norm, var_batch, gamma, eps = cache
    N, D = dout.shape

    var_batch += eps
    dx_norm = dout * gamma
    dvar_batch = (-0.5) * dx_norm.T.dot(x_norm / var_batch).diagonal()
    dx_center = (2.0 / N) * x_center * dvar_batch + dx_norm / np.sqrt(var_batch)
    dmean_batch = np.ones((D, N)).dot(-1.0 * dx_center).diagonal()

    dx = dx_center + (1.0 / N) * dmean_batch
    dgamma = dout.T.dot(x_norm).diagonal()
    dbeta = np.ones(N).dot(dout)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = x.T  # transfer x from (N, D) to (D, N)

    mean_batch = x.mean(axis=0)
    var_batch = x.var(axis=0)

    x_center = x - mean_batch
    x_norm = x_center / np.sqrt(var_batch + eps)
    out = x_norm.T * gamma + beta  # note that `beta` is added to features
    cache = (x_center, x_norm, var_batch, gamma, eps)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = (np.random.rand(*x.shape) < p )

        out = x * mask / p

    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = (dout * mask) / dropout_param["p"]

    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    w_ori = w
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N,C,H,W = x.shape
    F,_,HH,WW = w.shape

    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # HO = (H - HH) // stride + 1
    # WO = (W - WW) // stride + 1
    HO = 1 + (H + 2 * pad - HH) // stride
    WO = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values = (0,0))
    # print(x_pad.shape)
    x_stretch  = np.zeros((N,C,HH*WW,HO*WO))

    for y in range(HO):
        y_start = y * stride
        y_end = y_start + HH
        for x_idx in range(WO):
            x_start = x_idx * stride
            x_end = x_start + WW
            # print(x_pad[:, :, y_start:y_end, x_start:x_end].shape)
            x_stretch[:,:,0:,y*WO+x_idx] = x_pad[:,:,y_start:y_end,x_start:x_end].reshape(N,C,HH*WW)

    x_stretch = x_stretch.reshape(N,-1,HO*WO)
    w = w.reshape(F,C*HH*WW)

    out = (w.dot(x_stretch).transpose((1,0,2))+ b.reshape(F,1)).reshape(N,F,HO,WO)

    cache = (x, w_ori, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. Unpack the cache and get size of element
    x, w, b, conv_param = cache

    N, F, H_out, W_out = dout.shape  # `dout` shape: (N, F, H', W')
    _, C, H, W = x.shape  # `x` shape: (N, C, H, W)
    _, _, HH, WW = w.shape  # `w` shape: (F, C, HH, WW)
    stride = conv_param['stride']
    pad = conv_param['pad']

    # 2. Initialize and compute gradient of `b`
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.sum(dout, axis=(0, 2, 3))  # sum up dout for each filter

    # 3. Compute gradient of `w`
    # 3.1 "fast" version 2 (vectorized) -- just treat each sample as channel and perform conv_forward_naive():
    #                                      THE KERNEL USED IN CONVOLUTION IS `dout_pad_t`, see below for more details
    #     - time consume drop from 100s to 1.3s, together with
    #       "fast" version of `dx` computation and "fast" version of conv_forward_navie()

    #     (1) pad `dout` inside to fit the dimension of `dw`
    #         H_dout_pad should be (H'-1)*(S-1) + H', analogously W_dou_pad should be (W'-1)*(S-1) + W'
    dout_pad = np.zeros((N, F, (H_out - 1) * stride + 1, (W_out - 1) * stride + 1))
    dout_pad[:, :, ::stride, ::stride] = dout

    #     (2) perform conv_forward_naive()
    x_t = x.transpose((1, 0, 2, 3))  #
    dout_pad_t = dout_pad.transpose((1, 0, 2, 3))  # THE KERNEL USED IN CONVOLUTION IS `dout_pad_t`,
    # This kernel treat samples as channels !!!!
    b = np.zeros(F)
    param = {'stride': 1, 'pad': pad}  # Stride equals 1 and the same pad used in forward is needed

    dw_t, _ = conv_forward_naive(x_t, dout_pad_t, b, param)  # INPUT   `x_t` shape: (C, N, H_pad, W_pad)
    # INPUT   `dout_pad_t` shape: (F, N, H'_inpad, W'_inpad)
    # INPUT   `b` shape: (F,)
    # OUTPUT   `dw_t` shape: (C, F, HH, WW)
    dw = dw_t.transpose((1, 0, 2, 3))  # DON'T FORGET TO TRANSPOSE `dw_t` (no pad to remove !!!!)

    """

    # 3.1 "fast" version 1 (not vectorized): 
    #     - time consume drop from 100s to 0.8s, together with 
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')
    _, _, H_pad, W_pad = x_pad.shape
    for f in range(F):
        for c in range(C):
            for k in range(HH):
                for l in range(WW):
                    dw[f, c, k, l] = np.sum(x_pad[:, c, k:(k+H_pad-HH+1):stride, l:(l+W_pad-WW+1):stride] * dout[:, f, :, :])


    # 3.1 "clear" version (but much slower):
    for i in range(N):
        for f in range(F):
            for c in range(C):
                for k in range(HH):
                    for l in range(WW):
                        dw[f, c, k, l] += np.sum(x_pad[i, c, k:(k+H_pad-HH+1):stride, l:(l+W_pad-WW+1):stride] * dout[i, f])

    """

    # 4. Compute gradient of `x`
    # 4.1 pad `dout` both inside and outside to fit the dimension of `dx`
    #     H_dout_pad should be (H'-1)*(S-1) + H' + 2*(HH-1), analogously W_dou_pad should be (W'-1)*(S-1) + W' + 2*(HH-1)
    #     the inside padding is already done before `dw` computation, so I just need to pad the edge by 2*(HH-1) and 2*(WW-1)

    dout_pad = np.pad(dout_pad, ((0,), (0,), (HH - 1,), (WW - 1,)), 'constant')

    # 4.2 "fast" version (vectorized) -- just treat each filter as channel, rotate `w` and perform conv_forward_naive():
    #                                    THE KERNEL USED IN CONVOLUTION IS `w_rot`, see below for more details
    #     - time consume drop from 100s to 0.8s, together with
    #       "fast" version 1 of `dw` computation and "fast" version of conv_forward_navie()
    #     - time consume drop from 100s to 0.5s, together with
    #       "fast" version 2 of `dw` computation and "fast" version of conv_forward_navie()
    #       but still much slower than the provided backward function in `fast_layer.py` (0.02s)
    w_rot = np.rot90(w, k=2, axes=(2, 3)).transpose((1, 0, 2, 3))  # THE KERNEL USED IN CONVOLUTION IS `w_rot`
    # This kernel treats filters as channels !!!!
    b = np.zeros(C)
    param = {'stride': 1, 'pad': 0}  # Stride equals 1 and no pad needed

    dx_pad, _ = conv_forward_naive(dout_pad, w_rot, b,
                                   param)  # INPUT   `dout_pad` shape: (N, F, H'_inpad_pad, W'_inpad_pad)
    # INPUT   `w_rot` shape: (C, F, HH, WW)
    # INPUT   `b` shape: (C,)
    # OUTPUT  `dx_pad` shape: (N, C, H_pad, W_pad)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]  # DON'T FORGET TO REMOVE THE PAD

    """

    # 4.2 "clear" version (but much slower):
    param = {'stride': 1, 'pad': 0}
    dout_pad_slice = np.zeros((1, 1, dout_pad.shape[2], dout_pad.shape[3]))
    w_rot_slice = np.zeros((1, 1, HH, WW))
    b = np.zeros(1)
    for i in range(N):
        for f in range(F):
            for c in range(C):
                dout_pad_slice[:, :] = dout_pad[i, f]
                w_rot_slice[:, :] = np.rot90(w[f, c], k=2)
                dx_acc, _ = conv_forward_naive(dout_pad_slice, w_rot_slice, b, param)
                dx[i, c] += dx_acc[0, 0, pad:-pad, pad:-pad]

    """
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    N,C,H,W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    HO = (H - pool_height) // stride + 1
    WO = (W - pool_width) // stride + 1

    out = np.zeros((N,C,HO,WO))
    for y_idx in range(HO):
        y_start = y_idx * stride
        y_end = y_start + pool_height
        for x_idx in range(WO):
            x_start = x_idx * stride
            x_end = x_start + pool_width
            out[:,:,y_idx,x_idx] = np.max(x[:,:,y_start:y_end,x_start:x_end],axis = (2,3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x,pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    dx = np.zeros(x.shape)
    HO = (H - pool_height) // stride + 1
    WO = (W - pool_width) // stride + 1
    for i in range(N):
        for c in range(C):
            for y_idx in range(HO):
                y_start = y_idx * stride
                y_end = y_start + pool_height
                for x_idx in range(WO):
                    x_start = x_idx * stride
                    x_end= x_start + pool_width
                    a,b= np.where(np.max(x[i,c,y_start:y_end,x_start:x_end]) == x[i,c,y_start:y_end,x_start:x_end])
                    dx[i,c, y_start:y_end, x_start:x_end][a,b] =  dout[i,c,y_idx,x_idx]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #

    N,C,H,W = x.shape
    X_stretch = x.transpose((0,2,3,1)).reshape((-1,C))
    X_norm,cache = batchnorm_forward(X_stretch,gamma,beta,bn_param)
    out = X_norm.reshape((N,H,W,C)).transpose((0,3,1,2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N,C,H,W = dout.shape
    dout_stretch = dout.transpose((0,2,3,1)).reshape((-1,C))
    dx_stretch,dgamma,dbeta = batchnorm_backward(dout_stretch,cache)
    dx = dx_stretch.reshape((N,H,W,C)).transpose((0,3,1,2))

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
