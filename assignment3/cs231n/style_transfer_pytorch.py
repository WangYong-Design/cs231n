import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL

import matplotlib.pyplot as plt
import numpy as np

from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

# dtype = torch.FloatTensor
# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
dtype = torch.cuda.FloatTensor

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    _,C,H,W = content_original.shape
    content_current = content_current[0].reshape((C,-1))
    content_original = content_original[0].reshape((C,-1))
    loss = content_weight * torch.sum((content_current - content_original)**2)

    return loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """

    N,C,H,W = features.shape
    features = features.view((N,C,-1))

    gram = torch.bmm(features,features.permute(0,2,1))

    if normalize:
        gram /= H*W*C

    return gram


# Now put it together in the style_loss function...
def style_loss(feats,c_feats, style_layers, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - c_feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - feats: list of the features at every layer of the source image, as produced by
      the extract_features function.
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.

    style_loss = torch.tensor(0).type(dtype)
    for i in range(len(style_layers)):
        idx = style_layers[i]
        style_target = gram_matrix(feats[idx].clone())
        style_gen = gram_matrix(c_feats[idx].clone())
        style_loss += style_weights[i] * torch.sum((style_target - style_gen)**2)

    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    img_vert1 = img[:,:,1:,:]
    img_vert2 = img[:,:,:-1,:]
    img_hori1 = img[:,:,:, 1:].permute((0,1,3,2))
    img_hori2 = img[:,:,:,:-1].permute((0,1,3,2))

    mat_vert = difference_vertically(img_vert1,img_vert2)
    mat_hori = difference_vertically(img_hori1,img_hori2)

    return tv_weight * (mat_vert + mat_hori)

def difference_vertically(img_1,img_2):
    """
    Copmute difference vertically

    Input:
    - img_1: PyTorch tensor matrix.have shape (N,C,H-1,W)
    - img_2: PyTorch tensor matrix.have shape (N,C,H-1,W)

    Return:
    - grad_vert : PyTorch tensor marix.have shape (N,C,H-1,W)
    """
    img_sq1 = torch.sum(img_1 ** 2)
    img_sq2 = torch.sum(img_2 ** 2)
    mat_vari = torch.sum(-2 * torch.matmul(img_1,img_2.permute((0,1,3,2))).
                         diagonal(dim1 = 2,dim2 = 3))

    return img_sq1 + img_sq2 + mat_vari


def style_transfer(cnn,content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random=False):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size).type(dtype)
    feats = extract_features(content_img, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size).type(dtype)
    feats = extract_features(style_img, cnn)

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img.requires_grad_()

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img], lr=initial_lr)

    f, axarr = plt.subplots(1, 2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style_img.cpu()))
    plt.show()
    plt.figure()

    for t in range(1000):
        if t < 190:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        c_feats = extract_features(img, cnn)

        # Compute loss
        c_loss = content_loss(content_weight, c_feats[content_layer], content_target)
        s_loss = style_loss(feats, c_feats, style_layers, style_weights)
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss

        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            plt.imshow(deprocess(img.data.cpu()))
            plt.show()
    print('Iteration {}'.format(t))
    plt.axis('off')
    plt.imshow(deprocess(img.data.cpu()))
    plt.show()

def preprocess(img, size=512):
    """ Preprocesses a PIL JPG Image object to become a Pytorch tensor
        that is ready to be used as an input into the CNN model.
        Preprocessing steps:
            1) Resize the image (preserving aspect ratio) until the shortest side is of length `size`.
            2) Convert the PIL Image to a Pytorch Tensor.
            3) Normalize the mean of the image pixel values to be SqueezeNet's expected mean, and
                 the standard deviation to be SqueezeNet's expected std dev.
            4) Add a batch dimension in the first position of the tensor: aka, a tensor of shape
                 (H, W, C) will become -> (1, H, W, C).
    """
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    """ De-processes a Pytorch tensor from the output of the CNN model to become
        a PIL JPG Image that we can display, save, etc.
        De-processing steps:
            1) Remove the batch dimension at the first position by accessing the slice at index 0.
                 A tensor of dims (1, H, W, C) will become -> (H, W, C).
            2) Normalize the standard deviation: multiply each channel of the output tensor by 1/s,
                 scaling the elements back to before scaling by SqueezeNet's standard devs.
                 No change to the mean.
            3) Normalize the mean: subtract the mean (hence the -m) from each channel of the output tensor,
                 centering the elements back to before centering on SqueezeNet's input mean.
                 No change to the std dev.
            4) Rescale all the values in the tensor so that they lie in the interval [0, 1] to prepare for
                 transforming it into image pixel values.
            5) Convert the Pytorch Tensor to a PIL Image.
    """
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    """ A function used internally inside `deprocess`.
        Rescale elements of x linearly to be in the interval [0, 1]
        with the minimum element(s) mapped to 0, and the maximum element(s)
        mapped to 1.
    """
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

#please disregard warnings about initialization
def features_from_img(imgpath, imgsize, cnn):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    return extract_features(img_var, cnn), img_var




