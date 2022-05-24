import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None

    dot = z_i.dot(z_j.T)
    z_i_norm = torch.linalg.norm(z_i)
    z_j_norm = torch.linalg.norm(z_j)

    norm_dot_product = dot/(z_j_norm * z_i_norm)

    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0.0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #

        pos_dist1 = torch.exp(sim(z_k,z_k_N) / tau)
        pos_dist2 = torch.exp(sim(z_k_N,z_k) / tau)

        neg_dist1 = torch.tensor(0.0)
        neg_dist2 = torch.tensor(0.0)

        for i in range(2 * N):
            # Compute the distance_1 (z_k,z_k_N)
            if i == k:
                pass
            else:
                neg_dist1 += torch.exp(sim(z_k,out[i])/ tau)

            # Compute the distance_2 (z_k_N,z_k)
            if i == (k + N):
                pass
            else:
                neg_dist2 += torch.exp(sim(z_k_N,out[i])/ tau)

        # Compute two loss
        pos_loss = - torch.log(pos_dist1 / neg_dist1 )
        neg_loss = - torch.log(pos_dist2 / neg_dist2 )
        total_loss += pos_loss + neg_loss

        # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    
    out_left_norm  = torch.linalg.norm(out_left,dim =1)
    out_right_norm = torch.linalg.norm(out_right,dim=1)
    out_norm = out_left_norm * out_right_norm

    pos_pairs = (torch.diagonal(out_left.matmul(out_right.T)) / out_norm).unsqueeze(1)
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """

    out_norm1 = torch.linalg.norm(out,dim = 1,keepdim = True)
    norm = out_norm1.mm(out_norm1.T)

    sim_matrix = out.matmul(out.T) / norm

    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.

    exponential = torch.exp(sim_matrix / tau).to(device)

    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential,device = device) - torch.eye(2 * N,device = device)).bool()

    # We apply the binary mask.
    exponential = torch.masked_select(exponential,mask).view((2*N,-1))    # shape (2*N,2*N-1)
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential,dim = 1,keepdim = True)

    # Step 2: Compute the numerator value for all augmented samples.

    pos_pairs = sim_positive_pairs(out_left,out_right)
    pos_dist  = torch.exp(pos_pairs / tau).to(device)
    pos_dist = torch.cat([pos_dist,pos_dist],dim = 0)

    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.

    loss = torch.sum(-torch.log(pos_dist / denom)) / (2*N)

    return loss

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))