import math
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

############# for learning ##################
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def weights_init(m):
    # https://discuss.pytorch.org/t/weight-initialization-with-a-custom-method-in-nn-sequential/24846
    # https://blog.snowhork.com/2018/11/pytorch-initialize-weight
    # for mm in range(len(m)):
    mm=0
    if type(m) == nn.Linear: # in ,nn.GRU
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.GRU:
        nn.init.xavier_normal_(m.weight_hh_l0)
        nn.init.xavier_normal_(m.weight_ih_l0)

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states

def sample_gauss(mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(std).add_(mean)

def batch_error(predict, true, Sum=True, sqrt=False, diff=True, index=None):#, normalize=False):
    # error = torch.sum(torch.sum((predict[:,:2] - true[:,:2]),1))
    # index = (true[:,0]>9998)
    if predict.shape[1] > 1:
        if sqrt:
            # error = torch.sqrt(error)
            if diff:
                error = torch.norm(torch.abs(predict-true)+1e-6,p=2,dim=1)
            else:
                error = torch.norm(predict.abs()+1e-6,p=2,dim=1)
        else:
            if diff:
                error = torch.sum((torch.abs(predict - true)+1e-6).pow(2),1) # [:,:2]
            else:
                error = torch.sum((predict.abs()+1e-6).pow(2),1) # [:,:2]
    else:
        error = torch.abs(predict - true) 
    
    if index is not None and torch.sum(index) < len(index):
        error[~index] = 0

    if Sum:
        error = torch.sum(error)
    return error

def std_ste(x,std=False):
    if std:
        return np.std(x)
    else:
        return np.std(x)/np.sqrt(len(x))
