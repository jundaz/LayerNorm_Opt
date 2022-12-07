import numpy as np
import torch
from IPython import display
from torch import nn
import collections
import math
import random


def app_sigma(size):
    # using equation given in  getGaussianKernel() on page:
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    return 0.3*((size-1)*0.5 - 1) + 0.8


def gaussian_2d(size, sigma=None):
    # using equation given in  getGaussianKernel() on page:
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    a = []
    if sigma is None:
        sigma = app_sigma(size)
    down = 2 * sigma ** 2
    for i in range(size):
        up = -(i - (size - 1)/2) ** 2
        a.append(np.exp(up/down))
    gaussian_1d = (np.array(a)/np.sum(np.array(a))).reshape(size, 1)
    return np.multiply(gaussian_1d.T, gaussian_1d)


# helper function by towaki
def normalized_grid(width, height):
    """Returns grid[x,y] -> coordinates for a normalized window.

    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = np.linspace(-1, 1, num=width) * (width / height)
    window_x += np.random.rand(*window_x.shape) * (1. / width)
    window_y = np.linspace(1, -1, num=height)
    window_y += np.random.rand(*window_y.shape) * (1. / height)
    coord = np.array(np.meshgrid(window_x, window_y, indexing='xy')).transpose(2, 1, 0)

    return coord


def my_normalized_grid(width, height, start, end):
    x_dim = torch.linspace(start, end, width)
    y_dim = torch.linspace(start, end, height)
    meshx, meshy = torch.meshgrid((x_dim, y_dim))
    return torch.stack((meshx, meshy), 2)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


def get_accuracy(model, dataloader, device='cuda:0'):
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)  ## <---
            y = y.to(device)  ## <---
            prediction = model(x).argmax(dim=-1, keepdim=True)
            correct += prediction.eq(y.view_as(prediction)).sum().item()
    return correct / len(dataloader.dataset)


def set_random_seeds(seed_value=0, device='cuda:0'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    sum_list = []
    # for p in params:
    #     if p.grad is not None:
    #         sum_list.append(torch.sum(p.grad ** 2))
    # norm = torch.sqrt(sum(sum_list))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score





