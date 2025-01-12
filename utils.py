import pickle
import torch


import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
def visualize_binned_distribution(prob_tensor, title="Binned Probability Distribution"):
    """
    Visualizes a binned probability distribution represented as a PyTorch tensor.
    
    Args:
        prob_tensor (torch.Tensor): A 1D tensor representing the binned probabilities.
        title (str): The title of the plot (default: "Binned Probability Distribution").
    """
    if not isinstance(prob_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    if prob_tensor.dim() != 1:
        raise ValueError("Tensor must be 1-dimensional.")
    
    # Ensure the probabilities sum to 1
    if not torch.isclose(prob_tensor.sum(), torch.tensor(1.0), atol=1e-5):
        print("Warning: Tensor does not sum to 1. Normalizing...")
        prob_tensor = prob_tensor / prob_tensor.sum()

    # Create the x-axis bins
    bins = np.arange(len(prob_tensor))

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.bar(bins, prob_tensor.numpy(), color='skyblue', alpha=0.8, edgecolor='black')
    plt.xlabel("Bins")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(bins)  # Ensure all bins are labeled
    plt.tight_layout()
    plt.show()

def get_jacobian_norm(model, x, summary):

    # Compute the Jacobian by passing inputs as separate arguments
    jacobian = torch.autograd.functional.jacobian(model_with_inputs, (x, summary))

    # Compute the norm of the Jacobian
    jacobian_norm = sum(torch.norm(jacobian_part) for jacobian_part in jacobian)
    del x, summary
    return jacobian_norm

def get_distribution_entropy(tensor):
    return (-torch.sum(tensor * torch.log(tensor))).item()

def gaussian_noise(tensor, noise_level=5e-5):
    noise = torch.randn_like(tensor) * noise_level
    return tensor+noise

def noise_averaged_inference(model, data, summary, steps=3, device='cuda', noise_level=1e-6):
    avg = None

        #data = gaussian_noise(data, noise_level)
    price_preds = []
    for _model in model:
        price_preds.append(_model(data.to(device), summary.to(device)).squeeze(0))
        #price_pred = price_pred.squeeze(0)
    price_pred = sum(price_preds)/len(price_preds)

    return price_pred

def noise_averaged_inference_with_t_act(model, data, summary, steps=3, device='cuda', noise_level=1e-6):
    avg = None

        #data = gaussian_noise(data, noise_level)
    price_preds = []
    for _model in model:
        price_preds.append(_model.forward_with_t_act(data.to(device), summary.to(device)).squeeze(0))
        #price_pred = price_pred.squeeze(0)
    price_pred = sum(price_preds)/len(price_preds)

    return price_pred

def save_pickle(data, name):
    with open(name, 'wb') as out:
        pickle.dump(data, out)
    print(f'Saved {name}...')

def pic_load(pic):
    with open(pic, 'rb') as infile:
        return pickle.load(infile)

def cube_normalize(tensor, n_min=-1, n_max=1, dim=1):
    '''
    Takes a tensor and projects all elements onto a unit
    cube, size -1 to 1. 
    '''
    min, max = tensor.min(dim=dim, keepdim=True)[0], tensor.max(dim=dim, keepdim=True)[0]
    return (tensor-min)/(max-min)*(n_max-n_min) + n_min

def set_nan_inf(data):
    '''
    This eliminates any NaN or inf in a tensor by
    setting it to one, so downstream operations are
    minimally disrupted.
    '''
    if torch.isinf(data).any():
        inf_mask = torch.isinf(data)
        data[inf_mask] = 1
    if torch.isnan(data).any():
        nan_mask = torch.isnan(data)
        data[nan_mask] = 1
    return data

def get_abs_price_dim(price_seq):
    '''
    
    '''
    # Shape (batch, seq, dim)
    # 2.465073550667833 16111693133.838446 99981918.3880809 557781883.0235642
    a = 1
    _min, _max, mean, std = torch.tensor(2.465073550667833), torch.tensor(16111693133.838446), torch.tensor(99981918.3880809), torch.tensor(557781883.0235642)
    close = price_seq[:, :, -2]
    volume = price_seq[:, :, -1]
    price_volume = close*volume
    gauss_price_volume = (price_volume-mean)/std
    cube_price_volume = 2*(price_volume-_min)/(_max-_min)-1
    added_data = torch.cat((gauss_price_volume.unsqueeze(2), cube_price_volume.unsqueeze(2)), dim=2).to(torch.float32)
    return added_data

def _get_expected_price(pred, bin_edges, device='cuda', full_dist=True):
    """
    Computes the expected price from logits based on specified bin edges.

    Args:
        pred (Tensor): Logits for the probability distribution (num_bins,).
        bin_edges (Tensor): Bin edges that define the bins. Should be of size (num_bins + 1,).
        device (str): Device to run the computation (default='cuda').
        full_dist (bool): If True, assumes the pred tensor has all 4 rows.

    Returns:
        Tensor: Expected price.
    """
    bin_edges = torch.load('Bin_Edges_300')
    #softmax = torch.nn.Softmax(dim=0)
    
    # Compute the center value of each bin from the edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
    bin_centers = bin_centers.to(device)
    
    if full_dist:
        # Repeat bin centers for each row in pred
        bin_centers = bin_centers.repeat(pred.size(0), 1)
    
    # Apply softmax to logits and calculate the expected price
    #probabilities = softmax(pred)
    probabilities = pred
    expected_price = torch.sum(bin_centers * probabilities, dim=1 if full_dist else 0)
    
    return expected_price

def get_expected_price(pred, bin_edges, device='cuda'):
        '''
        Takes a vector of logits, corresponding to probability bins, and outputs a mean
        price prediction.
        '''
        device = pred.device
        reward_values = (bin_edges[:-1] + bin_edges[1:]) / 2
        reward_values = reward_values.to(device)
        expected_reward = torch.sum((reward_values*pred), dim=0)
        return expected_reward


def compute_loss_with_entropy(logits, targets, main_loss_fn, entropy_weight=0.01):
    """
    Compute the loss with an entropy regularization term.

    Args:
        logits (Tensor): Logits from the model (batch_size, num_classes).
        targets (Tensor): Ground-truth labels (batch_size,).
        main_loss_fn (callable): Main loss function (e.g., CrossEntropyLoss).
        entropy_weight (float): Weight of the entropy regularization term.

    Returns:
        Tensor: Total loss (main loss + entropy regularization).
    """
    # Main loss (e.g., CrossEntropyLoss)
    main_loss = main_loss_fn(logits, targets)

    # Compute probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)

    # Compute entropy: -sum(p * log(p)) across the class dimension
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1).mean()

    # Total loss: main loss + entropy regularization
    total_loss = main_loss - entropy_weight * entropy

    return total_loss

def compute_loss_with_gaussian_regularization(logits, targets, main_loss_fn, reg_weight=0.01, sigma=1.0):
    """
    Compute the loss with a Gaussian regularization term to encourage smoothness around neighboring bins.

    Args:
        logits (Tensor): Logits from the model (batch_size, num_classes).
        targets (Tensor): Ground-truth labels (batch_size,).
        main_loss_fn (callable): Main loss function (e.g., CrossEntropyLoss).
        reg_weight (float): Weight of the Gaussian regularization term.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        Tensor: Total loss (main loss + Gaussian regularization).
    """
    # Main loss (e.g., CrossEntropyLoss)
    main_loss = main_loss_fn(logits, targets)

    # Compute probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)

    # Create Gaussian kernels centered around the target bins
    num_classes = logits.size(-1)
    device = logits.device

    # One-hot encode targets and compute Gaussian kernel for each target
    #target_one_hot = F.one_hot(targets, num_classes).float().to(device)
    bin_indices = torch.arange(num_classes, device=device).view(1, -1)  # Shape: (1, num_classes)
    
    # Gaussian kernel for each target
    gaussians = torch.exp(-((bin_indices - targets.unsqueeze(1)) ** 2) / (2 * sigma ** 2))
    gaussians = gaussians / gaussians.sum(dim=-1, keepdim=True)  # Normalize to form a probability distribution

    # Compute Gaussian regularization loss
    gaussian_reg = torch.sum((probabilities - gaussians) ** 2, dim=-1).mean()

    # Total loss: main loss + Gaussian regularization
    total_loss = main_loss + reg_weight * gaussian_reg

    return total_loss

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import torch.nn as nn
def replace_square_linear_layers(module, activation_fn=None, exclude_types=(nn.MultiheadAttention,), perms=10):
    """
    Recursively replaces square linear layers in a module with Lin_Transpose,
    while avoiding specific module types.

    Args:
        module (nn.Module): The module to process.
        activation_fn (callable, optional): The activation function to use in Lin_Transpose.
        exclude_types (tuple, optional): Types of modules to exclude from replacement.

    Returns:
        nn.Module: The modified module.
    """
    for name, child in module.named_children():
        # Skip modules that should not be modified
        if isinstance(child, exclude_types):
            continue

        # Replace square nn.Linear layers
        #if isinstance(child, nn.Linear) and child.in_features == child.out_features:
        #    print(f"Replacing {name} in {module.__class__.__name__}")
        #    setattr(module, name, Lin_Transpose(child.in_features, bias=child.bias is not None, act=activation_fn))
        if isinstance(child, nn.Linear) and child.out_features != 1200:
            print(f"Replacing {name} in {module.__class__.__name__}")
            setattr(module, name, Rand_Lin_Perm(child.in_features, child.out_features, 
                                            perms=perms, act=activation_fn))
        
        # Recursively process child modules
        else:
            replace_square_linear_layers(child, activation_fn, exclude_types)

    return module

class Lin_Transpose(nn.Module):
    def __init__(self, width,bias=False,act=None,):
        super(Lin_Transpose, self).__init__()
        self.lin = nn.Linear(width,width,bias=bias)
        self.act = act
        #print(act)
    
    def forward(self,x):
        reg_out = self.lin(x)
        #print(self.lin.weight.data.shape, x.shape)
        t_out = x@torch.rot90(self.lin.weight,k=1) 
        t_out += x@torch.rot90(self.lin.weight,k=2)
        t_out += x@torch.rot90(self.lin.weight,k=3)
        t_out += x@self.lin.weight.t()
        t_out += x @ torch.flip(self.lin.weight, dims=[0])  # Swap rows
        t_out += x @ torch.flip(self.lin.weight, dims=[1])  # Swap columns
        t_out += x @ torch.flip(self.lin.weight.T, dims=[0, 1])
        t_out += x@torch.roll(self.lin.weight, shifts=1, dims=0)
        col_shifted = torch.roll(self.lin.weight, shifts=1, dims=1)
        t_out += x @ col_shifted

        
        if self.act is not None:
            reg_out = self.act(reg_out)
            t_out = self.act(t_out)
        out = reg_out + t_out
        return out

class Lin_Transpose_NonSquare(nn.Module):
    def __init__(self, in_features, out_features, bias=False, act=None, perms=10):
        super(Lin_Transpose_NonSquare, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = act
        self.num_shifts = perms
    
    def forward(self, x):
        # Regular linear operation
        batch, seq = x.shape[0], x.shape[1]
        reg_out = self.linear(x)
        #print(reg_out.shape)
        latent = self.linear.in_features
        #print(x.shape, self.linear.weight.shape)
        perm_out = x@torch.flip(self.linear.weight.t(), dims=[0])
        perm_out += x@torch.flip(self.linear.weight.t(), dims=[1])
        for i in range(int(self.num_shifts/2)):
            perm_out += x@torch.roll(self.linear.weight.t(), shifts=i+1,dims=1) # Col shifted
            perm_out += x@torch.roll(self.linear.weight.t(), shifts=i+1,dims=0) # Col shifted
        out = reg_out + perm_out
         
        return out

class Rand_Lin_Perm(nn.Module):
    def __init__(self,in_features,out_features,perms,use_norm=True,dropout=0.1,act=nn.GELU()):
        super(Rand_Lin_Perm, self).__init__()
        self.lin = nn.Linear(in_features,out_features)
        #self.row_perms = [torch.randperm(out_features).to('mps') for i in range(perms)]
        self.row_perms = [torch.randperm(out_features*in_features).to('cuda') for i in range(perms)]
        #self.col_perms = [torch.randperm(in_features).to('mps') for i in range(perms)]
        self.use_norm = use_norm
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        if self.use_norm:
            self.norm = nn.LayerNorm(out_features)
            #self.norm2 = nn.BatchNorm1d(out_features)
    
    def forward(self,x):
        if self.use_norm:
            #reg_out = self.dropout(self.norm2(self.norm(self.lin(x))))
            reg_out = self.dropout(self.norm(self.lin(x)))
        else:
            reg_out = self.dropout(self.act(self.lin(x)))
        t_out = 0
        for perm in self.row_perms:
           # w = random_permutation_transform(self.lin.weight,perm)
            w = permute_with_fixed_indices(self.lin.weight,perm)
            if self.use_norm:
                #t_out += self.dropout(self.norm2(self.norm(x@w.t())))
                t_out += self.dropout(self.norm(x@w.t()))
            else:
                t_out += self.act(self.dropout(x@w.t()))
            
        return t_out+reg_out 
    
def permute_with_fixed_indices(tensor, perm):
    """
    Apply a fixed permutation to the tensor using provided indices and values.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        perm_indices (torch.Tensor): A 1D tensor containing indices to permute.
        permuted_values (torch.Tensor): A 1D tensor containing permuted values to place at the indices.
        
    Returns:
        torch.Tensor: The tensor with permuted points.
    """
    # Flatten the tensor
    flat_tensor = tensor.flatten()
    
    # Place the permuted values at the given indices
    flat_tensor = flat_tensor[perm]
    
    # Reshape back to the original shape
    permuted_tensor = flat_tensor.view_as(tensor)
    
    return permuted_tensor


def get_model_parameter_count(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad==True))