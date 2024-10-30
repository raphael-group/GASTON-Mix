import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
import random
from sklearn import preprocessing

import os
from pprint import pprint
from csv import reader
import csv, os, argparse, sys

from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

def positional_encoding(coords, enc_dim=8, sigma=0.1, include_orig_coords=False):
    freqs = (
        2 * np.pi * sigma ** (torch.arange(enc_dim//2, dtype=torch.float, device=coords.device) / enc_dim)
    )
    freqs = torch.reshape(freqs, (1,1, torch.numel(freqs)))
    coords_copy = coords.unsqueeze(-1).clone().detach()

    freqs = coords_copy * freqs #N x 2 x enc_dim/2
    s = torch.sin(freqs)
    c = torch.cos(freqs)

    x = torch.cat((s,c), axis=-1) #N x 2 x enc_dim
    x = torch.reshape(x, (x.shape[0], -1))
    if include_orig_coords:
        x=torch.cat((coords,x),dim=1)

    return x

######################################################################################################
# MoE code 
# based on (an older version of) https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py 
# NOTE: Ignore noisy gating, I turned it off
######################################################################################################

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """
    
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

#############
# GASTON MoE
#############

class GASTON_MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with GASTON isodepths as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, G, S_hidden_list, A_hidden_list, gating_hidden_list, 
                 num_experts, k=1, activation_fn=nn.ReLU(), gate_nn=None, 
                 noisy_gating=False, routing_loss=False, 
                 pos_encoding_g=False, enc_dim_g=8, sigma_g=0.1,
                 pos_encoding_i=False, enc_dim_i=8, sigma_i=0.1,
                 include_orig_coords=False):
        super(GASTON_MoE, self).__init__()
        self.num_experts = num_experts
        self.output_size = G
        self.k = k

        self.pos_encoding_g=pos_encoding_g
        self.enc_dim_g=enc_dim_g
        self.sigma_g=sigma_g

        self.pos_encoding_i=pos_encoding_i
        self.enc_dim_i=enc_dim_i
        self.sigma_i=sigma_i
        
        self.include_orig_coords=include_orig_coords
        print(f'self.enc_dim_g: {self.enc_dim_g}, self.sigma_g: {self.sigma_g}, self.include_orig_coords: {self.include_orig_coords}')
        print(f'self.enc_dim_i: {self.enc_dim_i}, self.sigma_i: {self.sigma_i}, self.include_orig_coords: {self.include_orig_coords}')

        if self.pos_encoding_g:
            gating_input_dim=2*self.enc_dim_g
            if self.include_orig_coords:
                gating_input_dim += 2
        else:
            gating_input_dim=2
        
        if self.pos_encoding_i:
            S_input_dim=2*self.enc_dim_i
            if self.include_orig_coords:
                S_input_dim += 2
        else:
            S_input_dim=2

        # experts
        isodepths_list=[]
        for _ in range(num_experts):
            S_layer_list=[S_input_dim] + S_hidden_list + [1]
            S_layers=[]
            for l in range(len(S_layer_list)-1):
                # add linear layer
                S_layers.append(nn.Linear(S_layer_list[l], S_layer_list[l+1]))
                # add activation function except for last layer
                if l != len(S_layer_list)-2:
                    S_layers.append(activation_fn)
            isodepths_list.append(nn.Sequential(*S_layers))
        self.isodepths_list=nn.ModuleList(isodepths_list)

        expression_function_list=[]
        for _ in range(num_experts):
            A_layer_list=[1] + A_hidden_list + [G]
            A_layers=[]
            for l in range(len(A_layer_list)-1):
                # add linear layer
                A_layers.append(nn.Linear(A_layer_list[l], A_layer_list[l+1]))
                # add activation function except for last layer
                if l != len(A_layer_list)-2:
                    A_layers.append(activation_fn)
            expression_function_list.append(nn.Sequential(*A_layers))
        self.expression_functions_list=nn.ModuleList(expression_function_list)


        if gate_nn is None:
            gating_layer_list=[gating_input_dim] + gating_hidden_list + [num_experts]
            gating_layers=[]
            for l in range(len(gating_layer_list)-1):
                # add linear layer
                gating_layers.append(nn.Linear(gating_layer_list[l], gating_layer_list[l+1]))
                # add activation function except for last layer
                if l != len(gating_layer_list)-2:
                    gating_layers.append(activation_fn)
            self.gate_nn=nn.Sequential(*gating_layers)
        else:
            self.gate_nn=gate_nn

        self.noisy_gating=noisy_gating
        self.w_noise = nn.Parameter(torch.zeros(2, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.training=True
        self.softmax = nn.Softmax(1)

        if num_experts<2:
            self.routing_loss=False

        self.expert_means=torch.nn.Parameter( torch.Tensor(np.random.rand(num_experts,G)) )
        
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate_nn(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        if self.pos_encoding_g:
            x_pos_g=positional_encoding(x,enc_dim=self.enc_dim_g, sigma=self.sigma_g, include_orig_coords=self.include_orig_coords)
        else:
            x_pos_g=x

        if self.pos_encoding_i:
            x_pos_i=positional_encoding(x,enc_dim=self.enc_dim_i, sigma=self.sigma_i, include_orig_coords=self.include_orig_coords)
        else:
            x_pos_i=x

        gates, load = self.noisy_top_k_gating(x_pos_g, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        regularization_loss = self.cv_squared(importance) + self.cv_squared(load)
        regularization_loss *= loss_coef
        
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_pos_i)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.expression_functions_list[i](self.isodepths_list[i](expert_inputs[i])) for i in range(self.num_experts)]
        predicted_expression = dispatcher.combine(expert_outputs)

        logits=self.gate_nn(x_pos_g)
        return predicted_expression, gates, dispatcher._gates,logits, regularization_loss
    

############################################################
# Routing network
class GatingNetwork(nn.Module):
    def __init__(self, gating_hidden_list, num_experts, activation_fn=nn.ReLU(), pos_encoding=False, enc_dim=8, sigma=0.1, include_orig_coords=False):
        super().__init__()
        if pos_encoding:
            self.pos_encoding=True
            self.enc_dim=enc_dim
            input_dim=2*enc_dim
            self.sigma=sigma
            self.include_orig_coords=include_orig_coords
            if self.include_orig_coords:
                input_dim+=2
        else:
            self.pos_encoding=False
            self.enc_dim=None
            input_dim=2
            self.sigma=None
        gating_layer_list=[input_dim] + gating_hidden_list + [num_experts]
        gating_layers=[]
        for l in range(len(gating_layer_list)-1):
            # add linear layer
            gating_layers.append(nn.Linear(gating_layer_list[l], gating_layer_list[l+1]))
            # add activation function except for last layer
            if l != len(gating_layer_list)-2:
                gating_layers.append(activation_fn)
        self.gate_nn=nn.Sequential(*gating_layers)

    def forward(self, x):
        if self.pos_encoding:
            x_pos=positional_encoding(x, enc_dim=self.enc_dim, sigma=self.sigma, include_orig_coords=self.include_orig_coords)
        else:
            x_pos=x
        logits = self.gate_nn(x_pos)
        return logits

############################################################
# Helpers
############################################################

# Sin activation function
class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Rescale NN input so row/column means are 0
def rescale_input_data(S, A):
  assert S.shape[0] == A.shape[0], 'Input and output files do not have same number of rows! Some spots are missing or do not have expression PC values!'
  
  scaler = preprocessing.StandardScaler().fit(A)
  A_scaled = scaler.transform(A)
  A_torch = torch.tensor(A_scaled,dtype=torch.float32)

  scaler = preprocessing.StandardScaler().fit(S)
  S_scaled = scaler.transform(S)
  S_torch = torch.tensor(S_scaled,dtype=torch.float32)

  return S_torch, A_torch

############################################################
# Generating plots of intermediate isodepths
def plot_intermediate_isodepths(moe_model, S_torch, num_experts, SAVE_PATH=None, save=True, epoch=0, true_labels=None, y_factor=1,pos_encoding_i=False, num_contours=10,s_bg=1, s_clusters=5):
    N=S_torch.shape[0]
    S=S_torch.detach().cpu().numpy()
    if true_labels is not None:
        cell_labels_int, _ = pd.factorize(true_labels)
    else:
        cell_labels_int=np.zeros(N)

    # for backwards compatability
    if not hasattr(moe_model, 'include_orig_coords'):
        moe_model.include_orig_coords=False
    
    if pos_encoding_i:
        S_torch_input=positional_encoding(S_torch, enc_dim=moe_model.enc_dim_i, sigma=moe_model.sigma_i, include_orig_coords=moe_model.include_orig_coords)
        ds=[moe_model.isodepths_list[i](S_torch_input).cpu().detach().numpy().flatten() for i in range(num_experts)]
    else:
        ds=[moe_model.isodepths_list[i](S_torch).cpu().detach().numpy().flatten() for i in range(num_experts)]
    
    R=3
    C=int(np.ceil(num_experts/3))
    fig,axs=plt.subplots(R,C,figsize=(7*C,7*R), squeeze=False)
    
    e=0
    for r in range(R):
        for c in range(C):
            if e >= num_experts:
                continue
            for t in np.unique(cell_labels_int):
                axs[r,c].scatter(S[cell_labels_int==t,0],y_factor*S[cell_labels_int==t,1],s=s_bg,alpha=0.1)
            pts_e=moe_model(S_torch)[2][:,e]>0
            pts_e=pts_e.cpu()
            if pts_e.sum() == 0:
                e+=1
                continue
            
            im=axs[r,c].scatter(S[pts_e,0], y_factor*S[pts_e,1], c=ds[e][pts_e], cmap='Reds',s=s_clusters)
            if pts_e.sum()>2:
                try:
                    CS=axs[r,c].tricontour(S[pts_e,0], y_factor*S[pts_e,1], ds[e][pts_e], levels=np.min((num_contours,pts_e.sum())), colors='k', linestyles='solid')
                    axs[r,c].clabel(CS, CS.levels, inline=True, fontsize=10)
                except:
                    print('could not plot contours')
            axs[r,c].set_xlim([np.min(S[:,0]), np.max(S[:,0])])
            axs[r,c].set_ylim([np.min(y_factor*S[:,1]), np.max(y_factor*S[:,1])])
            axs[r,c].set_title(f'Isodepth {e}')
            e+=1
    if save:
        plt.savefig(SAVE_PATH+f'expert_isodepths_epoch_{epoch}.pdf')
        plt.close()
    else:
        plt.show()

############################################################

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, required=True)
    
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-f', '--folder_to_save', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('-n', '--num_experts', type=int)
    
    parser.add_argument('-e', '--num_epochs', type=int, required=True)
    parser.add_argument('-c', '--checkpoint', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=None)
    
    parser.add_argument('-g', '--gating_arch', nargs='*', type=int, default=[], help='A list of integers for gating architecture')
    parser.add_argument('-p', '--spatial_arch', nargs='*', type=int, default=[], help='A list of integers for isodepth function architecture')
    parser.add_argument('-x', '--expression_arch',nargs='*', type=int, default=[], help='A list of expression function architecture')

    parser.add_argument('--use_counts', action='store_true', help='whether or not to use raw counts matrix')
    
    parser.add_argument('-i', '--kmeans_init', action='store_true', help='whether or not to use kmeans initialization')
    parser.add_argument('-u', '--kmeans_num_clusters', type=int)
    parser.add_argument('-m', '--manual_init', default='', help='file for manual initialization')
    parser.add_argument('-a', '--num_epochs_init', type=int, default=20000)

    parser.add_argument('-l', '--plot_interm', action='store_true', help='whether or not to plot intermediate isodepths')

    parser.add_argument('--alternating', type=int, help='if set, then is equal to number of epochs to optimize each component')

    parser.add_argument('-r', '--regularization_coef', default=0, type=float)

    parser.add_argument('--activation_fn', default='ReLU', type=str)
    
    # parser.add_argument('--pos_encoding', action='store_true', help='whether or not to use positional encoding')
    parser.add_argument('--pos_encoding_gating', nargs='*', type=float, metavar=('enc_dim', 'sigma'),
                    help='Positional encoding for gating network with optional enc_dim and sigma. Defaults are enc_dim=8, sigma=0.1')

    parser.add_argument('--pos_encoding_isodepth', nargs='*', type=float, metavar=('enc_dim', 'sigma'),
                    help='Positional encoding for isoepth function with optional enc_dim and sigma. Defaults are enc_dim=8, sigma=0.1')

    parser.add_argument('--include_orig_coords', action='store_true', help='whether or not to include original coordinates in PE')

    
    return parser

def run(args):
    # Check for GPU
    if not args.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')
    
    ########################################
    # Load data/parameters and set seed
    ########################################
    S = np.load(f'{args.data_folder}/coords_mat.npy')
    if args.use_counts:
        A = np.load(f'{args.data_folder}/counts_mat.npy')
    else:
        A = np.load(f'{args.data_folder}/glmpca_mat.npy')

    seed=args.seed
    set_seeds(seed)
    
    N, G = A.shape
    S_torch, A_torch = rescale_input_data(S, A)
    if args.use_counts:
        A_torch=torch.Tensor(A) # preserve counts!
    S_torch, A_torch = S_torch.to(device), A_torch.to(device)

    num_epochs = args.num_epochs
    num_experts = args.num_experts
    
    if num_experts is None:
        print('num_experts not specified, loading from manual init')
        try:
            true_labels = np.load(args.manual_init, allow_pickle=True)
            num_experts = len(np.unique(true_labels))
        except:
            raise Exception('ERROR: num_experts and manual_init not specified')

    ########################################
    # Positional encoding
    ######################################## 
    if args.pos_encoding_gating is None:
        pos_encoding_g=False
        enc_dim_g=None
        sigma_g=None
    elif len(args.pos_encoding_gating) == 0:
        pos_encoding_g=True
        enc_dim_g=8
        sigma_g=0.1
    else:
        pos_encoding_g=True
        enc_dim_g=int(args.pos_encoding_gating[0])
        sigma_g=args.pos_encoding_gating[1]

    if args.pos_encoding_isodepth is None:
        pos_encoding_i=False
        enc_dim_i=None
        sigma_i=None
    elif len(args.pos_encoding_isodepth) == 0:
        pos_encoding_i=True
        enc_dim_i=8
        sigma_i=0.1
    else:
        pos_encoding_i=True
        enc_dim_i=int(args.pos_encoding_isodepth[0])
        sigma_i=args.pos_encoding_isodepth[1]
    
    include_orig_coords=args.include_orig_coords

    ########################################
    # Initialize routing network, gate_nn
    ########################################
    
    if args.kmeans_init or args.manual_init:
        if args.kmeans_init:
            print('k-means initialization')
            kmeans = KMeans(n_clusters=args.kmeans_num_clusters, random_state=1, n_init="auto").fit(A)
            true_labels = kmeans.labels_
        elif args.manual_init:
            print('Manual initialization')
            true_labels = np.load(args.manual_init, allow_pickle=True)
            if not np.issubdtype(true_labels.dtype, np.integer):  # if labels are strings, convert to integers
                labels_series = pd.Series(true_labels, dtype="category")
                true_labels = labels_series.cat.codes.to_numpy()

        true_labels_torch = torch.from_numpy(true_labels).long().to(device)
        gate_nn = GatingNetwork(args.gating_arch, num_experts, pos_encoding=pos_encoding_g, enc_dim=enc_dim_g, sigma=sigma_g, include_orig_coords=include_orig_coords).to(device)
        
        num_epochs_init = args.num_epochs_init
        opt = optim.Adam(gate_nn.parameters(), lr=1e-3)
        loss_list = np.zeros(num_epochs_init)
        
        weight=np.concatenate((np.unique(true_labels, return_counts=True)[1],np.zeros(np.max((num_experts-len(np.unique(true_labels)),0)))))
        weight=torch.Tensor(weight).to(device)
        loss_function = nn.CrossEntropyLoss(weight=weight)
        
        for epoch in range(num_epochs_init):
            if epoch % 500 == 0:
                print(epoch)
            opt.zero_grad()
            S_torch.requires_grad_()
        
            logits = gate_nn(S_torch)
            
            loss = loss_function(logits, true_labels_torch)
            loss_list[epoch] += loss.item()
            loss.backward()
            opt.step()
        gate_nn.pos_encoding=False # so you dont double positional encode input in gate_nn and moe_model.isodepth
    else:
        gate_nn = None
        true_labels=None

    ########################################
    # Create save path and save input
    ########################################
    SAVE_PATH = args.folder_to_save + f'/seed{seed}/'
    os.makedirs(args.folder_to_save, exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)

    if args.kmeans_init:
        np.save(SAVE_PATH+'kmeans_init.npy', true_labels)
    if args.manual_init:
        np.save(SAVE_PATH+'manual_init.npy', true_labels)

    torch.save(S_torch, SAVE_PATH+'Storch.pt')
    torch.save(A_torch, SAVE_PATH+'Atorch.pt')
    if args.kmeans_init or args.manual_init:
        np.save(SAVE_PATH+'init_loss_list.npy',loss_list)
        torch.save(gate_nn, SAVE_PATH+'init_NN.pt')

    ########################################
    # Activation function
    ########################################
    if args.activation_fn == 'ReLU':
        activation_fn=nn.ReLU()
    elif args.activation_fn == 'sine':
        activation_fn=Sin()
    else:
        raise Exception(f"Activation function {args.activation_fn} is not available")

    ########################################
    # Initialize MoE model
    ########################################
    moe_model = GASTON_MoE(A_torch.shape[1], args.spatial_arch, args.expression_arch, args.gating_arch, 
                           num_experts=num_experts, k=1, gate_nn=gate_nn, noisy_gating=False,
                           pos_encoding_g=pos_encoding_g, enc_dim_g=enc_dim_g, sigma_g=sigma_g, 
                           pos_encoding_i=pos_encoding_i, enc_dim_i=enc_dim_i, sigma_i=sigma_i, 
                           include_orig_coords=include_orig_coords).to(device)
    
    # opt=optim.Adam(moe_model.parameters(), lr=1e-2)

    opt_list=[]
    if args.alternating is not None:
        gate_nn_params = list(moe_model.gate_nn.parameters())
        non_routing_params = [p for n, p in moe_model.named_parameters() if 'gate_nn' not in n]
        
        gate_nn_optimizer = optim.Adam(gate_nn_params, lr=1e-3)
        # opt=optim.Adam(moe_model.parameters(), lr=1e-3)
        non_routing_opt = optim.Adam(non_routing_params, lr=1e-3)
        opt_list=[non_routing_opt,gate_nn_optimizer]
    else:
        opt=optim.Adam(moe_model.parameters(), lr=1e-3)
        opt_list=[opt]

    # Define the learning rate scheduler to reduce LR from 1e-1 to 1e-3 over 5000 epochs
    # lambda_lr = lambda epoch: max(1e-3 / 1e-1, (1 - epoch / 5000))  # Linear decay
    # scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
    
    loss_list = np.zeros(num_epochs)

    if args.use_counts:
        # loss_function = nn.PoissonNLLLoss(log_input=True, reduction='mean')
        exposure_mat=torch.tile(A_torch.sum(1), (A_torch.shape[1],1)).T
    else:    
        loss_function = nn.MSELoss(reduction='mean')
        exposure_mat=torch.Tensor(np.zeros((N,G)))
    
    ########################################
    # Train MoE model
    ########################################
    if args.batch_size is None:
        args.batch_size=N
    
    for epoch in range(num_epochs):
        if epoch % args.checkpoint == 0:
            print(f'epoch: {epoch}')
            torch.save(moe_model, SAVE_PATH + f'model_epoch_{epoch}.pt')
            if args.plot_interm:
                plot_intermediate_isodepths(moe_model, S_torch, num_experts, 
                                            SAVE_PATH=SAVE_PATH, epoch=epoch, true_labels=true_labels,pos_encoding_i=pos_encoding_i)


        # Shuffle the data before each epoch
        indices = torch.randperm(S_torch.size(0))
        S_torch_shuffled = S_torch[indices]
        A_torch_shuffled = A_torch[indices]
    
        # Split data into batches
        for i in range(0, S_torch.size(0), args.batch_size):
            # Create the batch
            S_batch = S_torch_shuffled[i:i + args.batch_size].clone().detach()
            A_batch = A_torch_shuffled[i:i + args.batch_size]
            exposure_mat_batch = exposure_mat[i:i + args.batch_size]
            
            for opt in opt_list:
                opt.zero_grad()

            S_batch.requires_grad_()
            
            pred, _, pred_gates, pred_logits, _ = moe_model(S_batch)
            pred.detach()
            
            if args.use_counts:
                loss = -1 * torch.mean((A_batch * pred) - (exposure_mat_batch * torch.exp(pred)))
            else:
                loss = loss_function(pred, A_batch)

            reg_loss=0
            # add mean regularization
            if args.regularization_coef > 0:
                # reg_loss += args.regularization_coef * torch.std( (pred_gates / (pred_gates.sum(0)+1)).T @ pred, 0).sum()
                reg_loss += args.regularization_coef * loss_function(pred_gates @ moe_model.expert_means, A_batch)
                loss += reg_loss
            
            ########################################
            # Loss / gradient update
            ########################################
            loss_list[epoch] += loss.item()
            loss.backward()
            
            if args.alternating is not None:
                to_update = int(epoch / args.alternating) % len(opt_list)
                opt_list[to_update].step()
                if epoch % args.checkpoint == 0:
                    print(f'updating {to_update}')
            else:
                opt_list[0].step()
        if epoch % args.checkpoint == 0:
            print(f'regularization loss: {reg_loss}')
            print(f'total loss: {loss_list[epoch]}')

    ########################################
    # Post-training: save final model and loss
    ########################################
    torch.save(moe_model, SAVE_PATH + f'final_model.pt')
    np.save(SAVE_PATH+'loss_list.npy', loss_list)
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(loss_list)
    plt.savefig(SAVE_PATH+'loss_list.pdf')

    if len(loss_list)>1000:
        fig,ax=plt.subplots(figsize=(6,6))
        ax.plot(loss_list[1000:])
        plt.savefig(SAVE_PATH+'loss_list_ignorefirst1000.pdf')
    

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))