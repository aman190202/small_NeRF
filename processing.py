import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

frequency = 10 # 2^0 to 2^10 for positional encoding
use_viewdirs = "True"
N_importance = 0 # Only use coarse sampling for once
device = "mps"
lrate = 5e-4
no_reload = "True"

def positional_encoding(inputs, num_freqs=10):
    """
    Positional encoding for inputs.

    Args:
        inputs (torch.Tensor): Input tensor of shape [N, d].
        num_freqs (int): Number of frequency bands for encoding.

    Returns:
        torch.Tensor: Positional encoded tensor.
        int: Output dimensionality of the encoding.
    """
    d = inputs.shape[-1]  # Dimensionality of the input (e.g., 3 for 3D coords)
    encoded = [inputs]
    freq_bands = torch.arange(1, num_freqs + 1, dtype=torch.float32)

    for freq in freq_bands:
        encoded.append(torch.sin(inputs * freq))
        encoded.append(torch.cos(inputs * freq))

    out_dim = d + 2 * d * num_freqs  # Original + 2 * num_freqs * input_dimensionality
    return torch.cat(encoded, dim=-1), out_dim

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, pos=3, hidden_dim=256, depth=8, skips=[4]):
        """
        NeRF Model with dynamic skip connections and without using shortcuts.

        Args:
            pos (int): Input dimensions (e.g., 3 for x, y, z coordinates).
            hidden_dim (int): Number of hidden units in each layer.
            depth (int): Number of layers in the network.
            skips (list): Layers where skip connections are added.
        """
        super(NeRF, self).__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.skips = skips

        # Define the input layer
        self.input_layer = nn.Linear(pos, hidden_dim)

        # Define the intermediate layers
        self.intermediate_layers = nn.ModuleList()
        for i in range(1, depth):
            if i in skips:
                self.intermediate_layers.append(nn.Linear(hidden_dim + pos, hidden_dim))
            else:
                self.intermediate_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Define the output layers for RGB and Alpha
        self.output_rgb = nn.Linear(hidden_dim, 3)  # RGB color output
        self.output_alpha = nn.Linear(hidden_dim, 1)  # Alpha (density) output

    def forward(self, x):
        """
        Forward pass for NeRF.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, pos].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 4] (RGB + Alpha).
        """
        # Store the original input for skip connections
        input_pos = x

        # Pass input through the input layer
        h = self.input_layer(input_pos)
        h = F.relu(h)

        # Pass through intermediate layers with skip connections
        for i, layer in enumerate(self.intermediate_layers):
            if i + 1 in self.skips:  # Check if the current layer is a skip connection
                h = torch.cat([input_pos, h], dim=-1)  # Concatenate the input with current hidden state
            h = layer(h)
            h = F.relu(h)

        # Compute RGB output
        rgb = self.output_rgb(h)

        # Compute Alpha output
        alpha = self.output_alpha(h)

        # Combine RGB and Alpha into a single output tensor
        output = torch.cat([rgb, alpha], dim=-1)
        return output



def create_nerf(basedir,expname):

    position = [1,2,3]
    views = [1,2]

    #positon embeddings
    positon_embeddings, input_ch = positional_encoding(position,frequency)

    # View direction embeddings
    view_embeddings, input_ch_views = positional_encoding(views,frequency)

    #NeRF Model
    model = NeRF(input_ch=input_ch, input_ch_views=input_ch_views).to(device)

    grad_vars = list(model.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

    start = 0
    basedir = basedir
    expname = expname
    
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    
# beginning
# render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
# if __name__=='__main__':
#     x = torch.Tensor([1,2,3])
#     y = positional_encoding(x,10)
#     print(x.shape, y.shape)