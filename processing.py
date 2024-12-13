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

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)

        return outputs    




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

x = torch.Tensor([1,2,3])
y = positional_encoding(x,10)
print(y.shape)