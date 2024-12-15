import numpy as np
import torch

def get_rays_np(H, W, K, c2w):
    
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.from_numpy(d.copy())
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    rays_o = torch.from_numpy(o.copy())
    return torch.cat((rays_o, rays_d),dim = -1)


