import numpy as np
from load_data import dataloader
from helper import 
#from processing import create_nerf

expname = "fern",
datadir = "./data/nerf_llff_data/fern",
basedir = "./logs",
factor = 2, # Downsample the image by a factor of what
spherify = True, # The spherify_poses function modifies a set of camera poses to fit a spherical trajectory around the scene, ensuring the camera’s path lies on a sphere. 
llffhold = 8, # every 8th image to be stored for test
render_test = "True",  # Render the test set instead of a given path
N_rand = 32*32*4 # Batch size
use_batching = True # Use batching

if __name__ == "__main__":
    
    # Retrieve data from the files
    images, i_train, i_test, K , near, far, poses,  hwf = dataloader()
    render_poses = np.array(poses[i_test])
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0


  
    # # setting up the model, optimizer, and rendering configurations
    # render_kwargs_train, reder_kwargs_test, start, grad_vars, optimizer = create_nerf(basedir)
    
    # global_step = start

    # bds_dict = {
    #     'near' : near,
    #     'far' : far,
    # }
    # render_kwargs_train.update(bds_dict)
    # render_kwargs_test.update(bds_dict)

