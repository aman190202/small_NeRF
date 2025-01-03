import numpy as np
from load_data import dataloader
from helper import get_rays_np
import torch
from model import NerfModel, train
from torch.utils.data import DataLoader
import time
from visualize import visualize_nerf_output
#from processing import create_nerf

expname = "fern",
datadir = "data",
basedir = "./logs",
factor = 2, # Downsample the image by a factor of what
spherify = True, # The spherify_poses function modifies a set of camera poses to fit a spherical trajectory around the scene, ensuring the camera’s path lies on a sphere. 
llffhold = 5, # every 8th image to be stored for test
render_test = "True",  # Render the test set instead of a given path
N_rand = 32 * 32 * 4  # Batch size
use_batching = True # Use batching
device = 'cuda'

if __name__ == "__main__":

    # get data
    # convert data into nerf form

    images, i_train, i_test, H, W ,  K , near, far, poses = dataloader(expname, datadir,basedir,factor,spherify)
    render_poses = np.array(poses[i_test])
        
    # Separate poses for training and testing
    train_poses = poses[i_train, :3, :4]  # Only training poses
    test_poses = poses[i_test, :3, :4]    # Only testing poses

    # Generate rays for training
    rays_train = np.stack([get_rays_np(H, W, K, p) for p in train_poses], axis=0)
    rays_train = rays_train.reshape(-1,6)
    images_train = images[i_train].reshape(-1, 3)  # Flatten training images
    training_dataset = torch.cat((torch.tensor(rays_train), torch.tensor(images_train)), dim=-1)  

    # Generate rays for testing
    rays_test = np.stack([get_rays_np(H, W, K, p) for p in test_poses], axis=0)
    rays_test = rays_test.reshape(-1,6)  
    images_test = images[i_test].reshape(-1, 3)  # Flatten testing images
    testing_dataset = torch.cat((torch.tensor(rays_test), torch.tensor(images_test)), dim=-1) 

    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=N_rand, shuffle=True)


    loss = train(model, model_optimizer, scheduler, data_loader, nb_epochs=120, device=device, hn=near, hf=far, nb_bins=10, H=H,W=W, testing_dataset = testing_dataset, test_len=len(i_test))

    # Save the model weights
    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f'out/nerf_model_weights_{current_datetime}_{expname}.pth')

    # Visualize NeRF from model rather than retraining
    # for img_index in range(2):
    #     visualize_nerf_output(model, near, far, testing_dataset, img_index=img_index, nb_bins=192, H=H, W=W, device=device)
