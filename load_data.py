from load_llff import load_llff_data
import numpy as np
import os 


def dataloader(
        expname = "fern",
        datadir = "./data/nerf_llff_data/fern",
        basedir = "./logs",
        factor = 2, # Downsample the image by a factor of what
        spherify = True, # The spherify_poses function modifies a set of camera poses to fit a spherical trajectory around the scene, ensuring the camera’s path lies on a sphere. 
        llffhold = 8, # every 8th image to be stored for test
        render_test = "True",  # Render the test set instead of a given path
        ):
    
    
    images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=spherify)
    hwf = poses[0,:3,-1] # height width focal length
    poses = poses[:,:3,:4] #
    print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')

    near = 0.
    far = 1.

    print('NEAR FAR', near, far)


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]


    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if render_test:
        render_poses = np.array(poses[i_test])


    # Create logs

    args_file_path = os.path.join(basedir, expname, 'args.txt')
    args_dir_path = os.path.dirname(args_file_path)  # Get the directory path

    # Ensure the directory exists
    os.makedirs(args_dir_path, exist_ok=True)

    # Create and write to args.txt
    with open(args_file_path, 'w') as args_file:
        args_file.write(f'expname = {expname}\n')
        args_file.write(f'datadir = {datadir}\n')
        args_file.write(f'basedir = {basedir}\n')
        args_file.write(f'factor = {factor}\n')
        args_file.write(f'spherify = {spherify}\n')
        args_file.write(f'llffhold = {llffhold}\n')
        args_file.write(f'render_test = {render_test}\n')

    print(f'Created {args_file_path}')

    return images, i_train, i_test, K , near, far, poses, render_poses, hwf , bds

