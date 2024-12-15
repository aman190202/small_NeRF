import torch
import numpy as np
import imageio
import torch.nn as nn
import numpy as np
from load_data import dataloader
from helper import get_rays_np
import os
from model import NerfModel, render_rays
import gc

def create_nerf_spiral_path(poses, num_frames=10):
    """
    Create a smooth spiral path connecting poses for NeRF rendering.
    
    Parameters:
    -----------
    poses : np.ndarray
        Original camera poses
    num_frames : int, optional
        Total number of frames to generate (default: 120 for 5 seconds at 24fps)
    
    Returns:
    --------
    np.ndarray
        Array of interpolated camera poses for smooth spiral path rendering
    """
    # Compute the center of all poses
    center = np.mean(poses[:, :3, 3], axis=0)
    
    # Compute the radius of the spiral
    radii = np.linalg.norm(poses[:, :3, 3] - center, axis=1)
    avg_radius = np.mean(radii)
    
    # Generate spiral path
    spiral_poses = np.zeros((num_frames, 3, 4))
    
    # Angle and height parameters for spiral
    theta = np.linspace(0, 4 * np.pi, num_frames)
    height = np.linspace(-avg_radius/2, avg_radius/2, num_frames)
    
    for i in range(num_frames):
        # Compute spiral coordinates
        x = avg_radius * np.cos(theta[i])
        y = avg_radius * np.sin(theta[i])
        z = height[i]
        
        # Create translation vector
        t = center + np.array([x, y, z])
        
        # Create rotation matrix (look at center)
        look_dir = center - t
        look_dir /= np.linalg.norm(look_dir)
        
        # Simple approximation of camera rotation
        up = np.array([0, 0, 1])
        right = np.cross(look_dir, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, look_dir)
        
        # Construct pose matrix
        R = np.column_stack((right, up, -look_dir))
        spiral_poses[i, :3, :3] = R
        spiral_poses[i, :3, 3] = t
    
    return spiral_poses

def generate_rays_from_poses(poses, H, W, K):
    """
    Generate rays for a set of poses.
    
    Parameters:
    -----------
    poses : np.ndarray
        Camera poses
    H : int
        Image height
    W : int
        Image width
    K : np.ndarray
        Camera intrinsic matrix
    
    Returns:
    --------
    np.ndarray
        Rays for the given poses
    """
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses], axis=0)
    return rays.reshape(-1, 6)


def render_smooth_video(checkpoint_path, 
                        H, 
                        W, 
                        K, 
                        near, 
                        far, 
                        poses, 
                        output_path='nerf_smooth_video.mp4', 
                        num_frames=120, 
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        batch_size=1024):
    """
    Memory-efficient rendering of smooth video from NeRF checkpoint.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to the saved model checkpoint
    H : int
        Image height
    W : int
        Image width
    K : np.ndarray
        Camera intrinsic matrix
    near : float
        Near plane distance
    far : float
        Far plane distance
    poses : np.ndarray
        Original camera poses (used to compute spiral path)
    output_path : str, optional
        Path to save the output video
    num_frames : int, optional
        Number of frames to render (default: 120 for 5 seconds at 24fps)
    device : str, optional
        Device to run the model on
    batch_size : int, optional
        Number of rays to render in each batch
    
    Returns:
    --------
    list
        List of rendered frames
    """
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create spiral path
    spiral_poses = create_nerf_spiral_path(poses, num_frames)
    
    # Generate rays for spiral poses
    spiral_rays = generate_rays_from_poses(spiral_poses, H, W, K)
    
    # Instantiate the model
    model = NerfModel(hidden_dim=256)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Prepare rays
    ray_origins = torch.tensor(spiral_rays[:, :3], dtype=torch.float32).to(device)
    ray_directions = torch.tensor(spiral_rays[:, 3:], dtype=torch.float32).to(device)
    
    # Render frames
    rendered_frames = []
    
    with torch.no_grad():
        for frame_idx in range(num_frames):
            # Clear cache periodically
            if frame_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Select rays for this frame
            start_idx = frame_idx * H * W
            end_idx = (frame_idx + 1) * H * W
            
            frame_ray_origins = ray_origins[start_idx:end_idx]
            frame_ray_directions = ray_directions[start_idx:end_idx]
            
            # Render frame in batches
            frame_pixels = []
            for batch_start in range(0, H*W, batch_size):
                batch_end = min(batch_start + batch_size, H*W)
                
                # Get batch of rays
                batch_ray_origins = frame_ray_origins[batch_start:batch_end].to(device)
                batch_ray_directions = frame_ray_directions[batch_start:batch_end].to(device)
                
                # Render batch of rays
                batch_pixels = render_rays(model, 
                                          batch_ray_origins, 
                                          batch_ray_directions, 
                                          hn=near, 
                                          hf=far, 
                                          nb_bins=192)
                
                frame_pixels.append(batch_pixels.cpu())
                
                # Free up memory
                del batch_ray_origins, batch_ray_directions, batch_pixels
                torch.cuda.empty_cache()
            
            # Combine pixels for this frame
            frame_image = torch.cat(frame_pixels).numpy().reshape(H, W, 3)
            imageio.imwrite(f'rendered/img_{frame_idx}.png', (frame_image * 255).astype(np.uint8))
            #rendered_frames.append(frame_image)
            
            # Print progress
            print(f"Rendered frame {frame_idx + 1}/{num_frames}")
    
    # Convert to uint8 for video
    video_frames = (np.array(rendered_frames) * 255).astype(np.uint8)
    
    # Save video
    imageio.mimsave(output_path, video_frames, fps=24)
    
    return video_frames


# Example usage
if __name__ == "__main__":
    # Memory optimization for PyTorch
    torch.backends.cudnn.benchmark = True
    
    # Load data
    images, i_train, i_test, H, W, K, near, far, poses = dataloader()
    
    # Path to your latest checkpoint (you may need to modify this)
    latest_checkpoint = "/root/small_NeRF/out/nerf_model_weights_20241215-223857_5.pth"
    
    # Render smooth video with memory optimization
    rendered_frames = render_smooth_video(
        checkpoint_path=latest_checkpoint,
        H=100, 
        W=100, 
        K=K, 
        near=near, 
        far=far, 
        poses=poses,
        output_path='nerf_smooth_spiral_video.mp4',
        batch_size=1024  # Adjustable batch size for memory management
    )