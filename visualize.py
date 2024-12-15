import torch, matplotlib.pyplot as plt

def visualize_nerf_output(model, near, far, testing_dataset, img_index, nb_bins, H, W, device):
    model.eval()
    with torch.no_grad():
        rays = testing_dataset[img_index * H * W:(img_index + 1) * H * W, :6].to(device)
        rays = rays.view(H, W, 6)
        rgb_map = torch.zeros((H, W, 3), device=device)
        
        for i in range(H):
            for j in range(W):
                ray = rays[i, j]
                rgb, _ = model(ray, near, far, nb_bins)
                rgb_map[i, j] = rgb

        rgb_map = rgb_map.cpu().numpy()
        plt.imshow(rgb_map)
        plt.title(f"NeRF Output Image {img_index}")
        plt.axis('off')
        plt.show()