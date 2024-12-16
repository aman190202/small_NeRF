import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def calculate_psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def main():
    # Paths
    reference_image_path = '/Users/aman/Developer/nerf/data/nerf_llff_data/fern/images_4/image007.png'
    comparison_directory =  '/Users/aman/Downloads/cit/fol'

    reference_img = Image.open(reference_image_path)

    image_files = [f for f in os.listdir(comparison_directory) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    # Sort files for ascending order
    image_files = sorted(image_files)

    results = []
    for img_index, filename in enumerate(image_files):
        img_path = os.path.join(comparison_directory, filename)
        comparison_img = Image.open(img_path)
        psnr_value = calculate_psnr(reference_img, comparison_img)
        results.append(psnr_value)

    # Print results
    for i, psnr_val in enumerate(results):
        print(f"Index {i}: {psnr_val:.2f} dB")

    # Plot graph with index on X-axis
    if results:
        indices = range(len(results))
        plt.figure(figsize=(10, 5))
        plt.plot(indices, results, marker='o')
        plt.title('PSNR Comparison')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)

        # Annotate each point with PSNR value only
        for i, psnr_val in enumerate(results):
            plt.annotate(f'{psnr_val:.2f}',
                         (i, psnr_val),
                         xytext=(5, 5),
                         textcoords='offset points')

        plt.tight_layout()
        plt.show()
    else:
        print("No images found in the comparison directory.")

if __name__ == "__main__":
    main()
