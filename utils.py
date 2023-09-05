import matplotlib.pyplot as plt
import imageio
import numpy as np


def plot_decompressed_images(decompressed_images, rows, cols):
    """
    Plot decompressed images in a grid format.

    Args:
        decompressed_images (list): List of decompressed images.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(8, 5))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(decompressed_images):
                axes[i, j].imshow(decompressed_images[index], cmap="gray")
                axes[i, j].axis("off")
                axes[i, j].set_title(f"Iteration {index+1}")

    plt.tight_layout()
    plt.show()


def compress_with_jpeg(input_image, jpeg_quality=95):
    """
    Compress the input image using JPEG compression.

    Args:
        input_image (numpy.ndarray): The input image.
        jpeg_quality (int): JPEG compression quality (0-100).

    Returns:
        numpy.ndarray: Compressed image.
    """
    output_path = "output.jpg"
    imageio.imwrite(output_path, input_image.astype(np.uint8), quality=jpeg_quality)
    return imageio.imread(output_path)


def calculate_psnr(original_image, compressed_image):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        original_image (numpy.ndarray): The original image (ground truth).
        compressed_image (numpy.ndarray): The compressed image.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = np.mean((original_image - compressed_image) ** 2)
    max_pixel_value = np.max(original_image)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr
