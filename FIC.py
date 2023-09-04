import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import transformations as transf


def compress_image(img, source_size, destination_size, step):
    """
    Compress the input image using fractal image compression.

    Args:
        img (numpy.ndarray): The input image.
        source_size (int): Size of the source block.
        destination_size (int): Size of the destination block.
        step (int): Step size for block extraction.

    Returns:
        list: List of transformations for each block.
    """
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
    i_block_count, j_block_count = img.shape[0] // destination_size, img.shape[1] // destination_size

    transformations = find_best_transformations(
        img, destination_size, transformed_blocks, i_block_count, j_block_count, step
    )

    return transformations


def generate_all_transformed_blocks(img, source_size, destination_size, step):
    """
    Generate all possible transformed blocks for the given image.

    Args:
        img (numpy.ndarray): The input image.
        source_size (int): Size of the source block.
        destination_size (int): Size of the destination block.
        step (int): Step size for block extraction.

    Returns:
        list: List of transformed blocks and their metadata.
    """
    directions = [1, -1]
    angles = [0, 90, 180, 270]
    candidates = [[direction, angle] for direction in directions for angle in angles]

    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            source_block = transf.reduce(
                img[k * step : k * step + source_size, l * step : l * step + source_size], factor
            )
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append(
                    (k, l, direction, angle, transf.apply_transformation(source_block, direction, angle))
                )
    return transformed_blocks


def find_best_transformations(img, destination_size, transformed_blocks, i_count, j_count, step):
    """
    Find the best transformations for all destination blocks in the image.

    Args:
        img (numpy.ndarray): The input image.
        destination_size (int): Size of the destination block.
        transformed_blocks (list): List of transformed blocks and their metadata.
        i_block_count (int): Number of rows of destination blocks.
        j_block_count (int): Number of columns of destination blocks.
        step (int): Step size for block extraction.

    Returns:
        list: List of transformations for each block.
    """
    transformations = []

    for i in range(i_count):
        transformations.append([None] * j_count)

        for j in range(j_count):
            destination_block = img[
                i * destination_size : (i + 1) * destination_size,
                j * destination_size : (j + 1) * destination_size,
            ]
            best_transformation = find_best_transformation(transformed_blocks, destination_block)
            transformations[i][j] = best_transformation

    return transformations


def find_best_transformation(transformed_blocks, destination_block):
    """
    Find the best transformation for a destination block from a list of transformed blocks.

    Args:
        transformed_blocks (list): List of transformed blocks and their metadata.
        destination_block (numpy.ndarray): The destination block.

    Returns:
        tuple: Best transformation metadata.
    """
    min_distance = float("inf")
    best_transformation = None

    for k, l, direction, angle, source_block in transformed_blocks:
        contrast, brightness = estimate_contrast_and_brightness(source_block, destination_block)
        source_block = contrast * source_block + brightness
        distance = np.sum(np.square(destination_block - source_block))

        if distance < min_distance:
            min_distance = distance
            best_transformation = (k, l, direction, angle, contrast, brightness)

    return best_transformation


def estimate_contrast_and_brightness(source_block, destination_block):
    """
    Estimate the contrast and brightness adjustments to match the destination block to the source block.

    Args:
        source_block (numpy.ndarray): The source block.
        destination_block (numpy.ndarray): The destination block.

    Returns:
        tuple: Estimated contrast and brightness values.
    """
    source_mean = np.mean(source_block)
    destination_mean = np.mean(destination_block)

    source_variance = np.var(source_block)
    covar = np.cov(source_block.flatten(), destination_block.flatten())[0][1]

    contrast = covar / source_variance if source_variance > 0 else 1.0
    brightness = destination_mean - (contrast * source_mean)

    return contrast, brightness


def decompress_image(transformations, source_size, destination_size, step, num_iterations=8):
    """
    Decompress the image using the given transformations.

    Args:
        transformations (list): List of transformations for each block.
        source_size (int): Size of the source block.
        destination_size (int): Size of the destination block.
        step (int): Step size for block extraction.
        num_iterations (int): Number of iterations for decompression.

    Returns:
        list: List of decompressed images at each iteration.
    """
    factor = source_size // destination_size
    height, width = len(transformations) * destination_size, len(transformations[0]) * destination_size
    iterations = [initialize_random_image(height, width)]
    current_img = np.zeros((height, width))

    for iteration in range(num_iterations):
        current_img = apply_transformations_to_blocks(
            transformations, iterations[-1], source_size, destination_size, step, factor
        )
        iterations.append(current_img)

    return iterations


def initialize_random_image(height, width):
    """
    Initialize a random image with the specified height and width.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        numpy.ndarray: Randomly initialized image.
    """
    return np.random.randint(0, 256, (height, width))


def apply_transformations_to_blocks(
    transformations, current_img, source_size, destination_size, step, factor
):
    """
    Apply transformations to blocks in the current image.

    Args:
        transformations (list): List of transformations for each block.
        current_img (numpy.ndarray): Current image.
        source_size (int): Size of the source block.
        destination_size (int): Size of the destination block.
        step (int): Step size for block extraction.
        factor (int): Reduction factor.

    Returns:
        numpy.ndarray: Updated image after applying transformations.
    """
    height, width = current_img.shape
    for i, row in enumerate(transformations):
        for j, t in enumerate(row):
            if t is not None:
                k, l, flip, angle, contrast, brightness = t
                source_block = transf.reduce(
                    current_img[k * step : k * step + source_size, l * step : l * step + source_size],
                    factor,
                )
                destination_block = transf.apply_transformation(
                    source_block, flip, angle, contrast, brightness
                )
                current_img[
                    i * destination_size : (i + 1) * destination_size,
                    j * destination_size : (j + 1) * destination_size,
                ] = destination_block

    return current_img


def load_and_preprocess_image(file_path, reduction_factor):
    """
    Load and preprocess the input image.

    Args:
        file_path (str): Path to the image file.
        reduction_factor (int): Reduction factor for initial processing.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    img = mpimg.imread(file_path)

    img = np.mean(img[:, :, :2], 2)  # Convert the image to grayscale

    img = transf.reduce(img, reduction_factor)  # Apply initial reduction

    return img


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "image.gif")
    reduction_factor = 4
    input_image = load_and_preprocess_image(file_path, reduction_factor)

    compressed_transformations = compress_image(input_image, 8, 4, 8)

    decompressed_images = decompress_image(compressed_transformations, 8, 4, 8, num_iterations=20)

    plt.figure()
    plt.imshow(decompressed_images[20], cmap="gray", interpolation="none")
    plt.show()
