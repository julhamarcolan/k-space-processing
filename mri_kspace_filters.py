import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft2, ifftshift

def reconstruct_image(k_space):
    """
    Reconstructs an image from its k-space representation using inverse Fourier transform.

    Parameters:
        k_space (ndarray): Input data in k-space (frequency domain), typically obtained via FFT.

    Returns:
        ndarray: Reconstructed image in the spatial domain, returned as a real-valued array.
    """
    k_space_decentered = ifftshift(k_space)
    reconstructed_image = ifft2(k_space_decentered)
    return np.abs(reconstructed_image)


def remove_central_frequencies(k_space, radius):
    """
    Applies a circular notch filter to remove central frequencies from the k-space data.

    Parameters:
        k_space (ndarray): Input k-space data (2D complex-valued array).
        radius (int): Radius (in pixels) of the circular region centered in k-space to be zeroed.

    Returns:
        ndarray: Filtered k-space data with the central circular region removed.
    """
    height, width = k_space.shape
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    distance_squared = (X - center_x)**2 + (Y - center_y)**2
    mask = distance_squared >= radius**2
    return k_space * mask


def keep_central_frequencies(k_space, radius):
    """
    Applies a circular low-pass filter to keep only the central low frequencies in the k-space data.

    Parameters:
        k_space (ndarray): Input k-space data (2D complex-valued array).
        radius (int): Radius (in pixels) of the circular region centered in k-space to be preserved.

    Returns:
        ndarray: Filtered k-space data with only the central circular region retained.
    """
    height, width = k_space.shape
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    distance_squared = (X - center_x)**2 + (Y - center_y)**2
    mask = distance_squared <= radius**2
    return k_space * mask


# --- Main Processing ---

# Load original k-space
k_space = np.load('k_space.npy')
image_original = reconstruct_image(k_space)

# High-pass filter (removes central frequencies)
k_space_highpass = remove_central_frequencies(k_space, radius=30)
image_highpass = reconstruct_image(k_space_highpass)

# Low-pass filter (keeps only central frequencies)
k_space_lowpass = keep_central_frequencies(k_space, radius=30)
image_lowpass = reconstruct_image(k_space_lowpass)

# --- Plotting ---
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(np.log(np.abs(k_space) + 1), cmap='gray')
plt.title('Original K-space')
plt.axis('off')

vmin = 0
vmax = np.log(np.abs(k_space).max() + 1)
plt.subplot(2, 3, 2)
plt.imshow(np.log(np.abs(k_space_highpass) + 1), cmap='gray', vmin=vmin, vmax=vmax)
plt.title('High-pass K-space')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.log(np.abs(k_space_lowpass) + 1), cmap='gray')
plt.title('Low-pass K-space')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(image_original), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(image_highpass), cmap='gray')
plt.title('High-pass Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(image_lowpass), cmap='gray')
plt.title('Low-pass Image')
plt.axis('off')

plt.tight_layout()
plt.show()
