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


# Load original k-space
k_space = np.load('k_space.npy')
print(k_space.shape)

image_data =  reconstruct_image(k_space)

# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(k_space) + 1), cmap='gray')
plt.title("Original K-space")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(image_data), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')

plt.tight_layout()
plt.show()