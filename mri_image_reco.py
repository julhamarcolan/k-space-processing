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

def zero_fill_2Ddata(kspace_data, dim):
    """
    Applies a trapezoidal weighting function to the input k-space data and then zero-fills it
    to match the desired output dimensions.

    Parameters:
        kspace_data (ndarray): Input data in k-space.
        dim (tuple): Tuple representing the desired dimension of the zero-filled data.

    Returns:
        ndarray: Weighted and zero-filled k-space data.
    """
    zeroFill = 2
    data_shape = kspace_data.shape
    rows, cols = data_shape

    # Create 1D trapezoidal window for rows and columns
    def create_trapezoid(size):
        ramp = np.linspace(0, 1, 11)
        flat = np.ones(size - 22)
        if size <= 22:
            return np.ones(size)  # fallback if matrix is too small
        taper = np.concatenate([ramp, flat, ramp[::-1]])
        return taper

    row_trap = create_trapezoid(rows)
    col_trap = create_trapezoid(cols)

    # Create 2D trapezoidal window
    trapezoid_2d = np.outer(row_trap, col_trap)

    # Apply trapezoid to k-space data
    weighted_kspace = kspace_data * trapezoid_2d

    # Compute padding amounts
    dim0Padding = max(0, int((dim[0] - rows) / zeroFill))
    dim1Padding = max(0, int((dim[1] - cols) / zeroFill))

    # Apply zero-padding
    zero_filled_data = np.pad(weighted_kspace, [(dim0Padding, dim0Padding), (dim1Padding, dim1Padding)], mode='constant')

    return zero_filled_data

# Load original k-space
k_space = np.load('k_space.npy')

zero_fill_data = zero_fill_2Ddata(k_space, [1024,1024])

image_data =  reconstruct_image(k_space)
image_data_zf = reconstruct_image(zero_fill_data)


# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(np.log(np.abs(k_space) + 1), cmap='gray')
plt.title("Original K-space")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(zero_fill_data) + 1), cmap='gray')
plt.title("Zero-Filled K-space")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(image_data, cmap='gray')
plt.title("Reconstructed Image (Original)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(image_data_zf, cmap='gray')
plt.title("Reconstructed Image (Zero-Filled)")
plt.axis('off')

plt.tight_layout()
plt.show()