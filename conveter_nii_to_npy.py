import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def convert_nifti_to_numpy(nifti_file_path):
    """
    Converts a NIfTI (.nii) file to a NumPy array.

    Parameters:
        nifti_file_path (str): Path to the .nii file.

    Returns:
        np.ndarray: The image data as a NumPy array.
    """
    nifti_img = nib.load(nifti_file_path)
    data = nifti_img.get_fdata()
    return data

#!----------main--------
nifti_path = "IXI039-HH-1261-IXIDE3Diso_-s3T116_-0401-00004-000001-01.nii"

image_data = convert_nifti_to_numpy(nifti_path)

fft_result = fft2(image_data[:,:,60])
k_space = fftshift(fft_result)
k_space = np.rot90(k_space)
np.save('k_space.npy', k_space)


# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(k_space) + 1), cmap='gray')
plt.title("Original K-space")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.rot90(np.abs(image_data[:,:,60])), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')

plt.tight_layout()
plt.show()