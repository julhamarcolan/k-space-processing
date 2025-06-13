# K-Space  and MRI Image Reconstruction

This repository contains  Python scripts for  k-space data processing (frequency domain) and reconstructing MRI images.

    - Manipulation of k-space using low-pass and high-pass filtering techniques.
    - Zero-filling applied to the data to enhance image detail.
    - Script for converting a NIfTI (.nii) file to a NumPy (.npy) file.

## 🔬 What is K-space?

In Magnetic Resonance Imaging (MRI), k-space is a matrix that stores the raw frequency-encoded data captured during a scan. This frequency information is not an image by itself, but a spatial frequency representation of the object. Applying an inverse Fourier transform to k-space data generates the actual MRI image in the spatial domain.

Understanding how spatial frequencies are distributed in k-space is essential:

🎯 Center of k-space: Contains low spatial frequencies, responsible for image contrast, brightness, and general shape.

🧠 Periphery of k-space: Contains high spatial frequencies, responsible for edges, details, and fine structures.

## 🛠 Requirements

To run the script, you need the following Python packages:

pip install numpy matplotlib

## 📁 Data

You can download a test dataset containing example k-space data from the link below:

**[Download Test Dataset](https://drive.google.com/drive/folders/14-C4XG2RXxJ6UIR2E59yeX-NDFvfbpIq?usp=sharing)**

🔗 IXI Dataset

## 🩻 Usage

You can use the "k_space.npy" file directly.
To do so, place the k_space.npy file in the same directory as the script before running it.

If you prefer to use the available NIfTI (.nii) file or any other from the IXI Dataset, first run the script "convert_nii_to_npy.py" to generate a compatible NumPy file.
