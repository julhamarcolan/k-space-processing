# K-Space  and MRI Image Reconstruction

This repository contains a Python script for  k-space data processing (frequency domain) and reconstructing MRI images using low-pass and high-pass filtering techniques. It also includes visualizations comparing original, high-frequency, and low-frequency reconstructions.

## 🔬 What is K-space?

In Magnetic Resonance Imaging (MRI), k-space is a matrix that stores the raw frequency-encoded data captured during a scan. This frequency information is not an image by itself, but a spatial frequency representation of the object. Applying an inverse Fourier transform to k-space data generates the actual MRI image in the spatial domain.

Understanding how spatial frequencies are distributed in k-space is essential:

🎯 Center of k-space: Contains low spatial frequencies, responsible for image contrast, brightness, and general shape.

🧠 Periphery of k-space: Contains high spatial frequencies, responsible for edges, details, and fine structures.

Refer to the diagram below (source: educational MRI content) for a visual understanding:

## 🛠 Requirements

To run the script, you need the following Python packages:

pip install numpy matplotlib

## 📁 Data

You can download a test dataset containing example k-space data from the link below:

**[Download Test Dataset](https://drive.google.com/drive/folders/14-C4XG2RXxJ6UIR2E59yeX-NDFvfbpIq?usp=sharing)**

Place the k_space.npy file in the same directory as the script before running it.
