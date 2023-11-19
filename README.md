<p align="center">
  <img src="https://img.shields.io/badge/Python-3.0-blue?style=flat-square" alt="Python version">
  <img src="https://img.shields.io/badge/TensorFlow-2.0-orange?style=flat-square" alt="TensorFlow version">
  <img src="https://img.shields.io/badge/Keras-2.0-green?style=flat-square" alt="Keras version">
</p>


# BrainMNet: A Unified Neural Network Architecture for Brain Image Classification

Accurate and timely diagnosis is imperative for effective medical intervention in brain-related diseases like Brain Tumors and Alzheimer’s. While contemporary medical imaging approaches tend to focus on individual brain diseases, our GitHub repository introduces BrainMNet, an innovative neural network architecture tailored for classifying brain images and diagnosing various interconnected diseases, with a particular emphasis on brain tumors. The primary objective is to present a unified framework capable of effectively diagnosing a spectrum of brain-related conditions. Our repository provides a thorough validation of BrainMNet's efficacy, specifically highlighting its diagnostic capabilities for Brain Tumor and Alzheimer’s disease. Significantly, our proposed model workflow surpasses current state-of-the-art methods, showcasing substantial improvements in accuracy and precision. Furthermore, it maintains a balanced performance across different classes in the Brain Tumor and Alzheimer’s dataset, underscoring the versatility of our architecture for precise disease diagnosis. This repository represents a noteworthy step towards a unified solution for diagnosing diverse brain-related diseases, initially focusing on Alzheimer's and Brain Tumor while designed to accommodate broader applications.

# About the repository

- The `visualization.py` script exhibits the visualization procedures, offering an in-depth illustration of the various methods employed for visualizing data. It comprehensively showcases the visual representations generated and their impact on the overall understanding of the data.
- `train.py` file showcases the comprehensive training process, providing a detailed demonstration of each step involved in training. Additionally, it presents the outcomes and results generated throughout the training procedure.
- The `Model` directory encompasses two defined Convolutional Neural Network (CNN) architectures and the resulting ensemble architecture derived from them, incorporating the optimal weights.

# Dataset Citations

- The brain tumor classification dataset used in this project is obtained from [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/). It is a combined dataset that incorporates data from [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), [SARTAJ dataset](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri), and [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no). The dataset consists of 7023 human brain MRI images, categorized into four classes: glioma, meningioma, no tumor, and pituitary.
- The Alzheimer's dataset is sourced from Kaggle's [Alzheimer 4 Class Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images), containing a total of 6400 images distributed across four classes: Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented.

# Usage

**Note:** The code is designed dynamically to facilitate the use of the same scripts across various datasets. Consequently, meticulous handling of file paths in the `train.py` and `visualization.py` scripts is essential to ensure effective reproducibility while training and visualizing different datasets.

Steps to be adhered to when running the repository:
- Clone the repository: `git clone https://github.com/sg-research08/Brain-Analysis.git`
- Install the required packages: `pip install -r requirements.txt`
- Run the training script: `python train.py`
- Run the visualization script: `python visualization.py`
