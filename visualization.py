
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
import cv2
import os
from matplotlib.font_manager import FontProperties

db='Alzheimer_s Dataset/train'
length=0
for path in os.listdir(db):
    fol=db+"/"+path
    print(path,':', len(os.listdir(fol)), 'images')
    length+=len(os.listdir(fol))
print("Total image: ",length)

title_font = FontProperties(weight='bold', size='large')
subfolders = [f.path for f in os.scandir(db) if f.is_dir()]

# Iterate through each subfolder and display the first image
for subfolder in subfolders:
    # List all image files within the subfolder
    image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.jpg')]

    if len(image_files) > 0:
        # Read the first image
        first_image = plt.imread(image_files[0])

        # Display the first image
        plt.imshow(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(subfolder), fontproperties=title_font)
        plt.axis('off')
        plt.savefig(os.path.basename(subfolder), dpi=500)
        plt.show()

subfolders = [f.path for f in os.scandir(db) if f.is_dir()]

# Iterate through each subfolder and display the first image with its histogram
for subfolder in subfolders:
    # List all image files within the subfolder
    image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.jpg')]

    if len(image_files) > 0:
        # Read the first image
        first_image = plt.imread(image_files[0])

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # Display the first image
        ax1.imshow(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(os.path.basename(subfolder), fontproperties=title_font)
        ax1.axis('off')

        # Plot the histogram of the image
        ax2.hist(first_image.ravel(), bins=256, color='red')
        ax2.set_title('Histogram', fontproperties=title_font)
        ax2.set_xlabel('Pixel Value', fontproperties=title_font)
        ax2.set_ylabel('Frequency', fontproperties=title_font)

        plt.tight_layout()
        plt.savefig("Hist"+os.path.basename(subfolder), dpi=500)
        plt.show()

import os
import matplotlib.pyplot as plt

# Function to count the number of files (images) in a directory
def count_files_in_directory(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            count += 1
    return count

# Path to the main folder containing the subfolders
main_folder_path = "Alzheimer_s Dataset/train"

# List of subfolder names
subfolder_names = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

# Count the number of images in each subfolder
counts = [count_files_in_directory(os.path.join(main_folder_path, subfolder)) for subfolder in subfolder_names]

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=subfolder_names, autopct='%1.1f%%', startangle=140)
plt.title("Data Distribution of Classes", weight='bold', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
plt.savefig("Data Distribution.png", dpi=500)
plt.show()