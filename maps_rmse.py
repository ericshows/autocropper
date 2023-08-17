import numpy as np
from PIL import Image
import os

def compare_images_rmse(image1_path, image2_path):
    # Load images
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    # Resize images to the same dimensions
    min_width = min(image1.width, image2.width)
    min_height = min(image1.height, image2.height)
    image1 = image1.resize((min_width, min_height))
    image2 = image2.resize((min_width, min_height))

    # Convert images to numpy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Calculate squared differences
    squared_diff = np.square(array1.astype(np.float32) - array2.astype(np.float32))

    # Compute mean squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_diff)

    return rmse

# Directory path for the image files
directory = '/Users/ericshows/Downloads/maps_cropped/QC'

# List all files in the directory
files = os.listdir(directory)

# Keep track of RMSE values
rmse_values = []

# Iterate over each file in the directory
for file in files:
    if file.endswith('w.jpg'):
        # Get the corresponding south image file name
        south_file = file.replace('w.jpg', 's.jpg')

        # Construct the full file paths
        image1_path = os.path.join(directory, file)
        image2_path = os.path.join(directory, south_file)

        # Calculate and print the RMSE for each pair
        rmse_value = compare_images_rmse(image1_path, image2_path)
        print("RMSE for", file, "and", south_file, ":", rmse_value)

        # Add the RMSE value to the list
        rmse_values.append(rmse_value)

# Compute and print the average RMSE
avg_rmse = np.mean(rmse_values)
print("Average RMSE:", avg_rmse)
