import cv2
import os
import time
from tqdm import tqdm

# Define the folder path containing the images
folder_path = "/Users/ericshows/Downloads/maps"

# Define the feature detection and matching techniques to compare
techniques = ["ORB", "FAST"]
matching_technique = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def detect_and_match_points(image1, image2, technique):
    if technique == "ORB":
        detector = cv2.ORB_create()
    elif technique == "FAST":
        detector = cv2.FastFeatureDetector_create()

    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    matches = matching_technique.match(descriptors1, descriptors2)
    num_matches = len(matches)

    return keypoints1, num_matches

# Initialize dictionaries to store the results
avg_points_detected = {technique: 0 for technique in techniques}
avg_speed_of_detection = {technique: 0 for technique in techniques}
avg_matches = {technique: 0 for technique in techniques}
avg_speed_of_matching = {technique: 0 for technique in techniques}
image_count = 0

# Get the total number of images
total_images = len([filename for filename in os.listdir(folder_path) if filename.endswith(".jpg") or filename.endswith(".tif")])

# Create a progress bar
progress_bar = tqdm(total=total_images, desc="Processing Images", unit="image")

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".tif"):
        image_count += 1
        file_path = os.path.join(folder_path, filename)

        # Load the images
        image1 = cv2.imread(file_path, 0)  # Read as grayscale
        image2 = cv2.imread(file_path, 0)  # Read as grayscale

        # Process each technique
        for technique in techniques:
            start_time = time.time()

            # Detect and match points using the technique
            keypoints, num_matches = detect_and_match_points(image1, image2, technique)

            end_time = time.time()
            execution_time = end_time - start_time

            # Update the results dictionaries
            avg_points_detected[technique] += len(keypoints)
            avg_speed_of_detection[technique] += execution_time
            avg_matches[technique] += num_matches
            avg_speed_of_matching[technique] += execution_time

        # Update the progress bar
        progress_bar.update(1)

# Calculate average results
for technique in techniques:
    avg_points_detected[technique] /= image_count
    avg_speed_of_detection[technique] /= image_count
    avg_matches[technique] /= image_count
    avg_speed_of_matching[technique] /= image_count

# Close the progress bar
progress_bar.close()

# Print the results
print("Average Points Detected:")
for technique in techniques:
    print(f"{technique}: {avg_points_detected[technique]}")

print("\nAverage Speed of Detection (seconds):")
for technique in techniques:
    print(f"{technique}: {avg_speed_of_detection[technique]}")

print("\nAverage Matches:")
for technique in techniques:
    print(f"{technique}: {avg_matches[technique]}")

print("\nAverage Speed of Matching (seconds):")
for technique in techniques:
    print(f"{technique}: {avg_speed_of_matching[technique]}")
