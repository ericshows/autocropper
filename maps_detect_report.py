import cv2
import os
import time
from tqdm import tqdm

# Function to load an image from file
def load_image(file_path):
    return cv2.imread(file_path)

# Function to detect keypoints and compute descriptors for an image using a given feature detector
def detect_keypoints_and_compute_descriptors(image, feature_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = feature_detector.detectAndCompute(gray, None)
    
    return keypoints, descriptors

# Function to match keypoints between two sets of descriptors using a given feature matcher
def match_keypoints(descriptors1, descriptors2, feature_matcher):
    matches = feature_matcher.match(descriptors1, descriptors2)
    return matches

# Function to perform homography estimation and count the inliers
def perform_homography_and_count_inliers(keypoints1, keypoints2, matches):
    src_pts = [keypoints1[m.queryIdx].pt for m in matches]
    dst_pts = [keypoints2[m.trainIdx].pt for m in matches]
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = np.sum(mask)
    return inliers

# Path to the folder containing the images
folder_path = "/Users/ericshows/Downloads/maps"

# List of image file extensions to consider
image_extensions = [".jpg", ".tif"]

# Initialize the feature detectors and descriptor matcher
feature_detectors = {
    "ORB": cv2.ORB_create(),
    "AKAZE": cv2.AKAZE_create(),
    "BRISK": cv2.BRISK_create(),
    "FAST": cv2.FastFeatureDetector_create(),
    "MSER": cv2.MSER_create()
}

descriptor_matcher = cv2.FlannBasedMatcher()

# Dictionary to store the number of inliers and time taken for each algorithm
results = {algorithm: {"inliers": 0, "time": 0.0} for algorithm in feature_detectors}

# Count the total number of image pairs
num_image_pairs = 0
for filename1 in os.listdir(folder_path):
    ext1 = os.path.splitext(filename1)[1]
    if ext1.lower() in image_extensions:
        for filename2 in os.listdir(folder_path):
            ext2 = os.path.splitext(filename2)[1]
            if ext2.lower() in image_extensions:
                num_image_pairs += 1

# Create a progress bar
progress_bar = tqdm(total=num_image_pairs, desc="Processing images")

# Loop through all image files in the folder
for filename1 in os.listdir(folder_path):
    ext1 = os.path.splitext(filename1)[1]
    if ext1.lower() in image_extensions:
        filepath1 = os.path.join(folder_path, filename1)
        image1 = load_image(filepath1)

        for filename2 in os.listdir(folder_path):
            ext2 = os.path.splitext(filename2)[1]
            if ext2.lower() in image_extensions:
                filepath2 = os.path.join(folder_path, filename2)
                image2 = load_image(filepath2)

                for feature_detector_name, feature_detector in feature_detectors.items():
                    try:
                        start_time = time.time()
                        keypoints1, descriptors1 = detect_keypoints_and_compute_descriptors(image1, feature_detector)
                        keypoints2, descriptors2 = detect_keypoints_and_compute_descriptors(image2, feature_detector)
                        matches = match_keypoints(descriptors1, descriptors2, descriptor_matcher)
                        inliers = perform_homography_and_count_inliers(keypoints1, keypoints2, matches)
                        end_time = time.time()
                        algorithm = f"{feature_detector_name}_FLANN"
                        results[algorithm]["inliers"] += inliers
                        results[algorithm]["time"] += end_time - start_time
                    except cv2.error:
                        print(f"Algorithm {feature_detector_name}_FLANN not supported. Skipping...")

                # Update the progress bar
                progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Calculate average number of inliers and time for each algorithm
for algorithm, data in results.items():
    num_pairs_processed = num_image_pairs
    avg_inliers = data["inliers"] / num_pairs_processed
    avg_time = data["time"] / num_pairs_processed
    print("Algorithm:", algorithm)
    print("Average number of inliers:", avg_inliers)
    print("Average time taken (seconds):", avg_time)
    print()
