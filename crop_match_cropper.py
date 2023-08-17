import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk, ImageCms
import subprocess
import math
import shutil

def check_invalid():

    folder_path = "./maps"
    invalid_folder_path = "./maps_invalid"

    # Create the invalid TIFF folder if it doesn't exist
    if not os.path.exists(invalid_folder_path):
        os.makedirs(invalid_folder_path)

    # Get a list of files in the folder
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        
        # Check if it's a TIFF or JPEG file
        if file.lower().endswith(('.tif', '.tiff')):
            try:
                # Open the image using PIL
                img = Image.open(file_path)
                img.verify()  # Verify the well-formedness of the TIFF
            except (IOError, SyntaxError) as e:
                # Move the invalid TIFF to the invalid folder
                new_file_path = os.path.join(invalid_folder_path, file)
                shutil.move(file_path, new_file_path)
                print(f"Moved invalid TIFF: {file}")



# Function to display the target and output images with accept and reject buttons
def display_preview(target_image, output_image):
    # Create a Tkinter window
    window = tk.Tk()

    # Convert the target image to RGB color space
    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Check if the output image is valid
    if output_image is not None and len(output_image) > 0:

        # Convert the cropped image to RGB color space
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Create a PIL image from the target image and output image
        target_pil = Image.fromarray(target_rgb)
        output_pil = Image.fromarray(output_rgb)

        # Resize the output image to fit the window
        width, height = target_pil.size
        output_pil = output_pil.resize((width, height))

        # Convert the PIL images to Tkinter-compatible images
        target_tk = ImageTk.PhotoImage(target_pil)
        output_tk = ImageTk.PhotoImage(output_pil)

        # Create a Tkinter label for the target image
        target_label = tk.Label(window, text='760px w file', image=target_tk, compound='bottom')
        target_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a Tkinter label for the output image
        output_label = tk.Label(window, text='s file crop preview', image=output_tk, compound='bottom')
        output_label.pack(side=tk.LEFT, padx=10, pady=10)

    else:
        # If the output image is not valid, display a message
        error_label = tk.Label(window, text="Invalid output image")
        error_label.pack(side=tk.LEFT, padx=10, pady=10)

    # Function to handle the accept button click
    def accept_crop():
        nonlocal accept
        accept = True
        window.destroy()

    # Function to handle the reject button click
    def reject_crop():
        nonlocal accept
        accept = False
        window.destroy()

    # Create the accept button
    accept_button = tk.Button(window, text="Accept", command=accept_crop)
    accept_button.pack(side=tk.BOTTOM, padx=10, pady=10)

    # Create the reject button
    reject_button = tk.Button(window, text="Reject", command=reject_crop)
    reject_button.pack(side=tk.BOTTOM, padx=10, pady=10)

    # Initialize the accept variable
    accept = None

    # Start the Tkinter event loop
    window.mainloop()

    return accept


def convert_tif_to_srgb(input_path, output_path):
    # Specify the list of problematic tags to ignore (comma-separated)
    tag_list = "PixelXDimension,PixelYDimension"

    # Use ImageMagick to convert the input image to sRGB, ignoring problematic tags and CRC errors, and silencing warnings
    command = f"magick {input_path} -colorspace sRGB -define tiff:ignore-tags={tag_list} -define tiff:ignore-crc -profile '/System/Library/ColorSync/Profiles/sRGB Profile.icc' 2>/dev/null {output_path}"
    subprocess.run(command, shell=True)


def get_rotation_angle(template_img, map_img):
    # Convert images to grayscale
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(map_gray, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match keypoints using FLANN matcher
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter good matches based on Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Estimate rotation angle using RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Calculate the rotation angle using the transformation matrix
    angle_rad = math.atan2(M[1, 0], M[0, 0])
    angle_deg = math.degrees(angle_rad)

    # Round the rotation angle to the nearest 90-degree increment
    rounded_angle_deg = round(angle_deg / 90.0) * 90.0
    return rounded_angle_deg


def rotate_images(target_path, output_folder):
    # Load the w.jpg and u.tif images
    w_img = cv2.imread(target_path, cv2.IMREAD_COLOR) 
    u_img = cv2.imread(converted_path, cv2.IMREAD_COLOR)

    # Get the rotation angle
    angle = get_rotation_angle(w_img, u_img)

    if angle is not None:
        # Calculate the number of 90-degree rotations
        num_rotations = int(angle / 90.0)

        # Perform the rotation(s)
        rotated_u_img = np.rot90(u_img, k=num_rotations)

        # Save the rotated u.tif image
        output_path = os.path.join(output_folder, filename.replace('w.jpg', 's.tif'))
        cv2.imwrite(output_path, rotated_u_img)

        print(f"Image '{filename.replace('w.jpg', 'u.tif')}' rotated by {angle} degrees.")

    else:
        print(f"No matching 'u.tif' file found for '{filename}'.")


# Function to perform the cropping and saving of the cropped image
def crop_and_save_image(target_path, rotated_path, output_folder):
    # Load the target image (smaller cropped JPEG)
    target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Load the source image (larger TIFF photo)
    rotated_image = cv2.imread(rotated_path, cv2.IMREAD_COLOR)

    # Initialize the feature detector and descriptor
    detector = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher()

    # Detect keypoints and compute descriptors for the target image
    target_keypoints, target_descriptors = detector.detectAndCompute(target_image, None)

    # Detect keypoints and compute descriptors for the source image
    source_keypoints, source_descriptors = detector.detectAndCompute(rotated_image, None)

    # Check if descriptors are computed successfully
    if target_descriptors is None or source_descriptors is None:
        print("Descriptor computation failed.")
        exit()

    # Perform feature matching using k-nearest neighbors
    k = 2  # Number of nearest neighbors to consider
    matches = matcher.knnMatch(target_descriptors, source_descriptors, k=k)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    target_points = np.float32([target_keypoints[m.queryIdx].pt for m in good_matches])
    source_points = np.float32([source_keypoints[m.trainIdx].pt for m in good_matches])

    # Find the perspective transformation between the matched keypoints
    M, mask = cv2.findHomography(target_points, source_points, cv2.RANSAC, 5.0)

    # Calculate the rotation angle
    rotation_rad = math.atan2(M[1, 0], M[0, 0])
    rotation_deg = math.degrees(rotation_rad)

    # Calculate the corners of the target image
    h, w = target_image.shape
    target_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])

    # Transform the corners based on the estimated rotation angle
    transformed_corners = cv2.perspectiveTransform(target_corners.reshape(1, -1, 2), M).reshape(-1, 2)

    # Rotate the transformed corners based on the estimated rotation angle
    center_x = (transformed_corners[0, 0] + transformed_corners[2, 0]) / 2
    center_y = (transformed_corners[0, 1] + transformed_corners[2, 1]) / 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, 1.0)
    rotated_corners = cv2.transform(np.array([transformed_corners]), rotation_matrix).squeeze()

    # Find the new minimum and maximum coordinates after rotation
    x_coords = rotated_corners[:, 0]
    y_coords = rotated_corners[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Round the coordinates to integers for cropping
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Calculate the width and height of the output image based on the rotated region of interest
    output_width = x_max - x_min
    output_height = y_max - y_min

    # Perform the rotation
    rows, cols = rotated_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, 1.0)
    rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (cols, rows))

    # Crop the rotated image to match the dimensions implied by the rotated region of interest
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]

    # Check if the cropped image is empty
    if cropped_image.size == 0:
        print("Cropped image is empty.")
        pass

    # Resize the cropped image to the calculated output width and height
    output_image = cv2.resize(cropped_image, (output_width, output_height))

    # Save the rotated and cropped image
    output_path = os.path.join(output_folder, os.path.basename(rotated_path))
    cv2.imwrite(output_path, output_image)
    print("Rotated and cropped image saved:", output_path)




# Folder paths
input_folder = "./maps"  # Folder containing target and source images
converted_folder = "./maps_converted"
rotated_folder = "./maps_rotated"
output_folder = "./maps_cropped"  # Output folder for cropped images
icc_path = "./sRGB Profile.icc"


# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('w.jpg'):
        # Construct the paths for the target and source images
        target_path = os.path.join(input_folder, filename)
        source_filename = filename.replace('w.jpg', 'u.tif')
        source_path = os.path.join(input_folder, source_filename)
        converted_filename = source_filename.replace('u.tif', 's.tif')
        converted_path = os.path.join(converted_folder, converted_filename)
        rotated_path = os.path.join(rotated_folder, converted_filename)
        output_path = os.path.join(output_folder, os.path.basename(rotated_path))
        # Check if the source image exists
        if os.path.isfile(source_path):
            # check_invalid()
            # if not output_path:
            convert_tif_to_srgb(source_path,converted_path)
            # Rotate u.tifs to match orientation of w.jpgs 
            rotate_images(target_path, rotated_folder)
            # Perform cropping and display the preview
            crop_and_save_image(target_path, rotated_path, output_folder)
            os.remove(converted_path)
            os.remove(rotated_path)
            convert_tif_to_srgb(output_path,output_path)
        else:
            print("Source image not found for:", target_path)



