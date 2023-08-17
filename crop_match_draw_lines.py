import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import random

MIN_MATCH_COUNT = 2

def get_matched_coordinates(template_path, map_path):
    """
    Gets template and map image paths and returns matched coordinates in map image

    Parameters
    ----------
    template_path: str
        path to the image to be used as template

    map_path: str
        path to the image to be searched in

    Returns
    ---------
    ndarray
        an array that contains matched coordinates

    """

    # Read template and map images
    temp_img_gray = cv2.imread(template_path, 0)
    map_img_gray = cv2.imread(map_path, 0)

    # Equalize histograms
    temp_img_eq = cv2.equalizeHist(temp_img_gray)
    map_img_eq = cv2.equalizeHist(map_img_gray)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(temp_img_eq, None)
    kp2, des2 = sift.detectAndCompute(map_img_eq, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches by knn which calculates point distance in 128 dim
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = temp_img_eq.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # Matched coordinates

        # Generate a random color for each line segment
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(dst) - 1)]
        for i in range(len(dst) - 1):
            color = colors[i]
            map_img_eq = cv2.polylines(map_img_eq, [np.int32([dst[i], dst[i+1]])], False, color, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # Update draw_params to use random colors
    draw_params = dict(matchColor=None,  # Do not draw matches
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw template and map image, matches, and keypoints
    img3 = cv2.drawMatches(temp_img_eq, kp1, map_img_eq, kp2, good, None, **draw_params)

    return dst, img3



if __name__ == "__main__":
    folder_path = "/Users/ericshows/Downloads/maps_test"
    output_dir = "/Users/ericshows/Downloads/maps_drawn_test"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".tif"):
            # Generate the template and map image paths
            template_path = os.path.join(folder_path, filename[:-5] + "w.jpg")
            map_path = os.path.join(folder_path, filename[:-5] + "u.tif")

            # Get the matched coordinates and result image
            coords, result_img = get_matched_coordinates(template_path, map_path)

            # Generate the output file path
            output_path = os.path.join(output_dir, filename[:-4] + "_drawn.tif")

            # Save the result image as TIF
            cv2.imwrite(output_path, result_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            print(f"Processed: {filename} -> Output: {output_path}")
