# autocropper
Automated Image Cropping Using Computer Vision Techniques

For large sets of high-resolution images where an uncropped master tif with color bars and borders surrounding subject are matched with low-resolution cropped jpeg files. Autocropper will detect and match features in pairs of images and automatically crop and save a new high-resolution cropped image for zoomable web usage. Autocropper also converts new files into srgb color space. 



Dependencies:
 - cv2 (openCV)
 - numpy
 - tkinter (if preview application is desired)
 - PIL



To run:
 - Place srgb color profile in correct system folder (if not already present)
 - Change folder destinations to match local system after cloning
 - crop_match_cropper.py is final working script

check_invalid function and tkinter preview applications are commented out by default