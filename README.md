# autocropper

Automated Image Cropping Using Computer Vision Techniques



For large matched sets of high-resolution images that include an uncropped master tif with color bars and borders surrounding subject (u.tif files in this case) and low-resolution cropped jpeg files (w.jpg files here). Autocropper will detect and match features in pairs of images and automatically crop and save a new high-resolution cropped image for zoomable web usage. Autocropper also converts new files into srgb color space. 



Dependencies:
 - cv2 (openCV)
 - numpy
 - tkinter (if preview application is desired)
 - PIL



To run:
 - crop_match_cropper.py is final working script
 - check_invalid function and tkinter preview applications are commented out by default