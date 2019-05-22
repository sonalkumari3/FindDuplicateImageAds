
###Proposed solution Approach

 

# FindDuplicateImageAds
This code compare the given image with all the images exist in a given directory and find duplicates based on two similarity metrics. The main purpose of this is to find the duplicate image ads.

 

It uses following python libraries:

import cv2

from skimage.measure import compare_ssim

import numpy as np
 

 


It takes one image as a query image and onother image to match and find if it is it's duplicate.
The image is read using cv2 library in template variable.


Image histogram features have been computed for all three (r, g, & b) channels (see function definition hist_features()).
 
 
Both the images have been converted to gray-scale before computing image similarity using python in built function cvtColor under cv2.


Images are normalized with the target image height and width for the comparison purpose (see normalize_image() function). The "normalize_image()" function first convert the image to grayscale and then normalize it to a given height and width.



For finding similarity between normalized images, isDuplicate() function is defined. It takes two images, their histogram based feature vectors, a similarity threshold and an error threshold. Threshold can be fine-tuned based on given duplicate and non-duplicate image ads. 
Following two similarity metrics have been used to get better accuracy: 1) Structural Similarity Index (SSI) and 2) Median Absolute Percentage Error (MdAPE).
Similarity between images are being computed based on Structural Similarity Index (compare_ssim) which is already defined in skimage.measure library. 
Dissimilarity between histogram features are being computed using Median Absolute Percentage Error. To mark an image as duplicate, similarity threshold and error threshold are set to 0.6 and 0.02, respectively. However, these parameters are user defined and empirically selected based on few sample image ads. Moreover, this is a sub-optimal solution and can be further improved in following ways:
New features (such as edges, object boundary, picture description if exists, etc.) can be extracted.
Different similarity and dissimilarity metrics can be explored.
Threshold can be fine-tuned.
 

