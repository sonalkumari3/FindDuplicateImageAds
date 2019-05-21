
###Proposed solution Approach

 

# FindDuplicateImageAds
This code compare the given image with all the images exist in a given directory and find duplicates based on two similarity metrics. The main purpose of this is to find the duplicate image ads.

 

It uses following python libraries:

import cv2

from skimage.measure import compare_ssim

import os

import numpy as np
 

 


It takes one image as a query image and traverse each of the image in the directory to find it’s duplicate.
The query image is read using cv2 library in template variable (see Line 65-68 in duplicate_ads.py).


Image histogram features have been computed for all three (r, g, & b) channels (see function definition “hist_features()” at Line 14-26).
 
 
Image has been converted to gray-scale before computing image similarity (see Line 75) using python in built function cvtColor under cv2.
 
 
 
The directory (named as duplicate_ads), in which images are stored, is traversed to find the similar ads.


Images are normalized with the target image height and width for the comparison purpose (see Line 86 in duplicate_ads.py). The "normalize_image()" function (see Line 28-38) first convert the image to grayscale and then normalize it to a given height and width.



For finding similarity between normalized images, isDuplicate() function is defined (see Line 41-55). It takes two images, their histogram based feature vectors, a similarity threshold and an error threshold. Threshold can be fine-tuned based on given duplicate and non-duplicate image ads. 
Following two similarity metrics have been used to get better accuracy: 1) Structural Similarity Index (SSI) and 2) Median Absolute Percentage Error (MdAPE).
Similarity between images are being computed based on Structural Similarity Index (compare_ssim) which is already defined in skimage.measure library. 
Dissimilarity between histogram features are being computed using Median Absolute Percentage Error. To mark an image as duplicate, similarity threshold and error threshold are set to 0.6 and 0.02, respectively. However, these parameters are user defined and empirically selected based on few sample image ads. Moreover, this is a sub-optimal solution and can be further improved in following ways:
New features (such as edges, object boundary, picture description if exists, etc.) can be extracted.
Different similarity and dissimilarity metrics can be explored.
Threshold can be fine-tuned.
 

