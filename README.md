# FindDuplicateImageAds
This code compare the images and identify similar image. The main purpose of this is to find the duplicate image ads.
It takes one image as a query image and rest of the images in the directory to find if there exists any of its duplicate.
The query image is read using cv2 library in template variable (see Line 36-43 in duplicate_ads.py).
The directory (named as duplicate_ads), in which images are stored, is traversed to find the similar ads.
Images are normalized for comparison purpose (see Line 52 in duplicate_ads.py). The "normalize_image" function is defined to normalize the image to a given height and width.
For finding similarity between normalized images, similarity_metrics function is defined. It takes two images to compare and a threshold to mark the duplicate ads. Threshold can be fine-tuned based on given duplicate and non-duplicate image ads. Similarity metric is defined based on Structural Similarity Index (compare_ssim) which is already defined in skimage.measure library.
