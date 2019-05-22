"""
@Author: sonal.kumari1910@gmail.com

This code find the duplicate image ads.
You need to give two images file names with their correct path. It will tell you whether the given images are duplicate of each other or not.
"""


###Importing important libraries
import cv2
from skimage.measure import compare_ssim
# import os
import numpy as np


def mdape(original: np.ndarray, other: np.ndarray):
    """
    This function computes Median Absolute Percentage Error (MdAPE).

    Parameters
    ----------
        original:      It is the actual or original values array
        other:         It is the predicted or duplicate values array which are getting compared w.r.t original

    Returns
    ----------
        an error (MdAPE) value: quantify the difference between the original and the other
    """

    return np.median(np.abs((original-other)/(original+1e-10)))

def hist_features(image, h, w):
    """
    This function computes the histogram for all three channels in the color image.

    Parameters
    ----------
        image:      a 3-dimensional (width, height, channel) color image
        h:          height of the image
        w:          width of the image

    Returns
    ----------
        features_1d:  1-dimensional histogram feature in the form of numpy array
    """
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # Traverse the channel and generate the histograms for each channel to get histogram based features
        hist = cv2.calcHist([chan], [0], None, [w], [0, h])
        features.extend(hist)

    features_1d = np.array(features).flatten()
    # print("features shape: %d" % (features_1d.shape))
    return features_1d

def normalize_image(image, dim=(224, 224)):
    """
    This function first convert the color image into grayscale and then rescale it to a given width and height.

    Parameters
    ----------
        image:      a 3-dimensional (width, height, channel) color image
        dim(w, h):  a tuple of (width, height) of the desired scaled image

    Returns
    ----------
        resized_image:  gray-scale, resized image
    """
    # convert the images to gray-scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ##Normalize the image for the given height and width
    resized_image = cv2.resize(gray_image, dim)

    return resized_image


def isDuplicate(image1, image2, feature1, feature2, similarity_threshold=0.7, error_threshold=0.02):
    """
    This function takes two images and a similarity threshold to find the similarity between them for the given threshold.
    If both the images are duplicate of each other then, it will print the similarity-score and show the duplicate image.
    Press enter key to return to exit from this function.

    Parameters
    ----------
        image1:                    a gray-scale image
        image2:                    another gray-scale image of same size as that of image1
        feature1:                  features extracted from image1 in the form of 1-d numpy array
        feature2:                  features extracted from image2 in the form of 1-d numpy array (size is same as that of feature1)
        similarity_threshold:      similarity threshold in the range of 0(lowest similarity) to 1(highest similarity)
        error_threshold:           error (dissimilarity) threshold in the range of 0(highest similarity) to 1(lowest similarity)


    """

    # Find the Structural Similarity Index (SSIM) between two images
    (ssim_sim, diff) = compare_ssim(image1, image2, full=True)

    # Find the Mean Absolute Percentage Error (MdAPE) between the histogram features of both the images
    mdap_error = mdape(feature1, feature2)

    ### make a decision based on both the scores: ssim and r2_score
    if ((mdap_error < error_threshold) & (ssim_sim > similarity_threshold)):
        print("Duplicate ad with similarity score: {} and MdAPE: {} ".format(ssim_sim, mdap_error))
        # cv2.imshow("Duplicate", image2)
        # cv2.waitKey(0)
    else: print("Images belong to two unique ads (Not duplicate)")

def run(template, other):
    template_img = cv2.imread(template)

    ## store the width and height of the query image to normalize the rest of images
    w, h, c = template_img.shape

    ##extract histogram features from the template image
    template_feat = hist_features(template_img, h, w)

    ## convert to the grayscale
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    ###read other image
    img = cv2.imread(other)

    ##extract histogram features from the image
    img_feat = hist_features(img, h, w)

    img = normalize_image(img, (h, w))
    isDuplicate(template_img, img, template_feat, img_feat, 0.6, 0.02)


    ### The below peace of code is to traverse through all the images in the directory to find the duplicates
    ##duplicat_ads = 'Find_Duplicate_Image/duplicate_ads/'
    # ##traverse the image directory to find duplicate ads
    # for f in duplicat_ads:
    #     img = cv2.imread(f)
    #     ##extract histogram features from the image
    #     img_feat = hist_features(img, h, w)
    #
    #     img = normalize_image(img, (h, w))
    #     isDuplicate(template_img, img, template_feat, img_feat, 0.6, 0.02)


if __name__ == '__main__':
    ##Get a query image to compare with other images
    template = 'Find_Duplicate_Image/duplicate_ads/17_828296168.jpg'
    # print(template)

    ###get the path of other image
    other = 'Find_Duplicate_Image/duplicate_ads/l_1490708845.jpg'

    run(template, other)

