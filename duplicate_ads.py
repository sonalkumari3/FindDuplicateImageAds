###Importing important libraries
import cv2
from skimage.measure import compare_ssim
import os
import numpy as np


def mdape(original: np.ndarray, sample: np.ndarray):
    """
    Median Absolute Percentage Error: MdAPE
    """
    return np.median(np.abs((original-sample)/(original+1e-10)))

def hist_features(image, h, w):
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
    '''
    This function takes normalize the input image with given height and width after converting it to gray-scale.
    '''
    # convert the images to gray-scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ##Normalize the image for the given height and width
    resized_image = cv2.resize(gray_image, dim)

    return resized_image


def isDuplicate(image1, image2, feature1, feature2, similarity_threshold=0.7, error_threshold=0.02):
    '''
    This function takes two images and a similarity threshold to find the similarity between them for the given threshold.
    '''
    # Find the Structural Similarity Index (SSIM) between two images
    (ssim_sim, diff) = compare_ssim(image1, image2, full=True)

    # Find the Mean Absolute Percentage Error (MdAPE) between the histogram features of both the images
    mdap_error = mdape(feature1, feature2)

    ### make a decision based on both the scores: ssim and r2_score
    if ((mdap_error < error_threshold) & (ssim_sim > similarity_threshold)):
        print("Duplicate ad with similarity score: {} and MdAPE: {} ".format(ssim_sim, mdap_error))
        cv2.imshow("Duplicate", image2)
        cv2.waitKey(0)


if __name__ == '__main__':
    ### Path of the image directory
    duplicate_ads_path = 'Find_Duplicate_Image/duplicate_ads/'
    #### list of image files
    duplicat_files = [os.path.join(duplicate_ads_path, p) for p in sorted(os.listdir(duplicate_ads_path))]

    ##Get a query image to compare with other images
    template = 'Find_Duplicate_Image/duplicate_ads/17_828296168.jpg'
    # template = 'Find_Duplicate_Image/duplicate_ads/l_1490708845.jpg'
    print(template)
    template = cv2.imread(template)
    w, h, c = template.shape

    ##extract histogram features from the template image
    template_feat = hist_features(template, h, w)

    ## convert to the grayscale
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ## store the width and height of the query image to normalize the rest of images


    ##traverse the image directory to find duplicate ads
    for f in duplicat_files:
        img = cv2.imread(f)
        ##extract histogram features from the image
        img_feat = hist_features(img, h, w)

        img = normalize_image(img, (h, w))
        isDuplicate(template, img, template_feat, img_feat, 0.6, 0.02)
