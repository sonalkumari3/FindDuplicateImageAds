###Importing important libraries
import cv2
from skimage.measure import compare_ssim
import os


def normalize_image(image, dim = (224, 224)):
    '''
    This function takes normalize the input image with given height and width after convering it to grayscale.
    '''
    # convert the images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, dim)
    # cv2.imshow("resized_image", resized_image)
    # cv2.waitKey(0)
    return resized_image

def similarity_metrics(image1, image2, similarity_threshold = 0.5):
    '''
    This function takes two images and a similarity threshold to find the similarity between them for the given threshold.
    '''
    # Find the Structural Similarity Index (SSIM) between the two images
    (ssim_sim, diff) = compare_ssim(image1, image2, full=True)
    # print("ssim_sim: {}".format(ssim_sim))

    if(ssim_sim > similarity_threshold):
        print("Duplicate ad")


if __name__ == '__main__':
    ### Path of the image directory
    duplicate_ads_path = 'duplicate_ads/'
    #### list of image files
    duplicat_files = [os.path.join(duplicate_ads_path, p) for p in sorted(os.listdir(duplicate_ads_path))]

    ##Get a query image to compare with other images
    template = duplicat_files[1]
#     print(template)
    template = cv2.imread(template)
    # print(template.shape)
    
    ## convert to the grayscale
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    ## store the width and height of the query image to normalize the rest of images
    w, h = template.shape

    ##traverse the image directory to find duplicate ads
    for f in duplicat_files:
#         print (f)
        img = cv2.imread(f)
        img = normalize_image(img, (h, w))
        similarity_metrics(template, img)
