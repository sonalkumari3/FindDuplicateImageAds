import cv2
from skimage.measure import compare_ssim
import os


def normalize_image(image, dim = (224, 224)):
    # convert the images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, dim)
    # cv2.imshow("resized_image", resized_image)
    # cv2.waitKey(0)
    return resized_image

def similarity_metrics(image1, image2, similarity_threshold = 0.5):
    # Find the Structural Similarity Index (SSIM) between the two images
    (ssim_sim, diff) = compare_ssim(image1, image2, full=True)
    # print("ssim_sim: {}".format(ssim_sim))

    if(ssim_sim > similarity_threshold):
        print("Duplicate ad")


if __name__ == '__main__':
    duplicate_ads_path = 'duplicate_ads/'
    non_duplicat_ads_path = 'non_duplicate_ads/'

    non_duplicat_files = [os.path.join(non_duplicat_ads_path, p) for p in sorted(os.listdir(non_duplicat_ads_path))]
    duplicat_files = [os.path.join(duplicate_ads_path, p) for p in sorted(os.listdir(duplicate_ads_path))]

    ##query image to compare with other ads
    template = duplicat_files[1]
    print(template)
    template = cv2.imread(template)
    # print(template.shape)
    w, h, shape = template.shape
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ##traverse the image directory to find duplicate ads
    for s in duplicat_files:
    # for s in non_duplicat_files:
        print (s)
        img = cv2.imread(s)
        img = normalize_image(img, (h, w))
        similarity_metrics(template, img)


