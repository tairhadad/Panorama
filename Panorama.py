#Shir Bar - 313253668
#Tair Hadad - 204651897

import cv2
from datetime import datetime
import numpy as np
import argparse

'''this function import and open the images from the chosen folder (1 or 2) and return it back'''
def importImages (rightP, leftP):
    print(leftP)
    left = cv2.imread(leftP)
    right = cv2.imread(rightP)

    return left, right

'''smaller the images in 30% in order to improve run time of the code'''
def getSmaller (img):
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

'''This function reasponsible for resize the images, same height , and get it smaller in 30%'''
def preProcessing(leftImg, rightImg):
    # get the images smaller in 30%
    leftImg = getSmaller(leftImg)
    rightImg = getSmaller(rightImg)

    # change the height of the images if necessary. should be equal, weight can be different between the images
    if leftImg.shape[0] < rightImg.shape[0]:
        size = rightImg.shape[1], leftImg.shape[0]
        rightImg = cv2.resize(rightImg, size)
    else:
        size = leftImg.shape[1], rightImg.shape[0]
        leftImg = cv2.resize(leftImg, size)

    return leftImg, rightImg

''' Selecting the best matches between outlines,
we check it by choosing ratio value = 0.85, and then check if the distances between 2 matches smaller than the ratio
We conclude that the match was good and not created following to noises'''
def bestMatches (match):
    good = []
    ratio = 0.85
    for m, n in match:
        if m.distance < n.distance * ratio:
            good.append(m)

    return good

'''Calculation of distances between each outline in one image for each outline in the second image using SIFT algo'''
def findKeypoints (leftImg, rightImg):

    #find the keypoints on each image
    sift = cv2.xfeatures2d.SIFT_create()
    rightKeyporints, rightDescriptor = sift.detectAndCompute(rightImg, None)
    leftKeyporints, leftDescriptor = sift.detectAndCompute(leftImg, None)

    # connect between the keypoints between 2 images
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(rightDescriptor, leftDescriptor, k=2)
    bestMatch = bestMatches (raw_matches)
    return bestMatch, rightKeyporints, leftKeyporints


def warpImages (img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    '''Create two lists with these points,list_of_points_1 represents coordinates of a reference image, 
    and the second list called temp_points represents coordinates of a second image that we want to transform. '''
    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    '''When we have established a homography we need to warp perspective'''
    ''' Change field of view'''
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    '''Warp the second image using the function cv2.warpPerspective()'''
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


'''create an homograpy - transformation that maps the points in one image to the corresponding points in the other image.'''
def HomograpyCreate (matches, rightKP, leftKP):
    # matrix homograpy need at least 4 matches
    if len(matches) >= 4:
        left_pts = np.float32([rightKP[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        right_pts = np.float32([leftKP[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Establish a homography
        H, _ = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)
        return H


parser = argparse.ArgumentParser()
parser.add_argument('p_l',metavar='Path',type=str)
parser.add_argument('p_r',metavar='Path',type=str)
parser.add_argument('p_o',metavar='Path',type=str)
args = parser.parse_args()

if args.p_l and args.p_r and args.p_o:
    p_l = args.p_l[0]
    p_r = args.p_r[0]

leftP = args.p_l
rightP = args.p_r
resP = args.p_o

start_time = datetime.now()
start_time_ = start_time.strftime("%H:%M:%S")
print("start time:" + start_time_)

print("Importing the images....")
leftImgC , rightImgC = importImages(rightP, leftP) # Images with colors
leftImg = cv2.cvtColor(leftImgC, cv2.COLOR_RGB2GRAY) #Convert left to gray
rightImg = cv2.cvtColor(rightImgC, cv2.COLOR_RGB2GRAY)#Convert right to gray

print("Preprocessing....")
leftImg , rightImg = preProcessing(leftImg , rightImg)
leftImgC , rightImgC = preProcessing(leftImgC , rightImgC)

print("Find keypoints....")
Matches, rightKeyporints, leftKeyporints = findKeypoints (leftImg , rightImg)

print("homograpy...")
H_matrix = HomograpyCreate(Matches, rightKeyporints, leftKeyporints)

print("wrap....")
panorama_img= warpImages (rightImgC, leftImgC, H_matrix)

cv2.imshow('res', panorama_img)
cv2.imwrite(resP,panorama_img)

print("Done")
end_time = datetime.now()
end_time = end_time.strftime("%H:%M:%S")
