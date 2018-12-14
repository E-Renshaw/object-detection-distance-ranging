import cv2
import os
import numpy as np
from pathlib import Path

def stereo_disparity(filepath_left, filepath_right, crop_disparity=False):
    """Calculate the stereo disparity between two images

    Args:
        filepath_left (string): location of the left image
        filepath_right (string): loaction of the right image
        crop_disparity (bool): crop disparity to chop out left part where there
            are with no disparity as this area is not seen by both cameras and
            also chop out the bottom area (where we see the front of car bonnet)
        show_disparity (bool): show the calculated disparity

    Returns:
        disparity: the stereo disparity of the two images
    """
    max_disparity = 128
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

    imgL = cv2.imread(filepath_left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(filepath_right, cv2.IMREAD_COLOR)

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # preprocess images by raising to a power - subjectively better response
    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')

    disparity = stereoProcessor.compute(grayL,grayR)

    # apply a noise filter
    dispNoiseFilter = 5 # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    if (crop_disparity):
        width = np.size(disparity_scaled, 1)
        disparity_scaled = disparity_scaled[0:390,135:width]

    output_image = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)

    return output_image
