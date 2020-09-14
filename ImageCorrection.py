import ImageObj as imgObj
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
import cv2
import math
import random as rng
import os
from datetime import datetime
import platform
from os import listdir
from os.path import isfile, join
import imageio
from plantcv import plantcv as pcv


def correct_angle(img_obj, idx=None):
    print("Correcting image angle")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '3_ANGLE_CORRECTED_FRAME' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '3_ANGLE_CORRECTED_FRAME' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    image = img_obj.get_image()
    queryImg = image
    img = queryImg.copy() - create_mask_filled_by_plants(queryImg)

    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax1.imshow(img, cmap="gray")
    # ax1.set_xlabel("Difference result image", fontsize=14)

    print("Shape: ", img.shape)
    center = [int(np.ceil(img.shape[0] / 2)), int(np.ceil(img.shape[1] / 2))]
    print("Center: ", center)
    img = img[(center[0] - int(np.ceil(img.shape[0] * 0.2))):(center[0] + int(np.ceil(img.shape[0] * 0.2))),
          (center[1] - int(np.ceil(img.shape[1] * 0.2))):(center[1] + int(np.ceil(img.shape[1] * 0.2)))]
    cropped_img = img

    # 1- Convert from RGB to grayscale (cvCvtColor)
    # LAB_image = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    # gray_img = cv2.cvtColor(LAB_image,cv2.COLOR_BGR2GRAY)

    # img = temp_image
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img_luv = img_luv[:, :, 0]
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax1.imshow(img_luv)
    # ax1.set_xlabel("Layer L of LUV image before morphology on RGB color space", fontsize=14)

    # ---------------------------------------------------------------------------
    # Morphology with RGB color space
    cols = img.shape[1]
    horizontal_size = cols // int(np.ceil(cols * 0.2))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    for i in range(0, 2, 1):
        img_erode = cv2.erode(img, horizontalStructure)  # ---- resalta lo brillante en linea recta
        img = cv2.dilate(img_erode, horizontalStructure)  # ---- reduce los artefactos oscuros
        print("Morphology with RGB color space")
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax1.imshow(img)
    # ax1.set_xlabel("Image after morphology on RGB color space", fontsize=14)

    # Morpholoy with 'l' layer from 'LUV' color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img = img[:, :, 0]
    # fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax1.imshow(img)
    # ax1.set_xlabel("Layer L of LUV image after morphology on RGB color space", fontsize=14)
    # ---------------------------------------------------------------------------
    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = img

    # 2 -  # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    # gray_img = cv2.bitwise_not(gray_img)

    '''fig, (ax3) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
    ax3.imshow(gray_img, cmap="gray")
    ax3.set_xlabel("Gray image bitwise_not ", fontsize=14)'''

    bw_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 15, -2)

    # fig, (ax4) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax4.imshow(bw_img, cmap="gray")
    # ax4.set_xlabel("Adaptative threshold", fontsize=14)

    # 3 - Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw_img)
    vertical = np.copy(bw_img)

    # 4-  [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // int(np.ceil(cols * 0.5))
    print("# COLS: ", cols)
    # horizontal_size = cols // 100
    # 4.1 - Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # 4.2 - Apply morphology operations
    # horizontal = cv2.erode(horizontal, horizontalStructure)
    # horizontal = cv2.dilate(horizontal, horizontalStructure)

    # 4.3 - Show extracted horizontal lines
    '''fig, (ax5) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
    ax5.imshow(horizontal, cmap="gray")
    ax5.set_xlabel("Extracted horizontal lines", fontsize=14)'''

    # 5 [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize_1 = rows // int(np.ceil(rows * 0.15))

    # 5.1 - Create structure element for extracting vertical lines through morphology operations
    verticalStructure_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize_1))

    # 5.2 - Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure_1)
    vertical = cv2.dilate(vertical, verticalStructure_1)

    #  fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical, cmap="gray")
    # ax6.set_xlabel("Extracted vertical lines up 15%", fontsize=14)

    # Analysis of connected componentes
    label_im, nb_labels = ndimage.label(vertical)

    sizes_labels = ndimage.sum(vertical, label_im, range(nb_labels + 1))

    mask_size = sizes_labels < (cols * rows) * 0.5
    remove_pixel = mask_size[label_im]

    vertical[remove_pixel] = 0
    #  fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical, cmap="gray")
    # ax6.set_xlabel("Small objects removed", fontsize=14)

    verticalsize_2 = rows // int(np.ceil(rows * 0.25))
    verticalStructure_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize_2))
    vertical_2 = cv2.erode(vertical, verticalStructure_2)
    vertical_2 = cv2.dilate(vertical_2, verticalStructure_2)
    # fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical_2, cmap="gray")
    # ax6.set_xlabel("Extracted vertical lines up 25%", fontsize=14)

    # 5.3 - Show extracted vertical lines
    '''fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
    ax6.imshow(vertical, cmap="gray")
    ax6.set_xlabel("Extracted vertical lines", fontsize=14)'''

    vertical_final = vertical - vertical_2
    # fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical_final, cmap="gray")
    # ax6.set_xlabel("Extracted vertical lines up 15% from 25%", fontsize=14)

    # Analysis of connected componentes
    label_im, nb_labels = ndimage.label(vertical_final)

    sizes_labels = ndimage.sum(vertical_final, label_im, range(nb_labels + 1))

    mask_size = sizes_labels < (cols * rows) * 0.005  # previous value = '0.01'
    remove_pixel = mask_size[label_im]

    vertical_final[remove_pixel] = 0
    # fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical_final, cmap="gray")
    # ax6.set_xlabel("Small objects removed", fontsize=14)

    vertical_final = cv2.dilate(vertical_final, horizontalStructure)
    # fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    # ax6.imshow(vertical_final, cmap="gray")
    # ax6.set_xlabel("Extracted horizontal structure", fontsize=14)
    # 9 - Getting major line measure
    blackAndWhiteImage = vertical_final
    result = get_largest_line_setup(vertical_final)
    print("Orientation parameters: ", result)

    # final_rotation = rotate_image(blackAndWhiteImage, result[0])
    '''fig, (ax14) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
    ax14.imshow(final_rotation, cmap="gray")
    ax14.set_xlabel("Final rotation", fontsize=14)'''

    corrected_angle = result[0]
    # 10 - Correct angle in original img
    rotation_on_original_image = rotate_image(queryImg, result[0])
    # print("Rotation on original image data type :", rotation_on_original_image.dtype)
    # cropped_img, blackAndWhiteImage, rotation_on_original_image
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + '3_ANGLE_CORRECTED_FRAME' + SLASH + str(idx) + '_frame.png',
                    rotation_on_original_image)
    # image = imageio.imread(PATH_ROOT + SLASH + '4_ANGLE_CORRECTED_FRAME' + SLASH + str(idx) + '_frame.jpg')
    # print("Rotation on compressed image data type :", image.dtype)
    img_obj = imgObj.ImageObj(rotation_on_original_image, 0, 0)
    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_image_angle_corrected_frame.jpg',
                img_obj.get_image())"""
    return img_obj, corrected_angle


def get_largest_line_setup(thresholded_image):
    blackAndWhiteImage = thresholded_image
    result = [0, 0, 0]
    result_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    col_limit = blackAndWhiteImage.shape[1] * 255
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = 0, int(np.ceil(rotated.shape[0] / 3)), rotated.shape[1]
        last_col = cols - 1
        # print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print("Result angle # 1:", result)
    result_list[0] = result

    result = [0, 0, 0]
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = int(np.ceil(rotated.shape[0] / 3) + 1), int(np.ceil((rotated.shape[0] / 3) * 2)), \
                                      rotated.shape[1]
        last_col = cols - 1
        # print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print("Result angle # 2:", result)
    result_list[1] = result

    result = [0, 0, 0]
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = int(np.ceil(((rotated.shape[0] / 3) * 2) + 1)), rotated.shape[0], rotated.shape[1]
        last_col = cols - 1
        # print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print("Result angle # 3:", result)
    result_list[2] = result
    votes = [0, 0, 0]

    voting_options = [result_list[0], result_list[1], result_list[2]]
    if voting_options[0][0] == voting_options[1][0]:
        votes[0] = votes[0] + 1
        votes[1] = votes[1] + 1
    elif voting_options[0][0] - 1 <= voting_options[1][0] <= voting_options[0][0] + 1:
        votes[0] = votes[0] + 0.5
        votes[1] = votes[1] + 0.5
    if voting_options[0][0] == voting_options[2][0]:
        votes[0] = votes[0] + 1
        votes[2] = votes[2] + 1
    elif voting_options[0][0] - 1 <= voting_options[2][0] <= voting_options[0][0] + 1:
        votes[0] = votes[0] + 0.5
        votes[2] = votes[2] + 0.5
    if voting_options[1][0] == voting_options[2][0]:
        votes[1] = votes[1] + 1
        votes[2] = votes[2] + 1
    elif voting_options[1][0] - 1 <= voting_options[2][0] <= voting_options[1][0] + 1:
        votes[1] = votes[1] + 0.5
        votes[2] = votes[2] + 0.5

    print("Votos: ", votes)
    print("Max voting: ", max(votes))
    voting_election = 0, 0
    if votes == [0, 0, 0]:
        final_result = 0
        temp_result_measure = result_list[0][2]
        for result in range(1, 3, 1):
            print("result_list[0][2] = ", result_list[0][2], " < result_list[result][2] = ", result_list[result][2])
            if temp_result_measure < result_list[result][2]:
                final_result = result
                temp_result_measure = result_list[result][2]
        voting_election = 0, final_result
    else:
        for index in range(0, 3, 1):
            print("Chequeando votos", index)
            if votes[index] > voting_election[0]:
                voting_election = votes[index], index
    print("Voting election: ", voting_election)
    return voting_options[voting_election[1]]


def create_plants_mask(BGR_image, idx=None):
    print("Creating plants mask")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    l_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'l')
    a_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'a')
    b_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'b')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'l_channel_RGB2LAB_frame.jpg', l_channel)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_1_a_channel_RGB2LAB_frame.jpg', a_channel)
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'b_channel_RGB2LAB_frame.jpg', b_channel)
    # l_thresh = pcv.threshold.binary(gray_img=l_channel, threshold=150, max_value=255, object_type='light')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'l_channel_tresh_frame.jpg', l_thresh)
    # img_binary = pcv.threshold.binary(gray_img=a_channel, threshold=0, max_value=255, object_type='dark')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '2_a_channel_tresh_frame.jpg', img_binary)
    threshold_value, img_thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_2_a_channel_tresh_frame.jpg', img_thresh)
    img_thresh_bitwised = cv2.bitwise_not(img_thresh)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_3_a_channel_tresh_bitwised_frame.jpg', img_thresh_bitwised)
    result = img_thresh_bitwised
    # dim = np.expand_dims(img_thresh_bitwised, axis=2)
    # new_mask = np.concatenate((dim, dim, dim), axis=2)
    # result = BGR_image * new_mask
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'a_channel_color_space_frame.jpg', result)
    return result


def create_plants_color_mask(BGR_image, idx=None):
    print("Creating plants mask")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    l_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'l')
    a_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'a')
    b_channel = pcv.rgb2gray_lab(BGR_image.copy(), 'b')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'l_channel_RGB2LAB_frame.jpg', l_channel)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_1_a_channel_RGB2LAB_frame.jpg', a_channel)
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'b_channel_RGB2LAB_frame.jpg', b_channel)
    # l_thresh = pcv.threshold.binary(gray_img=l_channel, threshold=150, max_value=255, object_type='light')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'l_channel_tresh_frame.jpg', l_thresh)
    # img_binary = pcv.threshold.binary(gray_img=a_channel, threshold=0, max_value=255, object_type='dark')
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '2_a_channel_tresh_frame.jpg', img_binary)
    threshold_value, img_thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_2_a_channel_tresh_frame.jpg', img_thresh)
    """img_thresh_bitwised = cv2.bitwise_not(img_thresh)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_3_a_channel_tresh_bitwised_frame.jpg', img_thresh_bitwised)
    BW_3_ch = np.stack((img_thresh_bitwised,)*3, axis=-1)"""
    # print(BGR_image)
    # print(type(BW_3_ch))
    # print(BW_3_ch.shape)
    color_mask = BGR_image.copy()
    # color_mask[np.tile(~BW_3_ch, (1,1))] = 0
    color_mask[img_thresh == 255] = 0
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '1_4_a_channel_color_mask_frame.jpg',
                    color_mask)
    # dim = np.expand_dims(img_thresh_bitwised, axis=2)
    # new_mask = np.concatenate((dim, dim, dim), axis=2)
    # result = BGR_image * new_mask
    # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'a_channel_color_space_frame.jpg', result)
    return color_mask


def trim_black_stripes_by_angle(RGB, angle, idx=None):
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    # angle = 1
    # initial_heigth = 1920
    # initial_width = 1080
    initial_heigth = RGB.shape[0]
    initial_width = RGB.shape[1]
    R2 = initial_heigth
    R1 = initial_width
    # print("Sin of 1º: ", math.sin(math.radians(angle)))
    # print("Cos of 1º: ", math.cos(math.radians(angle)))

    H = int(np.ceil((R2 * math.cos(math.radians(abs(angle))) - R1 * math.sin(math.radians(abs(angle)))) / math.cos(
        math.radians(2 * abs(angle)))))
    W = int(np.ceil((R1 * math.cos(math.radians(abs(angle))) - R2 * math.sin(math.radians(abs(angle)))) / math.cos(
        math.radians(2 * abs(angle)))))

    # width_enclosing_square = int(np.ceil(1920/math.sin(math.radians(angle))))
    # heigth_enclosing_square = int(np.ceil(width_enclosing_square*math.cos(math.radians(angle))))

    y1 = int(np.ceil(W * math.sin(math.radians(abs(angle))) * math.cos(math.radians(abs(angle)))))
    y3 = y1
    x1 = int(np.ceil(R2 * math.sin(math.radians(abs(angle))) - W * (math.sin(math.radians(abs(angle)))) ** 2))
    x2 = x1
    y2 = y1 + H
    y4 = y2
    x4 = x2 + W
    x3 = x4

    # print("Width of enclosing square: ", W)
    # print("Heigth of enclosing square: ", H)
    # print("Coordinates point 1: ", x1, " and ", y1)
    # print("Coordinates point 2: ", x2, " and ", y2)
    # print("Coordinates point 3: ", x3, " and ", y3)
    # print("Coordinates point 4: ", x4, " and ", y4)

    if (y1 >= y3):
        init_row = y1
    else:
        init_row = y3

    if (y2 <= y4):
        final_row = y2
    else:
        final_row = y4

    if (x1 >= x2):
        init_col = x1
    else:
        init_col = x2

    if (x3 <= x4):
        final_col = x3
    else:
        final_col = x4
    # print("Init row: ", init_row)
    # print("Final row: ", final_row)
    # print("Init col: ", init_col)
    # print("Final col: ", final_col)
    RGB = RGB[init_row:final_row, init_col:final_col]
    # RGB = RGB[init_col:final_col, init_row:final_row]
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_' + '0_angle_corrected_frame.jpg', RGB)
    return RGB


def create_mask_filled_by_plants(BGR_image):
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"

    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_1_mask_by_plants_init_frame.jpg',
                BGR_image)"""
    # RG_image = BGR_image.copy()
    RGB_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)  # this is performed for get mixed color values
    # RG_image[:,:,0] = np.zeros([RG_image.shape[0], RG_image.shape[1]])
    """RGB_image_inv = cv2.bitwise_not(RGB_image.copy())"""
    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_2_mask_by_plants_not_blue_channel_frame.jpg',
                RGB_image)"""
    # Convert RGB image to chosen color space
    I = color.rgb2lab(RGB_image.copy(), illuminant="D50")
    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_3_mask_by_plants_RGB2LAB_frame.jpg',
                I)"""

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.000  # or '0.000'
    channel1Max = 100.000  # or '100.000'

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = -46.769  # or '-46.769'
    channel2Max = -3.750  # or '-3.750'

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.993  # or '0.993'
    channel3Max = 91.735  # or '91.735'

    # Create mask based on chosen histogram thresholds
    sliderBW = (I[:, :, 0] >= channel1Min) & (I[:, :, 0] <= channel1Max) & \
               (I[:, :, 1] >= channel2Min) & (I[:, :, 1] <= channel2Max) & \
               (I[:, :, 2] >= channel3Min) & (I[:, :, 2] <= channel3Max)
    BW = sliderBW
    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_4_mask_by_plants_sliderBW_frame.jpg',
                np.float32(BW.copy()))"""

    # ------------
    # Copy the thresholded image.
    BW_floodfill = np.float32(BW.copy())

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = BW.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(BW_floodfill, mask, (0, 0), 255)

    im_out = np.float32(BW.copy()) - BW_floodfill
    np.invert(np.uint8(im_out))
    BW[:, :] = im_out[:, :]
    maskedRGBImage = RGB_image.copy()
    maskedRGBImage[np.tile(BW, (1, 1))] = 0

    return maskedRGBImage


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def find_major_rectangle_area(set_of_rects_coordinates):
    rects_area = np.empty(len(set_of_rects_coordinates))
    index = 0
    for rect_tuple in set_of_rects_coordinates:
        print("Rect tuple: ", rect_tuple)
        rects_area[index] = int(rect_tuple[3] * rect_tuple[2])
        index += 1
    max_index = np.argmax(rects_area, axis=0)
    print(max_index)

    return max_index


def get_seedbed_contour_rect_coordinates(RGB_image, idx=None):
    print("Getting seedbed contour rect coordinates")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    #   Using object centroid coordinates to compute distance between

    #   1 - Get individual plant bounding circle
    im_gray = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
    # fig, ax1 = plt.subplots(figsize=(16, 9))
    # ax1.imshow(RGB_image)
    # ax1.set_xlabel("Plant mask binarizes", fontsize=14)
    source_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(hierarchy))
    # print(len(contours[500]))

    # Approximate contours to polygons + get bounding circles
    contours_poly = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    circles_contour = [centers, radius]
    print(radius)
    print(len(radius))

    drawing = np.zeros((im_gray.shape[0], im_gray.shape[1], 3), dtype=np.uint8)
    # Draw polygonal contour + bounding circles
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        im_with_contours = cv2.drawContours(drawing, contours_poly, i, color)
        # im_with_contours = cv2.drawContours(RGB_image.copy(), contours, -1, (0,0,255), 2)
        im_with_circle_bounding = cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color,
                                             2)
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_1_circle_bounding_frame.jpg', im_with_circle_bounding)
    #   2 - Get maximum radius value of individuals plants bounding circles
    outliers = []
    threshold = 3
    mean = np.mean(radius)
    std = np.std(radius)
    print("Mean: ", mean)
    print("Standar deviation: ", std)
    for y in radius:
        z_score = (y - mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(y)
    if outliers:
        for each_value in outliers:
            index = radius.index(each_value)
            del radius[index]
    print("Radius maximum value: ", math.ceil(max(radius)))
    max_radius = math.ceil(max(radius))
    kernel_width = math.ceil(max_radius * 2)
    kernel_height = math.ceil(max_radius)
    #   3 - Getting circular structuring element
    circular_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))

    #   4 - Getting seedbeds using circular structuring element and an opening
    for i in range(0, 1, 1):
        # img_erode = cv2.erode(thresh, circular_structure) #---- resalta lo brillante en forma circular
        # img = cv2.dilate(img_erode, circular_structure) #---- reduce los artefactos oscuros
        im_dilated = cv2.dilate(thresh, circular_structure)

    rows, cols = RGB_image.shape[0], RGB_image.shape[1]
    min_limit_area = (rows * cols) * 0.125
    min_limit_area_big_object = (rows * cols) * 0.125

    SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(np.ceil(rows * 0.05))))  # vertical line
    SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols * 2, 1))  # horizontal line
    print("S3 Diamond factor: ", int(rows * 0.01))
    # removes small objects by connected components analysis
    # nlabel, labels = cv2.connectedComponents(closing_img_thres, connectivity=8)
    structure = [[1, 1, 1],  # structure for a connectivity analysis of 8 components
                 [1, 1, 1],
                 [1, 1, 1]]
    label_im, nb_labels = ndimage.label(im_dilated, structure)

    sizes_labels = ndimage.sum(im_dilated, label_im, range(nb_labels + 1))
    for each in sizes_labels:
        print(str(each / 255) + " vs to limit area: " + str(min_limit_area))
    # mask_size = sizes_labels/255 < min_limit_area
    labels_to_remove = check_small_object_to_remove(sizes_labels, min_limit_area, min_limit_area_big_object)
    # big_objects = sizes_labels/255 < min_limit_area_big_object

    print("Size_labels type: ", type(sizes_labels))
    print("Mask size: ", labels_to_remove)
    print("Mask size type: ", type(labels_to_remove))
    remove_pixel = labels_to_remove[label_im]
    im_dilated[remove_pixel] = 0
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_2_small_objects_removed_frame.jpg',
                im_dilated)
    # -----
    closing_img_thres = cv2.morphologyEx(im_dilated, cv2.MORPH_CLOSE, SE1)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_3_closing_frame.jpg',
                closing_img_thres)
    """filled_image = ndimage.binary_fill_holes(closing_img_thres.copy(), structure=np.ones((20,20)))
    print(filled_image.astype(int))
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_filled_frame.jpg',
                filled_image.astype(int))
    im_floodfill = closing_img_thres.copy()
    # Mask used to flood filling
    h, w = closing_img_thres.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Flood fill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 0)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_floodfill_frame.jpg',
                im_floodfill)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '6_floodfill_inv_frame.jpg',
                im_floodfill_inv)

    # Combine the to image to get the foreground
    im_out = closing_img_thres | im_floodfill_inv
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '7_foreground_frame.jpg',
                im_out)"""""
    """# im_out = im_out * 255
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_fill_holes_frame.jpg',
                im_out)"""

    # -----
    # Select the greater object by connected components analysis
    label_im, nb_labels = ndimage.label(closing_img_thres, structure)

    sizes_labels = ndimage.sum(closing_img_thres, label_im, range(nb_labels + 1))
    print(sizes_labels.dtype)
    max_value = np.max(sizes_labels)
    print("Max value: ", max_value)
    for each in sizes_labels:
        print(str(each) + " vs to limit area: " + str(min_limit_area))
    # idx_value = np.where(sizes_labels == max_value)
    # seedbed_mask = im_out * 0
    # seedbed_mask[label_im == idx_value] = 255
    mask_size = sizes_labels < max_value
    remove_pixel = mask_size[label_im]
    closing_img_thres[remove_pixel] = 0
    seedbed_mask = closing_img_thres
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_4_greater_object_frame.jpg',
                seedbed_mask)
    seedbed_mask = cv2.dilate(seedbed_mask, SE2)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_5_greater_object_dilated_frame.jpg',
                seedbed_mask)

    # ----
    if OS.lower() == 'windows':
        source_image, seedbed_contours, seedbed_hierarchy = cv2.findContours(seedbed_mask, cv2.RETR_TREE,
                                                                             cv2.CHAIN_APPROX_SIMPLE)
    elif OS.lower() == 'linux':
        source_image, seedbed_contours, seedbed_hierarchy = cv2.findContours(seedbed_mask, cv2.RETR_TREE,
                                                                             cv2.CHAIN_APPROX_SIMPLE)

    # Getting main seedbed polygon
    # Approximate contours to polygons + get bounding rects
    contours_poly = [None] * len(seedbed_contours)
    bound_rect = [None] * len(seedbed_contours)
    for i, c in enumerate(seedbed_contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    max_area_rect_index = find_major_rectangle_area(bound_rect)
    max_area_rect_atributes = bound_rect[max_area_rect_index]
    top_left = (max_area_rect_atributes[0], max_area_rect_atributes[1])
    bottom_left = (max_area_rect_atributes[0], max_area_rect_atributes[1] + max_area_rect_atributes[3])
    top_right = (max_area_rect_atributes[0] + max_area_rect_atributes[2], max_area_rect_atributes[1])
    bottom_right = (
        max_area_rect_atributes[0] + max_area_rect_atributes[2],
        max_area_rect_atributes[1] + max_area_rect_atributes[3])
    coordinates = (top_left, top_right, bottom_left, bottom_right)

    # ----
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH + str(idx) + '_frame.jpg', seedbed_mask)

    return coordinates


def get_seedbed_mask(img_obj):  # 10_Frames_zona_de_plantulas alternative solution
    print("Getting_seed_bed_mask")
    image = img_obj.get_image()
    # 1 - Gets plants mask
    mask = create_mask_filled_by_plants(image)
    # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 2 - Gets seedbed contour rect coordinates
    seedbed_coordinates, circulars_contour, im_tresh, seedbed_thresh = get_seedbed_contour_rect_coordinates(mask)
    # print(seedbed_coordinates)
    # This function get an image with just the seedbed
    drawing_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawing_mask[seedbed_coordinates[0][1]:seedbed_coordinates[2][1], :, :] = image[seedbed_coordinates[0][1]:
                                                                                    seedbed_coordinates[2][1], :, :]
    seedbed_mask = imgObj.ImageObj(drawing_mask, 0, 0)

    # print(seedbed_coordinates[0][1], seedbed_coordinates[2][1])
    return seedbed_mask


def check_small_object_to_remove(sizes_labels, min_limit_area, min_limit_area_for_big_object):
    small_labels = sizes_labels/255 < min_limit_area
    print("Small labels: ", small_labels)
    print("Small labels type: ", type(small_labels))
    big_labels = sizes_labels/255 > min_limit_area_for_big_object
    print("Big labels: ", big_labels)
    labels_to_remove = np.zeros(len(sizes_labels)).astype(bool)
    print("Lables to remove: ", labels_to_remove)
    print("Labels to remove type: ", type(labels_to_remove))
    counter = 0
    before_label = False
    after_label = False
    for each_label_size in small_labels:
        if each_label_size and not counter == 0:
            for idx in range(counter, -1, -1):
                if big_labels[idx]:
                    before_label = True
                    break
            for idx in range(counter, len(sizes_labels), 1):
                if big_labels[idx]:
                    after_label = True
                    break
            if not (before_label and after_label):
                labels_to_remove[counter] = True
        else:
            labels_to_remove[counter] = each_label_size
        counter += 1
        before_label = False
        after_label = False
    return labels_to_remove


def get_seedbed_coordinates(binary_img, idx=None):
    print("Getting_seedbed_mask")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_0_binary_frame.jpg',
                binary_img)
    rows, cols = binary_img.shape[0], binary_img.shape[1]
    min_limit_area = (rows * cols) * 0.125
    min_limit_area_big_object = (rows * cols) * 0.125

    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (int(np.ceil(cols * 0.95)), 1))  # horizontal line
    SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(np.ceil(rows * 0.05))))  # vertical line
    SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols * 2, 1))  # horizontal line
    print("S3 Diamond factor: ", int(rows*0.01))
    SE3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (int(rows*0.0075), int(rows*0.0075)))  # diamond

    """cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_image_frame.jpg',
                image)
    img_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '1_gray_frame.jpg',
                img_gray)
    threshold_value, img_thres = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)"""
    img_thresh = binary_img.copy()
    dilated_img_thres = cv2.dilate(img_thresh, SE)
    closing_img_thres = cv2.morphologyEx(dilated_img_thres, cv2.MORPH_CLOSE, SE3)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_1_close_frame.jpg',
                closing_img_thres)
    # removes small objects by connected components analysis
    # nlabel, labels = cv2.connectedComponents(closing_img_thres, connectivity=8)
    structure = [[1, 1, 1],  # structure for a connectivity analysis of 8 components
                 [1, 1, 1],
                 [1, 1, 1]]
    label_im, nb_labels = ndimage.label(closing_img_thres, structure)

    sizes_labels = ndimage.sum(closing_img_thres, label_im, range(nb_labels + 1))
    for each in sizes_labels:
        print(str(each/255) + " vs to limit area: " + str(min_limit_area))
    # mask_size = sizes_labels/255 < min_limit_area
    labels_to_remove = check_small_object_to_remove(sizes_labels, min_limit_area, min_limit_area_big_object)
    # big_objects = sizes_labels/255 < min_limit_area_big_object

    print("Size_labels type: ", type(sizes_labels))
    print("Mask size: ", labels_to_remove)
    print("Mask size type: ", type(labels_to_remove))
    remove_pixel = labels_to_remove[label_im]
    closing_img_thres[remove_pixel] = 0
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_2_small_objects_removed_frame.jpg',
                closing_img_thres)
    # -----
    closing_img_thres = cv2.morphologyEx(closing_img_thres, cv2.MORPH_CLOSE, SE1)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_3_closing_frame.jpg',
                closing_img_thres)
    """filled_image = ndimage.binary_fill_holes(closing_img_thres.copy(), structure=np.ones((20,20)))
    print(filled_image.astype(int))
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_filled_frame.jpg',
                filled_image.astype(int))
    im_floodfill = closing_img_thres.copy()
    # Mask used to flood filling
    h, w = closing_img_thres.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Flood fill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 0)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_floodfill_frame.jpg',
                im_floodfill)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '6_floodfill_inv_frame.jpg',
                im_floodfill_inv)

    # Combine the to image to get the foreground
    im_out = closing_img_thres | im_floodfill_inv
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '7_foreground_frame.jpg',
                im_out)"""""
    """# im_out = im_out * 255
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '5_fill_holes_frame.jpg',
                im_out)"""

    # -----
    # Select the greater object by connected components analysis
    label_im, nb_labels = ndimage.label(closing_img_thres, structure)

    sizes_labels = ndimage.sum(closing_img_thres, label_im, range(nb_labels + 1))
    print(sizes_labels.dtype)
    max_value = np.max(sizes_labels)
    print("Max value: ", max_value)
    for each in sizes_labels:
        print(str(each) + " vs to limit area: " + str(min_limit_area))
    # idx_value = np.where(sizes_labels == max_value)
    # seedbed_mask = im_out * 0
    # seedbed_mask[label_im == idx_value] = 255
    mask_size = sizes_labels < max_value
    remove_pixel = mask_size[label_im]
    closing_img_thres[remove_pixel] = 0
    seedbed_mask = closing_img_thres
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_4_greater_object_frame.jpg',
                seedbed_mask)
    seedbed_mask = cv2.dilate(seedbed_mask, SE2)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_2_5_greater_object_dilated_frame.jpg',
                seedbed_mask)

    # ----
    if OS.lower() == 'windows':
        source_image, seedbed_contours, seedbed_hierarchy = cv2.findContours(seedbed_mask, cv2.RETR_TREE,
                                                                             cv2.CHAIN_APPROX_SIMPLE)
    elif OS.lower() == 'linux':
        source_image, seedbed_contours, seedbed_hierarchy = cv2.findContours(seedbed_mask, cv2.RETR_TREE,
                                                                             cv2.CHAIN_APPROX_SIMPLE)

    # Getting main seedbed polygon
    # Approximate contours to polygons + get bounding rects
    contours_poly = [None] * len(seedbed_contours)
    bound_rect = [None] * len(seedbed_contours)
    for i, c in enumerate(seedbed_contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    max_area_rect_index = find_major_rectangle_area(bound_rect)
    max_area_rect_atributes = bound_rect[max_area_rect_index]
    top_left = (max_area_rect_atributes[0], max_area_rect_atributes[1])
    bottom_left = (max_area_rect_atributes[0], max_area_rect_atributes[1] + max_area_rect_atributes[3])
    top_right = (max_area_rect_atributes[0] + max_area_rect_atributes[2], max_area_rect_atributes[1])
    bottom_right = (
        max_area_rect_atributes[0] + max_area_rect_atributes[2],
        max_area_rect_atributes[1] + max_area_rect_atributes[3])
    coordinates = (top_left, top_right, bottom_left, bottom_right)

    # ----
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + '4_SEEDBED_MASK' + SLASH + str(idx) + '_frame.jpg', seedbed_mask)

    return coordinates


def get_seedbed(image, seedbed_coordinates, idx=None):
    print("Getting_seedbed")
    # This function get an image with just the seedbed
    drawing_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawing_mask[seedbed_coordinates[0][1]:seedbed_coordinates[2][1], :, :] = image[seedbed_coordinates[0][1]:
                                                                                    seedbed_coordinates[2][1], :, :]
    seedbed_mask = drawing_mask

    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '5_SEEDBED' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '5_SEEDBED' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    if not idx is None:
        cv2.imwrite(PATH_ROOT + SLASH + '5_SEEDBED' + SLASH + str(idx) + '_frame.jpg', seedbed_mask)
    # print(seedbed_coordinates[0][1], seedbed_coordinates[2][1])
    return seedbed_mask


def delete_repeated_frames(frames_list):
    print("Deleting repeated frames at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '2_DELETED_REPEATED_FRAME' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '2_DELETED_REPEATED_FRAME' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    removed = 0
    final_frames_list = []
    for idx in range(0, len(frames_list) - 2):
        img1 = frames_list[idx]
        img2 = frames_list[idx + 1]
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1_gray_double = np.array(img1_gray).astype(np.float)
        img2_gray_double = np.array(img2_gray).astype(np.float)
        R = corr2(img1_gray_double, img2_gray_double)
        if R >= 0.875:
            cv2.imwrite(PATH_ROOT + SLASH + '2_DELETED_REPEATED_FRAME' + SLASH + str(idx) + '_frame.jpg',
                        frames_list[idx])
            removed = removed + 1
        elif R < 0.875:
            final_frames_list.append(frames_list[idx])
    print("Removed " + str(removed) + " frames.")
    print("Finished successfully at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return final_frames_list


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum());
    return r


def split_video_frames(VIDEO_PATH):
    print("Splitting video frames")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    frames_list = []
    # Read the video from specified path
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Number of frame in the video file: ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Frame rate: ", cap.get(cv2.CAP_PROP_FPS))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '1_VIDEO_SPLIT' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '1_VIDEO_SPLIT' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

        # fps = cap.get(5)
    # time_length = cap.get(7)/fps
    # frame_seq = 749
    # frame_no = (frame_seq /(time_length*fps))
    # print("Time length: ", time_length)
    # frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(np.floor(fps))
    if fps == 30:
        frames_frequency = 5  # before each 5th frames, now 10% of fps
    elif fps == 60:
        frames_frequency = 10  # before each 10th frames, now 10% of fps

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frames_frequency)
    frame_indexes = frames_frequency * np.arange(frames)
    current_frame_name = 1000
    currentframe = 0

    for index in frame_indexes:

        # reading from frame
        ret, frame = cap.read()
        # if index > 210 * frames_frequency:
        #    break
        if ret:
            # if video is still left continue creating images
            name = PATH_ROOT + SLASH + '1_VIDEO_SPLIT' + SLASH + str(current_frame_name) + '_frame.jpg'
            print('Creating...' + name)

            # Rotating 180º the frame

            '''rows,cols = frame.shape[:2]
            center = (cols/2, rows/2)
            angle90 = 90
            angle180 = 180
            angle270 = 270
            scale = 1

            M = cv2.getRotationMatrix2D(center,angle270,scale)
            frame = cv2.warpAffine(frame,M,(cols,rows))'''

            frame = rotate_image(frame, 270)

            # writing the extracted images
            cv2.imwrite(name, frame)
            frames_list.append(frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 10
            current_frame_name += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # before each 15 frames currentframe * 15
        else:
            break

    # Release all space and windows once done
    cap.release()
    cv2.destroyAllWindows()
    frames_list = delete_repeated_frames(frames_list)
    return frames_list


def is_trash_frame(img_obj_plants_mask):
    print("Deleting trash frames")
    is_a_trash = False
    return


def establish_reference_size_for_scaling(image_obj_list):
    print("Stablishing scaling factor")
    first_time = True
    reference_height = 0
    for each_tuple in image_obj_list:
        height = (each_tuple[1])[2][1] - (each_tuple[1])[0][1]
        # img = each_triplet[1].get_image()
        # height, width, channels = img.shape
        if first_time or height < reference_height:
            reference_height = height
            first_time = False
    return reference_height


def scale(image_obj_list, reference_height_factor):
    print("Scaling image objects list by factor")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '6_SCALED_FRAME' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '6_SCALED_FRAME' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    scaled_img_obj_list = []
    idx = 1000
    for each_tuple in image_obj_list:
        # img = each_triplet[1].get_image()
        # height, width, channels = img.shape
        height = (each_tuple[1])[2][1] - (each_tuple[1])[0][1]
        rescale_factor = reference_height_factor / height
        # mask_dim = int((each_triplet[1].shape[1]) * rescale_factor), int((each_triplet[1].shape[0]) * rescale_factor)
        # scaled_mask = cv2.resize(each_triplet[1], mask_dim, interpolation=cv2.INTER_AREA)
        # scaled_mask_obj = imgObj.ImageObj(scaled_mask)
        img_dim = int(((each_tuple[0].get_image()).shape[1]) * rescale_factor), int(
            ((each_tuple[0].get_image()).shape[0]) * rescale_factor)
        scaled_img = cv2.resize(each_tuple[0].get_image(), img_dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(PATH_ROOT + SLASH + '6_SCALED_FRAME' + SLASH + str(idx) + '_frame.jpg', scaled_img)
        scaled_img_obj = imgObj.ImageObj(scaled_img, 0, 0)
        scaled_img_obj_list.append(scaled_img_obj)
        idx += 1
    return scaled_img_obj_list


def set_standard_size_frame(img_obj_triplet, standard_size):
    print("Setting standard size frame")
    standard_frame_1 = np.zeros((standard_size[0], standard_size[1], 3), dtype=np.uint8)
    standard_frame_2 = standard_frame_1.copy()
    # set each image on the corner of the standard frame and each one will be centered with center function
    # img_obj '0' --> original image
    img_1 = img_obj_triplet[0].get_image()
    standard_frame_1[0:img_1.shape[0] - 1, 0:img_1.shape[1] - 1] = img_1[:, :]

    # img_obj '1' --> mask image
    img_2 = img_obj_triplet[1].get_image()
    standard_frame_2[0:img_2.shape[0] - 1, 0:img_2.shape[1] - 1] = img_2[:, :]
    img_obj_triplet_output = imgObj.ImageObj(standard_frame_1, 0, 0), imgObj.ImageObj(standard_frame_2, 0, 0), \
                             img_obj_triplet[2]
    return img_obj_triplet_output


def center_seedbed(image_obj, standard_size, idx=None):
    print("Centering image object")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '7_CENTERED_STANDARD_FRAME' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '7_CENTERED_STANDARD_FRAME' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    complete_image = image_obj.get_image()
    plants_mask = create_plants_mask(complete_image)
    print("Plants mask: ", plants_mask.shape)
    width = plants_mask.shape[1]
    seedbed_coordinates = get_seedbed_coordinates(plants_mask)
    standard_frame = np.zeros((standard_size[0], standard_size[1], 3), dtype=np.uint8)
    print("Standard frame: ", standard_frame.shape)
    standard_frame_height_middle_point = int(np.ceil(standard_size[0] / 2))
    print("Standard frame height middle point: ", standard_frame_height_middle_point)
    seedbed_height_middle_point = seedbed_coordinates[0][1] + int(
        np.ceil((seedbed_coordinates[2][1] - seedbed_coordinates[0][1]) / 2))
    print("Seedbed height middle point: ", seedbed_height_middle_point)
    initial_row = standard_frame_height_middle_point - seedbed_height_middle_point
    print("Initial row: ", initial_row)
    # bottom_row = seedbed_coordinates[2][1] - seedbed_coordinates[0][1]
    bottom_row = plants_mask.shape[0]
    print("Bottom row: ", bottom_row)
    standard_frame[initial_row:initial_row + bottom_row, :width, :] = complete_image[:, :, :]
    cv2.imwrite(PATH_ROOT + SLASH + SLASH + '7_CENTERED_STANDARD_FRAME' + SLASH + str(idx) + '_frame.jpg',
                standard_frame)
    centred_frame_standard_img_obj = imgObj.ImageObj(standard_frame, 0, 0)
    return centred_frame_standard_img_obj


def trim_by_right(images_list):
    print("Trimming image object by right side")
    output_images_list = []
    acum_init_col = []
    acum_end_col = []
    acum_init_row = []
    acum_end_row = []

    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:
        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '8_TRIMMED_FRAME' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '8_TRIMMED_FRAME' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    for each_image_obj in images_list:
        each_image = each_image_obj.get_image()
        img_gray = cv2.cvtColor(each_image.copy(), cv2.COLOR_BGR2GRAY)
        vertical_proyection = img_gray.sum(axis=0)
        # index = vertical_proyection.ravel().nonzero()
        index = (vertical_proyection > 0).nonzero()
        print("Index length for vertical proyection: ", len(index[0]))
        # print(index[0])
        init_col = index[0][0]
        end_col = len(index[0])
        acum_init_col.append(init_col)
        acum_end_col.append(end_col)

        horizontal_proyection = img_gray.sum(axis=1)
        # index = horizontal_proyection.ravel().nonzero()
        index = (horizontal_proyection > 0).nonzero()
        print("Index length for horizontal proyection: ", len(index[0]))
        # print(index[0])
        init_row = index[0][0]
        end_row = len(index[0])
        acum_init_row.append(init_row)
        acum_end_row.append(end_row)

    # print("acum_init_row ", acum_init_row)
    # print("acum_end_row ", acum_end_row)
    init_col = min(acum_init_col)
    end_col = min(acum_end_col)

    init_row = min(acum_init_row)
    end_row = max(acum_end_row)
    idx_frame = 1000
    for each_image_obj in images_list:
        each_image = each_image_obj.get_image()
        output_image = each_image[init_row:end_row, init_col:end_col, :]
        output_images_list.append(output_image)
        cv2.imwrite(PATH_ROOT + SLASH + SLASH + '8_TRIMMED_FRAME' + SLASH + str(idx_frame) + '_frame.jpg',
                    output_image)
        idx_frame += 1
    return output_images_list


def trim_by_right_op(image_obj):
    print("Trimming image object")
    RGB = image_obj.get_image()
    img_gray = cv2.cvtColor(RGB.copy(), cv2.COLOR_BGR2GRAY)
    horizontal_stripe_projection = np.zeros(img_gray.shape[0])
    # print("Shape gray img: ", img_gray.shape)
    for each_row in range(0, img_gray.shape[0], 1):
        pixel = img_gray[each_row, img_gray.shape[1] - 1]
        # print("New column pixel: ", pixel)
        counter = 0
        # print(img_gray[img_gray.shape[0] - 2:img_gray.shape[0], 0:250])
        # print(img_gray[img_gray.shape[0] - 1:img_gray.shape[0], 0:250])
        while (pixel <= 1):
            counter += 1
            # print("Counter: ", counter)
            pixel = img_gray[each_row][(img_gray.shape[1] - 1) - counter]
            # print(pixel)
        horizontal_stripe_projection[each_row] = counter
    # print(vertical_stripe_projection)
    print("Max horizontal projection is below 15%: ",
          max(horizontal_stripe_projection) < int(np.ceil(img_gray.shape[1] * 0.10)))

    outliers = []

    threshold = 3
    mean = np.mean(horizontal_stripe_projection)
    std = np.std(horizontal_stripe_projection)
    # print("Media: ", mean)
    # print("STD: ", std)

    for y in horizontal_stripe_projection:
        z_score = (y - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(y)
    if not outliers:
        pixel_trim = max(horizontal_stripe_projection)
        print("Pixels to trim by right side: ", int(pixel_trim))
    else:
        # print(outliers)
        definitive_projection = horizontal_stripe_projection
        for each_outlier in outliers:
            definitive_projection = np.delete(definitive_projection, np.argwhere(definitive_projection == each_outlier))
        pixel_trim = max(definitive_projection)
        print("Pixels to trim by right side: ", int(pixel_trim))
    RGB = RGB[:RGB.shape[0], :RGB.shape[1] - int(pixel_trim)]
    image_obj = imgObj.ImageObj(RGB, 0, 0)
    return image_obj


def trim_by_center(trimmed_by_right_images_list):
    print("Trimming images list by center, 1/9 of image width")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    OS = platform.system()
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    try:
        # creating a folder named data
        if not os.path.exists(PATH_ROOT + SLASH + '9_TRIMMED_FRAME_TO_STITCH' + SLASH):
            os.makedirs(PATH_ROOT + SLASH + '9_TRIMMED_FRAME_TO_STITCH' + SLASH)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    output_images_list = []
    previous_width = trimmed_by_right_images_list[0].shape[1]
    width_center = int(np.ceil(previous_width / 2))
    new_width = int(np.ceil(previous_width * (1 / 9)))
    init_col = width_center - int(np.ceil(new_width / 2))
    end_col = width_center + int(np.ceil(new_width / 2))
    idx_frame = 1000
    for image in trimmed_by_right_images_list:
        output_image = image[:, init_col:end_col, :]
        output_images_list.append(output_image)
        cv2.imwrite(PATH_ROOT + SLASH + SLASH + '9_TRIMMED_FRAME_TO_STITCH' + SLASH + str(idx_frame) + '.jpg',
                    output_image)
        idx_frame += 1
    return output_images_list


def identify_util_frames_range(frames_list):
    print('Identifying')
    OS = platform.system()
    print(OS)
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    first_frame = 0
    last_frame = 0
    idx = 1000
    height, width = frames_list[0].shape[0], frames_list[0].shape[1]
    # print("Height: ", height)
    h_structuring_element = int(np.floor(height * 0.06))
    w_structuring_element = int(np.floor(width * 0.12))
    radio_for_circle = int((h_structuring_element + w_structuring_element)/2)
    cirular_structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio_for_circle, radio_for_circle))
    for each_frame in frames_list:
        ch_b_frame = pcv.rgb2gray_lab(each_frame, 'b')
        threshold_value, img_thresh = cv2.threshold(ch_b_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '2_a_channel_tresh_frame.jpg', img_thresh)
        img_thresh_bitwised = cv2.bitwise_not(img_thresh)
        eroded_frame = cv2.erode(img_thresh_bitwised, cirular_structuring_element)
        # Analysis of connected componentes
        label_im, nb_labels = ndimage.label(eroded_frame)

        sizes_labels = ndimage.sum(eroded_frame, label_im, range(nb_labels + 1))

        mask_size = sizes_labels < (width * height) * 0.25
        remove_pixel = mask_size[label_im]

        eroded_frame[remove_pixel] = 0
        dilated_frame = cv2.dilate(eroded_frame, cirular_structuring_element)
        label_im, nb_labels = ndimage.label(dilated_frame)
        print("Number of objects on frame " + str(idx) + " ==> " + str(nb_labels) + " objects")
        cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + str(idx) + '_frame.jpg',
                        dilated_frame)
        idx += 1
    return first_frame, last_frame


def homogenize_image_set(path):
    print("Homogenizing image set...")
    #  Splits video
    print("Started split_video_frames() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    frames_list = split_video_frames(path)  # pending generates frame_list into function
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # -------
    """OS = platform.system()
    print(OS)
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    frames_list = []
    idx = 1127
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    image = imageio.imread(PATH_ROOT + SLASH + '1_VIDEO_SPLIT' + SLASH + str(idx) + '_frame.jpg')
    print("Image dtype: ", image.dtype)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_0_split_video_frame.jpg',
                image)
    frames_list.append(image)"""
    # -------
    images_list = []
    standard_size = int(np.ceil(frames_list[0].shape[0] * 1.35)), int(np.ceil(frames_list[0].shape[1] * 1.275))
    init_frame_found = False
    idx_frame = 1000
    # first_frame, last_frame = identify_util_frames_range(frames_list)
    first_frame = 1017  # 1040
    last_frame = 1567  # 2410
    print("Frames quantity: ", len(frames_list))
    for each_frame in frames_list:
        print("IDX_FRAME: ", idx_frame)
        if first_frame <= idx_frame <= last_frame:  # <= 1378, 1137 <= idx_frame <= 1556 P_Smart_1, 1016 <= idx_frame <= 1392 P_Smart_2, 1052 <= idx_frame <= 1367 P_10_Lite
            # if idx_frame > 1131:
            #    print("IDX_frame break: ", idx_frame)
            #    break
            print("Started correct_angle() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            angle_corrected_img_obj, angle_corrected = correct_angle(imgObj.ImageObj(each_frame, 0, 0), idx_frame)
            print("Finished successfully at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Started trim_black_stripes_by_angle() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            trimmed_angle_corrected = trim_black_stripes_by_angle(angle_corrected_img_obj.get_image(), angle_corrected, idx_frame)
            print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Started trim_black_stripes_by_angle() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            plant_mask = create_plants_color_mask(trimmed_angle_corrected, idx_frame)
            print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Started get_seedbed_contour_rect_coordinates() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # seedbed_coordinates = get_seedbed_coordinates(plant_mask, idx_frame)
            seedbed_coordinates = get_seedbed_contour_rect_coordinates(plant_mask, idx_frame)
            print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Started get_seedbed_mask() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            only_seedbed = get_seedbed(angle_corrected_img_obj.get_image().copy(), seedbed_coordinates, idx_frame)
            print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            only_seedbed_img_obj = imgObj.ImageObj(only_seedbed, 0, 0)
            if init_frame_found or not is_trash_frame(only_seedbed_img_obj):
                init_frame_found = True
                images_list.append((angle_corrected_img_obj, seedbed_coordinates))
            # idx_frame += 1
        idx_frame += 1
    print("Started establish_reference_size_for_scaling() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scaling_factor = establish_reference_size_for_scaling(images_list)
    print("Scaling factor: ", scaling_factor)
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Started escale() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scaled_images_list = scale(images_list, scaling_factor)
    idx_frame = 1000
    images_list = []
    for each_img_obj in scaled_images_list:
        # standarized_frame_size_triplet = set_standard_size_frame(each_triplet, standard_size)
        print("Started center_seedbed() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        centered_img_obj = center_seedbed(each_img_obj, standard_size, idx_frame)
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        images_list.append(centered_img_obj)
        idx_frame += 1

    print("Started trim_by_right() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trimmed_by_right_images_list = trim_by_right(images_list)
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Started trim_by_right() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    final_images_list = trim_by_center(trimmed_by_right_images_list)
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return final_images_list


def routine_test(path):
    print("Started split_video_frames() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    frames_list = split_video_frames(path)  # pending generates frame_list into function
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(len(frames_list), " frames after removed")
    first_frame, last_frame = identify_util_frames_range(frames_list)


if __name__ == '__main__':
    """path = '/home/mrwolf/Projects/Gepar - UdeA/Flowers/Capiros - UdeA Project/REPO/SICOP/images/'
    source_path = "../../images/"
    files_path = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    files_path.sort()
    output_name_counter = 1
    for file_pointer in range(0, len(files_path), 1):
        print(files_path[file_pointer])
        image = imageio.imread(source_path + files_path[file_pointer])
        standard_size = image.shape[0] * (1.25), image.shape[1] * (1.25)
        standard_img_obj = set_standard_size_frame(imgObj.ImageObj(image), standard_size)
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
        ax1.imshow(standard_img_obj.get_image())
        ax1.set_xlabel("Image with standard size color", fontsize=14)
        plt.show()
    """
    init_time = datetime.now()
    OS = platform.system()
    print(OS)
    if OS.lower() == 'windows':
        SLASH = "\\"
    elif OS.lower() == 'linux':
        SLASH = "/"
    VIDEO_PATH = os.path.dirname(os.path.abspath(__file__)) + SLASH + 'Capture_21_08_2020' + SLASH + 'VID_20200821_6.mp4'
    # VIDEO_PATH = os.path.dirname(os.path.abspath(__file__)) + SLASH + 'P_Smart_VID_20200320_1.mp4'
    # VIDEO_PATH = os.path.dirname(os.path.abspath(__file__)) + SLASH + 'P_Smart_VID_20200320_2.mp4'
    # VIDEO_PATH = os.path.dirname(os.path.abspath(__file__)) + SLASH + 'Huawei_P_10_Lite_VID_20200117.mp4'
    print("Started")
    # routine_test(VIDEO_PATH)
    homogenize_image_set(VIDEO_PATH)
    print("Finished with success =D")
    print("Started at ", init_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Finished at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    """idx_frame = 1020
    IMAGE_PATH = os.path.dirname(
        os.path.abspath(__file__)) + SLASH + "3_ANGLE_CORRECTED_FRAME" + SLASH + str(
        idx_frame) + "_frame.png"
    image = imageio.imread(IMAGE_PATH)
    plants_color_mask = create_plants_color_mask(image, idx_frame)
    coordinates = get_seedbed_contour_rect_coordinates(plants_color_mask, idx_frame)
    get_seedbed(image, coordinates, idx_frame)"""

    """idx_frame = 1058
    # IMAGE_PATH = os.path.dirname(os.path.abspath(__file__)) + SLASH + "images" + SLASH + 'check' + SLASH + 'original' + SLASH + str(idx_frame) + "_frame.png"
    IMAGE_PATH = os.path.dirname(
        os.path.abspath(__file__)) + SLASH + "4_ANGLE_CORRECTED_FRAME" + SLASH + str(idx_frame) + "_frame.png"
    image = imageio.imread((IMAGE_PATH))
    print("Rows: " + str(image.shape[0]) + ", cols: " + str(image.shape[1]))
    # just_plants = create_mask_filled_by_plants(image)
    just_plants = create_plants_mask(image)
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '0_just_plants_frame.jpg',
                just_plants)
    # plt.imshow(just_plants)
    # plt.show()
    coordinates = get_seedbed_coordinates(just_plants)
    just_seedbed = get_seedbed(image, coordinates)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + '10_just_seedbed_frame.jpg',
                just_seedbed)"""
    # create_plants_mask(image)


    """print("Started")
    idx_frame = 1058
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = PATH_ROOT + SLASH + "4_ANGLE_CORRECTED_FRAME" + SLASH + str(
        idx_frame) + "_frame.jpg"
    image = imageio.imread((IMAGE_PATH))
    standard_size = int(np.ceil(image.shape[0] * (1.25))), int(np.ceil(image.shape[1] * (1.25)))
    print("Standard size: ", standard_size)
    img_obj = imgObj.ImageObj(image, 0, 0)
    centered_img_obj = center_seedbed(img_obj, standard_size, idx_frame)
    cv2.imwrite(PATH_ROOT + SLASH + 'TEST' + SLASH + 'centered_frame.jpg',
                centered_img_obj.get_image())
    """

    """print("Started")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    FOLDER_PATH = PATH_ROOT + SLASH + "8_CENTERED_STANDARD_FRAME" + SLASH
    source_path = FOLDER_PATH
    only_files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    only_files.sort()
    images_list = []
    for file_pointer in range(0, len(only_files), 1):
        print(only_files[file_pointer])
        image = imageio.imread(source_path + only_files[file_pointer])
        images_list.append(imgObj.ImageObj(image, 0, 0))
    images_list_output = trim_by_right(images_list)
    print("Images quantity: ", len(images_list_output))"""
