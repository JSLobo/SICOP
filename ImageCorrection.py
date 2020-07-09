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
from os import listdir
from os.path import isfile, join
import imageio


def correct_angle(img_obj, idx):
    print("Correcting image angle")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + '/4_ANGLE_CORRECTED_FRAME/'):
            os.makedirs(PATH_ROOT + '/4_ANGLE_CORRECTED_FRAME/')

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

    mask_size = sizes_labels < (cols * rows) * 0.01
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

    # 10 - Correct angle in original img
    rotation_on_original_image = rotate_image(queryImg, result[0])
    # cropped_img, blackAndWhiteImage, rotation_on_original_image
    cv2.imwrite(PATH_ROOT + '/4_ANGLE_CORRECTED_FRAME/' + str(idx) + '_frame.jpg', rotation_on_original_image)
    img_obj = imgObj.ImageObj(rotation_on_original_image, 0, 0)

    return img_obj


def get_largest_line_setup(thresholded_image):
    blackAndWhiteImage = thresholded_image
    result = [0, 0, 0]
    result_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    col_limit = blackAndWhiteImage.shape[1] * 255
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = 0, int(np.ceil(rotated.shape[0] / 3)), rotated.shape[1]
        last_col = cols - 1
        #print(rotated.shape)
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
        #print(rotated.shape)
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


def create_mask_filled_by_plants(RGB_image):
    # Convert RGB image to chosen color space
    I = color.rgb2lab(RGB_image, illuminant="D50")

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.000
    channel1Max = 100.000

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = -46.769
    channel2Max = -3.750

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.993
    channel3Max = 91.735

    # Create mask based on chosen histogram thresholds
    sliderBW = (I[:, :, 0] >= channel1Min) & (I[:, :, 0] <= channel1Max) & \
               (I[:, :, 1] >= channel2Min) & (I[:, :, 1] <= channel2Max) & \
               (I[:, :, 2] >= channel3Min) & (I[:, :, 2] <= channel3Max)
    BW = sliderBW

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
        rects_area[index] = int(rect_tuple[3] * rect_tuple[2])
        index += 1
    max_index = np.argmax(rects_area, axis=0)
    print(max_index)

    return max_index


def get_seedbed_contour_rect_coordinates(RGB_image):
    print("Getting seedbed contour rect coordinates")
    #   Using object centroid coordinates to compute distance between

    #   1 - Get individual plant bounding circle
    im_gray = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
    # fig, ax1 = plt.subplots(figsize=(16, 9))
    # ax1.imshow(RGB_image)
    # ax1.set_xlabel("Plant mask binarizes", fontsize=14)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    #   2 - Get maximum radius value of individuals plants bounding circles

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

    #   5 - Isolate main seedbed
    seedbed_contours, seedbed_hierarchy = cv2.findContours(im_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(seedbed_contours))
    # seedbed_contours = max(seedbed_contours, key=cv2.contourArea)
    # print(type(seedbed_contours))
    #   6 - Getting main seedbed polygon
    # Approximate contours to polygons + get bounding rects
    contours_poly = [None] * len(seedbed_contours)
    bound_rect = [None] * len(seedbed_contours)
    for i, c in enumerate(seedbed_contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])
    print(bound_rect)
    print(len(bound_rect))
    # print(type(bound_rect))
    # bound_rect = max(bound_rect, key=cv2.contourArea)
    # bound_rect = get_mayor_area_boundrect(bound_rec)
    # print(type(bound_rect))
    drawing_im = np.zeros((im_gray.shape[0], im_gray.shape[1], 3), dtype=np.uint8)
    # Draw polygonal contour + bounding circles
    for i in range(len(seedbed_contours)):  # len(seedbed_contours)
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        im_with_contours = cv2.drawContours(drawing_im, contours_poly, i, color)
        # im_with_contours = cv2.drawContours(RGB_image.copy(), contours, -1, (0,0,255), 2)
        im_with_rect_bounding = cv2.rectangle(drawing_im, (int(bound_rect[i][0]), int(bound_rect[i][1])), \
                                              (int(bound_rect[i][0] + bound_rect[i][2]),
                                               int(bound_rect[i][1] + bound_rect[i][3])), color, 2)

    max_area_rect_index = find_major_rectangle_area(bound_rect)

    drawing_seedbed = np.zeros((im_gray.shape[0], im_gray.shape[1], 3), dtype=np.uint8)
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    seedbed_with_contours = cv2.drawContours(drawing_seedbed, contours_poly, i, color)
    # im_with_contours = cv2.drawContours(RGB_image.copy(), contours, -1, (0,0,255), 2)
    seedbed_with_rect_bounding = cv2.rectangle(drawing_seedbed, (
        int(bound_rect[max_area_rect_index][0]), int(bound_rect[max_area_rect_index][1])), \
                                               (int(
                                                   bound_rect[max_area_rect_index][0] + bound_rect[max_area_rect_index][
                                                       2]), int(
                                                   bound_rect[max_area_rect_index][1] + bound_rect[max_area_rect_index][
                                                       3])), color, 2)
    max_area_rect_atributes = bound_rect[max_area_rect_index]
    top_left = (max_area_rect_atributes[0], max_area_rect_atributes[1])
    bottom_left = (max_area_rect_atributes[0], max_area_rect_atributes[1] + max_area_rect_atributes[3])
    top_right = (max_area_rect_atributes[0] + max_area_rect_atributes[2], max_area_rect_atributes[1])
    bottom_right = (
        max_area_rect_atributes[0] + max_area_rect_atributes[2],
        max_area_rect_atributes[1] + max_area_rect_atributes[3])
    seedbed_coordinates = (top_left, top_right, bottom_left, bottom_right)

    return seedbed_coordinates # , circles_contour, im_with_rect_bounding, seedbed_with_rect_bounding


def get_seedbed_mask(img_obj):  # 10_Frames_zona_de_plantulas
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


def get_seedbed(image, seedbed_coordinates, idx):
    print("Getting_seedbed")
    # This function get an image with just the seedbed
    drawing_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    drawing_mask[seedbed_coordinates[0][1]:seedbed_coordinates[2][1], :, :] = image[seedbed_coordinates[0][1]:
                                                                                    seedbed_coordinates[2][1], :, :]
    seedbed_mask = drawing_mask

    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + '/5_SEEDBED/'):
            os.makedirs(PATH_ROOT + '/5_SEEDBED/')

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    cv2.imwrite(PATH_ROOT + '/5_SEEDBED//' + str(idx) + '_frame.jpg', seedbed_mask)
    # print(seedbed_coordinates[0][1], seedbed_coordinates[2][1])
    return seedbed_mask


def delete_repeated_frames(frames_list):
    print("Deleting repeated frames at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + '/2_DELETED_REPEATED_FRAME/'):
            os.makedirs(PATH_ROOT + '/2_DELETED_REPEATED_FRAME/')

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
        if R >= 0.9:
            cv2.imwrite(PATH_ROOT + '/2_DELETED_REPEATED_FRAME/' + str(idx) + '_frame.jpg', frames_list[idx])
            removed = removed + 1
        elif R < 0.9:
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

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def split_video_frames(VIDEO_PATH):
    print("Splitting video frames")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    frames_list = []
    # Read the video from specified path
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + '/1_VIDEO_SPLIT/'):
            os.makedirs(PATH_ROOT + '/1_VIDEO_SPLIT/')

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

        # fps = cap.get(5)
    # time_length = cap.get(7)/fps
    # frame_seq = 749
    # frame_no = (frame_seq /(time_length*fps))
    # print("Time length: ", time_length)
    # frame
    current_frame_name = 1000
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cap.read()

        if ret:
            # if video is still left continue creating images
            name = PATH_ROOT + '/1_VIDEO_SPLIT/' + str(current_frame_name) + '_frame.jpg'
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
            currentframe += 1
            current_frame_name += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, currentframe * 15)
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
    try:

        # creating a folder named data
        if not os.path.exists(PATH_ROOT + '/6_SCALED_FRAME/'):
            os.makedirs(PATH_ROOT + '/6_SCALED_FRAME/')

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
        img_dim = int(((each_tuple[0].get_image()).shape[1]) * rescale_factor), int(((each_tuple[0].get_image()).shape[0]) * rescale_factor)
        scaled_img = cv2.resize(each_tuple[0].get_image(), img_dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(PATH_ROOT + '/6_SCALED_FRAME/' + str(idx) + '_frame.jpg', scaled_img)
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
    img_obj_triplet_output = imgObj.ImageObj(standard_frame_1, 0, 0), imgObj.ImageObj(standard_frame_2, 0, 0), img_obj_triplet[2]
    return img_obj_triplet_output


def center_seedbed(image_obj, standard_size):
    print("Centering image object")
    complete_image = image_obj.get_image()
    plants_mask = create_mask_filled_by_plants(complete_image)
    seedbed_coordinates = get_seedbed_contour_rect_coordinates(plants_mask)
    standard_frame = np.zeros((standard_size[0], standard_size[1], 3), dtype=np.uint8)
    standard_frame_height_middle_point = int(np.ceil(standard_size[0]/2))
    seedbed_height_middle_point = seedbed_coordinates[0][1] + int(np.ceil((seedbed_coordinates[2][1] - seedbed_coordinates[0][1])/2))
    initial_row = standard_frame_height_middle_point - seedbed_height_middle_point
    standard_frame[initial_row:, :, :] = complete_image[seedbed_coordinates[0][1]:seedbed_coordinates[2][1], :, :]
    centred_frame_standard_img_obj = imgObj.ImageObj(standard_frame, 0, 0)
    return centred_frame_standard_img_obj


def trim_by_right(image_obj):
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


def homogenize_image_set(path):
    print("Homogenizing image set...")
    #  Splits video
    print("Started split_video_frames() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    frames_list = split_video_frames(path)  # pending generates frame_list into function
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    images_list = []
    final_images_list = []
    standard_size = int(np.ceil(frames_list[0].shape[0] * (1.25))), int(np.ceil(frames_list[0].shape[1] * (1.25)))
    init_frame_found = False
    idx_frame = 1000
    for each_frame in frames_list:
        print("Started correct_angle() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        angle_corrected_img_obj = correct_angle(imgObj.ImageObj(each_frame, 0, 0), idx_frame)
        print("Finished successfully at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Started create_mask_filled_by_plants() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        plants_mask = create_mask_filled_by_plants(angle_corrected_img_obj.get_image())
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Started get_seedbed_contour_rect_coordinates() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        seedbed_coordinates = get_seedbed_contour_rect_coordinates(plants_mask)
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Started get_seedbed_mask() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        only_seedbed = get_seedbed(angle_corrected_img_obj.get_image().copy(), seedbed_coordinates, idx_frame)
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        only_seedbed_img_obj = imgObj.ImageObj(only_seedbed, 0, 0)
        if init_frame_found or not is_trash_frame(only_seedbed_img_obj):
            init_frame_found = True
            images_list.append((angle_corrected_img_obj, seedbed_coordinates))
        idx_frame += 1
    print("Started establish_reference_size_for_scaling() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scaling_factor = establish_reference_size_for_scaling(images_list)
    print("Scaling factor: ", scaling_factor)
    print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Started escale() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scaled_images_list = scale(images_list, scaling_factor)
    idx_frame = 1000
    for each_img_obj in scaled_images_list:
        # standarized_frame_size_triplet = set_standard_size_frame(each_triplet, standard_size)
        print("Started center_seedbed() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        centered_img_obj = center_seedbed(each_img_obj, standard_size)
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Started trim_by_right() at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trimmed_img_obj = trim_by_right(centered_img_obj)
        print("Finished successfully  at ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        final_images_list.append(trimmed_img_obj)
        idx_frame += 1
    return final_images_list


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
    VIDEO_PATH = os.path.dirname(os.path.abspath(__file__)) + '/VID_cucharita.mp4'
    print("Started")
    homogenize_image_set(VIDEO_PATH)
    print("Finished with success")