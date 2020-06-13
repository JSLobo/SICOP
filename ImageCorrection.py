import ImageObj as imgObj
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
import cv2


def correct_angle(img_obj):
    print("Correcting image angle")
    image = img_obj.get_image()
    queryImg = image
    img = queryImg.copy() - createMask_filled_byPlants(queryImg)

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img, cmap="gray")
    ax1.set_xlabel("Difference result image", fontsize=14)

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
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img_luv)
    ax1.set_xlabel("Layer L of LUV image before morphology on RGB color space", fontsize=14)

    # ---------------------------------------------------------------------------
    # Morphology with RGB color space
    cols = img.shape[1]
    horizontal_size = cols // int(np.ceil(cols * 0.2))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    for i in range(0, 2, 1):
        img_erode = cv2.erode(img, horizontalStructure)  # ---- resalta lo brillante en linea recta
        img = cv2.dilate(img_erode, horizontalStructure)  # ---- reduce los artefactos oscuros
        print("Morphology with RGB color space")
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img)
    ax1.set_xlabel("Image after morphology on RGB color space", fontsize=14)

    # Morpholoy with 'l' layer from 'LUV' color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    img = img[:, :, 0]
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(img)
    ax1.set_xlabel("Layer L of LUV image after morphology on RGB color space", fontsize=14)
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

    fig, (ax4) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax4.imshow(bw_img, cmap="gray")
    ax4.set_xlabel("Adaptative threshold", fontsize=14)

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

    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical, cmap="gray")
    ax6.set_xlabel("Extracted vertical lines up 15%", fontsize=14)

    # Analysis of connected componentes
    label_im, nb_labels = ndimage.label(vertical)

    sizes_labels = ndimage.sum(vertical, label_im, range(nb_labels + 1))

    mask_size = sizes_labels < (cols * rows) * 0.5
    remove_pixel = mask_size[label_im]

    vertical[remove_pixel] = 0
    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical, cmap="gray")
    ax6.set_xlabel("Small objects removed", fontsize=14)

    verticalsize_2 = rows // int(np.ceil(rows * 0.25))
    verticalStructure_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize_2))
    vertical_2 = cv2.erode(vertical, verticalStructure_2)
    vertical_2 = cv2.dilate(vertical_2, verticalStructure_2)
    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical_2, cmap="gray")
    ax6.set_xlabel("Extracted vertical lines up 25%", fontsize=14)

    # 5.3 - Show extracted vertical lines
    '''fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
    ax6.imshow(vertical, cmap="gray")
    ax6.set_xlabel("Extracted vertical lines", fontsize=14)'''

    vertical_final = vertical - vertical_2
    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical_final, cmap="gray")
    ax6.set_xlabel("Extracted vertical lines up 15% from 25%", fontsize=14)

    # Analysis of connected componentes
    label_im, nb_labels = ndimage.label(vertical_final)

    sizes_labels = ndimage.sum(vertical_final, label_im, range(nb_labels + 1))

    mask_size = sizes_labels < (cols * rows) * 0.01
    remove_pixel = mask_size[label_im]

    vertical_final[remove_pixel] = 0
    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical_final, cmap="gray")
    ax6.set_xlabel("Small objects removed", fontsize=14)

    vertical_final = cv2.dilate(vertical_final, horizontalStructure)
    fig, (ax6) = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16, 9))
    ax6.imshow(vertical_final, cmap="gray")
    ax6.set_xlabel("Extracted horizontal structure", fontsize=14)
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
    img_obj = imgObj.ImageObj(rotation_on_original_image)

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
        print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print(result)
    result_list[0] = result

    result = [0, 0, 0]
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = int(np.ceil(rotated.shape[0] / 3) + 1), int(np.ceil((rotated.shape[0] / 3) * 2)), \
                                      rotated.shape[1]
        last_col = cols - 1
        print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print(result)
    result_list[1] = result

    result = [0, 0, 0]
    for angle in range(-20, 20, 1):
        rotated = rotate_image(blackAndWhiteImage, angle)
        rows_init, rows_limit, cols = int(np.ceil(((rotated.shape[0] / 3) * 2) + 1)), rotated.shape[0], rotated.shape[1]
        last_col = cols - 1
        print(rotated.shape)
        for index_row in range(rows_init, rows_limit, 1):
            vector_line = rotated[index_row:index_row + 1, 0:cols]
            # print(vector_line.shape)
            summatory = np.sum(vector_line, axis=1)
            if summatory > result[2] and summatory <= col_limit:
                result[0] = angle
                result[1] = index_row
                result[2] = summatory
                # print(summatory)
    print(result)
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


def createMask_filled_byPlants(RGB_image):
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
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

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
