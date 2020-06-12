import ImageObj as img
import ImageCorrection as corrector
from os import listdir
from os.path import isfile, join
import imageio
import cv2


def main():
    source_path = "/../images/"
    destiny_path = "/../results/"
    files_path = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    files_path.sort()
    output_name_counter = 1
    for file_pointer in range(0, len(files_path), 1):
        print(files_path[file_pointer])
        image = imageio.imread(source_path + files_path[file_pointer])
        img_obj = corrector.correct_angle(img.ImageObj(image))
        image_result = img_obj.get_image()
        output_name = destiny_path + 'image_' + str(output_name_counter) + '.jpg'
        cv2.imwrite(output_name, image_result)
        output_name_counter += 1


if __name__ == '__main__':
    main()
