import cv2 as cv
import os
import numpy as np

show_crosshair = False
from_center = False
template_name = "./template/yaleB11_P00A+005E+10.pgm"  # name of template in /template
template_cropped_name = "./template/template_cropped.jpg"  # name of template after cropping in /template
threshold = 0.6

# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


def get_template():  # select ROI template
    template_image = cv.VideoCapture(template_name, 0)
    ret, frame = template_image.read()
    if not ret:
        print("Could not read frame")
        exit()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    r = cv.selectROI("Your template", frame, show_crosshair, from_center)
    cv.destroyAllWindows()
    im_crop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    cv.imwrite(template_cropped_name, im_crop)

    return im_crop


def search():
    template = get_template()
    w, h = template.shape[::-1]
    for root, dirs, files in os.walk("./data"):  # find templates in all files in /data directory
        for file_name in files:
            image = cv.VideoCapture("./data/" + file_name, 0)
            ret, frame = image.read()

            if not ret:
                print("Could not read frame")
                exit()

            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_width, frame_height = frame.shape[::-1]

            matching = []

            for width in range(frame_width // 10, int(frame_width * 0.8), 5):  # cycle for resizing template
                height = int(width / w * h)
                if height > frame_height:
                    break

                template_copy = template.copy()
                template_copy = cv.resize(template_copy, (width, height))
                # print(width, "x", height)
                # print(template_copy.shape[::-1])
                # cv.imshow("resizing", template_copy)
                # cv.waitKey(2000)
                res = cv.matchTemplate(frame, template_copy, cv.TM_CCOEFF_NORMED)

                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

                if max_val > threshold:
                    matching.append((max_val, max_loc, width, height))

            # print("Max score is ", max_val)

            for match in matching:
                cv.rectangle(frame, match[1], ((match[1][0] + match[2]), (match[1][1] + match[3])), 255, 1)

            # loc = np.where(res >= threshold)
            # for pt in zip(*loc[::-1]):
            #     cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), 255, 2)

            # top_left = max_loc
            # bottom_right = (top_left[0] + w, top_left[1] + h)
            #
            # cv.rectangle(frame, top_left, bottom_right, 255, 2)
            # cv.imshow("results_test", res)
            cv.imshow("Results", frame)

            k = cv.waitKey(2000)
            if k == 27:  # Esc key to stop
                exit()
