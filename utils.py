import cv2 as cv
import os
import time
import numpy as np

show_crosshair = False
from_center = False
template_name = "./template/4_7.png"  # name of template in /template
template_cropped_name = "./template/template_cropped.jpg"  # name of template after cropping in /template
threshold = 0.7

# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


def get_template():  # select ROI template
    template_image = cv.VideoCapture(template_name, 0)
    ret, frame = template_image.read()
    if not ret:
        print("Could not read frame")
        exit()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.namedWindow("Your template", cv.WINDOW_NORMAL)
    cv.resizeWindow('Your template', 600, 493)
    cv.moveWindow("Your template", 660, 294)
    r = cv.selectROI("Your template", frame, show_crosshair, from_center)
    cv.destroyAllWindows()
    im_crop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    cv.imwrite(template_cropped_name, im_crop)

    return im_crop


def search():
    template = get_template()
    w, h = template.shape[::-1]
    cv.namedWindow("Results", cv.WINDOW_NORMAL)
    cv.resizeWindow('Results', 900, 493)
    cv.moveWindow("Results", 660, 294)
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


def viola():
    face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('./haarcascade_eye.xml')

    files = next(os.walk("./data"))[2]
    for photo in files:
        img = cv.imread(f"./data/{photo}")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        img = cv.resize(img, (664, 784))

        cv.imshow('Viola-Jones detector', img)
        cv.waitKey(5000)
    cv.destroyAllWindows()
