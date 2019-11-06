import cv2 as cv
import os
import imageio
import numpy as np

show_crosshair = False
from_center = False
template_name = "./template/subject01.gif"
template_cropped_name = "./template/template_cropped.jpg"


def search():
    template_image = cv.VideoCapture(template_name, 0)
    ret, frame = template_image.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    r = cv.selectROI("Your template", gray, show_crosshair, from_center)
    im_crop = gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # cv.imshow("test", im_crop)
    cv.imwrite(template_cropped_name, im_crop)
    cv.waitKey(0)
    cv.destroyAllWindows()
