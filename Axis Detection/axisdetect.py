#Program to detect the medial axis of video.

import cv2 as cv
import numpy as np
import math

#Reading the videos and then the frames
#And then applying the required processes.
def Process(vidpath):

    #VARIABLE DEFNS
    video = cv.VideoCapture(vidpath)
    #Object for background subtraction
    subtract = cv.createBackgroundSubtractorMOG2()
    cond = 1
    #Kernel for cleaning operation
    structureKernel = cv.getStructuringElement(cv.MORPH_ERODE, (6,6))

    while True:

        cond, frameorg = video.read()

        frame = subtract.apply(frameorg)#background subtraction

        frame = cv.erode(frame, structureKernel, iterations=3)#cleaning the image by erosion

        #Applying Sobel Filters to calculate the gradients and find the edges.

        gx = cv.Scharr(frame, cv.CV_16S, 1, 0, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        gy = cv.Scharr(frame, cv.CV_16S, 0, 1, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        abs_gx = cv.convertScaleAbs(gx)
        abs_gy = cv.convertScaleAbs(gy)
        frame = cv.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)

        #Detecting the straight lines in the image.
        #And drawing them on the original frame
        lines = cv.HoughLinesP(frame, 1, np.pi/180, 220, minLineLength=100, maxLineGap=70)

        if lines is not None:
            for i in range(0 , len(lines)):
                line = lines[i][0]
                cv.line(frameorg, (line[0], line[1]), (line[2], line[3]), (255,0,0), 2, cv.LINE_AA)


        cv.imshow("Edged", frameorg)
        k = cv.waitKey(1) & 0xFFF
        if k == 20:
            break

    cv.destroyAllWindows()




condition = input("Y/N")
if condition == "Y":
    Process("axis_1.mp4")
else:
    pass