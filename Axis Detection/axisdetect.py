#Program to detect the medial axis of video.

import cv2 as cv
import numpy as np

#Reading the videos and then the frames
#And then applying the required processes.
def Process(vidpath):

    #VARIABLE DEFNS
    video = cv.VideoCapture(vidpath)
    #Object for background subtraction
    subtract = cv.createBackgroundSubtractorMOG2()
    cond = 1
    #Kernel for cleaning operation
    structureKernel = cv.getStructuringElement(cv.MORPH_ERODE, (5,5))


    while True:

        cond, frame = video.read()

        frameorg = subtract.apply(frame)#background_subtraction

        frame = cv.erode(frameorg, structureKernel, iterations=2)#cleaning the image by erosion

        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) Frame is already grayscale.
        #Applying Sobel Filters to calculate the gradients and find the edges.

        gx = cv.Sobel(frame, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        gy = cv.Sobel(frame, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        abs_gx = cv.convertScaleAbs(gx)
        abs_gy = cv.convertScaleAbs(gy)
        grad = cv.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)

        #Detecting the straight lines in the image.
        #And drawing them on the original frame
        lines = cv.HoughLinesP(grad, 1, np.pi/180, 150, minLineLength=50, maxLineGap=50)
        print(lines)



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