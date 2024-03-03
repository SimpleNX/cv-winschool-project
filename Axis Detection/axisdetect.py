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

        threshold = 150

        #For detection of the lines in the edge frame.
        lines = hough_detection(frame, threshold, -np.pi/2, np.pi/2, frameorg)

        #For drawing the lines on the original image.
        for rho, theta in lines:
            x0 = changespace(rho, theta)
            direction = np.array([x0[1], -x0[0]])
            pt1 = np.round(x0 + 1000 * direction).astype(int)
            pt2 = np.round(x0 - 1000 * direction).astype(int)
            cv.line(frameorg, pt1, pt2, [0, 255, 255], 3, cv.LINE_AA)

        cv.imshow("Edged", frameorg)
        k = cv.waitKey(1) & 0xFFF
        if k == 20:
            break

    cv.destroyAllWindows()


def hough_detection(edges: np.ndarray, threshold:float, min_theta:float, max_theta:float, image) -> np.ndarray:
    diagonal = math.sqrt(image.shape[0]**2 + image.shape[1]**2)

    theta = 0.2
    rho = 10

    #Defining the parameters and their range for the calculation
    thetas = np.arange(min_theta, max_theta, theta)
    rhos = np.arange(-diagonal, diagonal, rho)

    #Dividing the space into bunch of small cells.
    num1 = len(thetas)
    num2 = len(rhos)
    cells = np.zeros([num1, num2])

    #Calculation values of sines and cosines.
    sines = np.sin(thetas)
    cosines = np.cos(thetas)

    #Considering only the edges
    xe, ye = np.where(edges > 0)

    #Now iterating through the points and plotting them in the Hough Space with parameters m and b.
    #Also voting for the cells if they have a line passing through them
    for x,y in zip(xe, ye):
        for t in range (len(thetas)):
        #Calculating the actual line in the hough space as rho = x*sin(theta) + y*cos(theta)
            rho_space = x*sines[t] + y*cosines[t]
        #Now finding the cell rho nearest to the rho calculating and voting for the cell where it is present.
            pos_cell = np.where(rho_space > rhos)[0][-1]
            cells[pos_cell, t] += 1

    #Taking the cell and the coordinates which have been voted the most.
    matched_rho_idx, matched_theta_idx = np.where(cells>threshold)
    matched_rho = rhos[matched_rho_idx]
    matched_theta = thetas[matched_theta_idx]

    polar_coordinates = np.vstack([matched_rho, matched_theta]).T
    return polar_coordinates

def changespace(radius: np.ndarray, angle: np.ndarray, cv2_setup: bool = True) -> np.ndarray:
    return radius * np.array([np.sin(angle), np.cos(angle)])



condition = input("Y/N")
if condition == "Y":
    Process("axis_1.mp4")
else:
    pass