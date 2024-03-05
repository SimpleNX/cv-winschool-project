import cv2 as cv
import numpy as np
import math


map = cv.imread("KGP MAP2.png")
map_gray = cv.cvtColor(map, cv.COLOR_BGR2GRAY)
map_gray = cv.resize(map_gray, (1600,800))
sample_map = np.zeros((1600,800), dtype=np.uint8)

def sample_map():
    canny = cv.Canny(map_gray, 200, 300)
    cv.imshow("Canny", canny)
    lines = cv.HoughLinesP(canny, 1, np.pi/180, 1000, 2, 10, 10)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(canny, pt1, pt2, (255,255,255), 3, cv.LINE_AA)
    cv.imshow("Canny", canny)
    k = cv.waitKey(0) & 0xFFF
    cv.destroyAllWindows()

def RRT():


sample_map()
RRT()

cv.imshow("map", map_gray)
k = cv.waitKey(0) & 0xFFF
cv.destroyAllWindows()