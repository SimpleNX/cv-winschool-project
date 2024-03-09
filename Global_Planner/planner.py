import numpy as np
import cv2 as cv
import math


class Node:
    def __init__(self, x_cd, y_cd, refpar):
        self.x_cd = x_cd
        self.y_cd = y_cd
        self.refpar = refpar


class RRT :
    def __init__(self, start, goal, img_arr, step):
        self.start = start
        self.goal = goal
        self.img_arr = img_arr
        self.step = step

    #Function to randomly sample points
    def sample(self):
        pass

    #Function to find nearest node
    def nearest_node(self, point):
        pass

    #Function to extend the point
    def extend(self, node_prev, node_next):
        pass

    #Function to continue the algorithm for a specified number of iterations
    def cont(self, max_time):
        pass


#Reading the image and converting it into grayscale.
image = cv.imread("Map_KGP.png")
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = cv.resize(image_gray, dsize=(1200,800))
invert_image = cv.bitwise_not(image_gray)

#Converting the image into a binary numpy array
image_arr = np.zeros((800,1200), dtype=np.uint8)

#Only putting white pixels for roads and converting the image into a numpy array
for i in range(invert_image.shape[0]):
    for j in range(invert_image.shape[1]):
        print(i, j)
        pixel_intensity = invert_image[i, j]
        if 13 <= pixel_intensity <= 18:
            image_arr[i, j] = 255
        else:
            continue

cv.imshow("Current", image_arr)
k = cv.waitKey() & 0xFFF
cv.destroyAllWindows()