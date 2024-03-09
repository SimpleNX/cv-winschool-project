import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math


class Node:
    def __init__(self, x_cd, y_cd):
        self.x_cd = x_cd
        self.y_cd = y_cd
        self.child = []
        self.par = None


class RRT:
    def __init__(self, start, goal, img_arr, in_time,  step):
        self.tree = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.nearestNode = Node(0, 0)
        self.time = min(in_time, 150)
        self.img_arr = img_arr
        self.step = step
        self.total_dist = 0
        self.nearDist = 10000
        self.noWayP = 0
        self.wayPs = []

    #function to add child to the parent
    #complete
    def addchild(self, xc, yc):
        if self.goal.x_cd == xc:
            self.nearestNode.child.append(self.goal)
            self.goal.par = self.nearestNode
        else :
            newnode = Node(xc, yc)
            self.nearestNode.child.append(newnode)
            newnode.par = self.nearestNode

    #function to find the distance between two points.
    #complete
    def euc_dist(self, node1, s_point):
        x_diff = node1.x_cd - s_point[0]
        y_diff = node1.y_cd - s_point[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        return dist

    #Function to randomly sample points in space
    #Complete
    def sample(self):
        x = random.randint(1, image_arr.shape[1])
        y = random.randint(1, image_arr.shape[0])
        point = np.array([x, y])
        return point

    #function to check if the there is a obstacle in between the prev node and the sampled point
    #Complete
    def obstinbet(self, l_start, lend):
        unit = self.unitVec(l_start, lend)
        check_arr = np.array([0, 0])
        for i in range(self.step):
            check_arr[0] = l_start.x_cd + i*unit[0]
            check_arr[1] = l_start.y_cd + i*unit[1]
            if self.img_arr[check_arr[1], check_arr[0]] == 0:
                return True
            else:
                continue
        return False


    #Function to find the unit vector from the prev node and the sampled point
    #complete
    def unitVec(self, l_start, lend):
        unit_v = np.array([lend[0]-l_start.x_cd, lend[1]-l_start.y_cd])
        unit_v = unit_v/np.linalg.norm(unit_v)
        return unit_v

    #Function to find nearest node from the sampled point
    #Complete
    def nearest_node(self, root, s_point):
        if root is None:
            return
        dist = self.euc_dist(root, s_point)
        if dist <= self.step:
            self.nearestNode.x_cd = root.x_cd
            self.nearestNode.y_cd = root.y_cd
            self.nearDist = dist

        for child in root.child:
            self.nearest_node(child, s_point)

    #Function to extend the point from the sampled point to the nearest node.
    #Complete
    def extend(self, l_start, lend):
        move = self.step*self.unitVec(l_start, lend)
        new = np.array([l_start.x_cd+move[0], l_start.y_cd+move[1]])
        if new[0] >= self.img_arr.shape[1]:
            new[0] = self.img_arr.shape[1]-1
        if new[1] >= self.img_arr.shape[0]:
            new[1] = self.img_arr.shape[0]-1
        return new

    #Function to check if the goal has been reached or not.
    #complete
    def isgoal(self, s_point):
        step = self.step
        dis = self.euc_dist(self.goal, s_point)
        if 0 <= dis <= step:
            return True
        else:
            return False

    #Function to update the parameters.
    #Complete
    def changePara(self):
        self.nearestNode = Node(0, 0)
        self.nearDist = 10000

    #function to retrace the path from the goal to start
    def retrace(self, goal):
        #End Condition
        if goal.x_cd == self.tree.x_cd:
            return

        current_point = np.array([goal.x_cd, goal.y_cd])
        self.noWayP += 1
        self.wayPs.insert(0, current_point)
        self.total_dist += self.step
        self.retrace(goal.par)



#Reading the image and converting it into grayscale.
image = cv.imread("Map_KGP.png")
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = cv.resize(image_gray, dsize=(1200,800))
invert_image = cv.bitwise_not(image_gray)

#Converting the image into a binary numpy array
image_arr = np.zeros((800,1200), dtype=np.uint8)

#Only putting white pixels for roads and converting the image into a numpy array
for k in range(invert_image.shape[0]):
    for j in range(invert_image.shape[1]):
        pixel_intensity = invert_image[k, j]
        if 13 <= pixel_intensity <= 18:
            image_arr[k, j] = 255
        else:
            continue

end = (810, 138)
srt = (600, 136)

rrt_obj = RRT(srt, end, image_arr, 200, 4)

for a in range(rrt_obj.time):
    #Resetting the parameters every iteration
    rrt_obj.changePara()
    #Sampling a new point in space
    samp_rand = rrt_obj.sample()
    rrt_obj.nearest_node(rrt_obj.tree, samp_rand)
    new = rrt_obj.extend(rrt_obj.nearestNode, samp_rand)
    obst = rrt_obj.obstinbet(rrt_obj.nearestNode, new)

    if obst==False:
        rrt_obj.addchild(new[0], new[1])
        if rrt_obj.isgoal(new):
            rrt_obj.addchild(end[0], end[1])
            print("Reached")
            break

rrt_obj.retrace(rrt_obj.goal)
rrt_obj.wayPs.insert(0, srt)

#Drawing the iterations
for i in range(len(rrt_obj.wayPs)-1):
    cv.line(image_arr, (rrt_obj.wayPs[i][0], rrt_obj.wayPs[i][1]), (rrt_obj.wayPs[i+1][0], rrt_obj.wayPs[i+1][1]), [255,0,0], 1)
    cv.imshow("The Map", image_arr)
    k = cv.waitKey(1) & 0xFFF
    if k == 27:
        break

cv.destroyAllWindows()


fig = plt.figure("Check")
plt.imshow(image_arr, cmap='binary')
plt.show()



#node parent and node types check once