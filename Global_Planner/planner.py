import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math


class Node:
    #Class for the tree the points will be represented in
    def __init__(self, x_cd, y_cd):
        self.x_cd = x_cd
        self.y_cd = y_cd
        self.child = [] #List for child nodes
        self.par = [] #Parent Node to a node


class RRT:
    #Class for the RRT Algorithm
    def __init__(self, start, goal, img_arr, in_time: int,  step: int):
        self.tree = Node(start[0], start[1])    #The start Node
        self.goal = Node(goal[0], goal[1])  #The goal Node
        self.nearestNode = []   #List of all nearest nodes
        self.time = min(in_time, 1000) #Min number of iterations
        self.img_arr = img_arr #Map
        self.step = step
        self.total_dist = 0
        self.nearDist = 10000
        self.no_path_points = 0 #Number of path points used to reach the start from the goal
        self.path_points = [] #The path points used to reach the start node from the goal node.

    #function to add child to the parent
    def addchild(self, xc, yc):
        print(xc, yc)
        if self.goal.x_cd == xc and seld.goal.y_cd == yc:  #checking if the current node is the goal node
            self.nearestNode[0].child.append(self.goal)
            self.goal.par.append(self.nearestNode[0])
        else:
            newnode = Node(xc, yc)
            self.nearestNode[0].child.append(newnode)
            newnode.par.append(self.nearestNode[0])

    #Function to randomly sample points in space
    def sample(self):
        x = random.randint(0, image_arr.shape[1]-1)
        y = random.randint(0, image_arr.shape[0]-1)
        point = np.array([x, y])
        print(point)
        return point

    #Function to extend the point from the sampled point to the nearest node.
    #Complete
    def extend(self, l_start, lend):
        #Using a unit vector to steer the path towards the previously sampled point
        move = self.step*self.unitVec(l_start, lend)
        new = np.array([l_start.x_cd+move[0], l_start.y_cd+move[1]])
        #Boundary conditions
        if new[0] >= self.img_arr.shape[1]:
            new[0] = self.img_arr.shape[1]-1
        if new[1] >= self.img_arr.shape[0]:
            new[1] = self.img_arr.shape[0]-1
        return new

    #function to check if the there is a obstacle in between the prev node and the sampled point
    def obstinbet(self, l_start, lend):
        unit = self.unitVec(l_start, lend)
        check_arr = np.array([0, 0])
        for i in range(self.step):
            check_arr[0] = l_start.x_cd + i*unit[0]
            check_arr[1] = l_start.y_cd + i*unit[1]
            print(self.img_arr[check_arr[0], check_arr[1]])
            #Continuing the iteration if the there is only free space between the nearest node and the sampled point
            if self.img_arr[check_arr[1], check_arr[0]] == 255:
                print(self.img_arr[check_arr[0], check_arr[1]])
                continue
            else:
                return False
        return True


    #Function to find the unit vector from the prev node and the sampled point
    def unitVec(self, l_start, lend):
        unit_v = np.array([lend[0]-l_start.x_cd, lend[1]-l_start.y_cd])
        unit_v = unit_v/self.euc_dist(l_start, lend)
        return unit_v

    #Function to find nearest node from the sampled point
    def nearest_node(self, root, s_point):
        if not root:
            return
        dist = self.euc_dist(root, s_point)
        #Updating the nearest node if the current sampled point is closer to the root
        if dist <= self.nearDist:
            self.nearestNode.append(root)
            self.nearDist = dist
        #Checking for children in the list for the nearest node to the previously sampled point
        for child in root.child:
            self.nearest_node(child, s_point)

    #function to find the euclidean distance between two points.
    def euc_dist(self, node1, s_point):
        x_diff = node1.x_cd - s_point[0]
        y_diff = node1.y_cd - s_point[1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        return dist

    #Function to check if the goal has been reached or not.
    #complete
    def isgoal(self, s_point):
        step = self.step
        dis = self.euc_dist(self.goal, s_point)
        #If the goal is within the step size
        if dis <= step:
            print(s_point)
            print(" up")
            return True
        else:
            return False

    #Function to update the parameters.
    #Complete
    def changePara(self):
        self.nearestNode = []
        self.nearDist = 10000

    #function to retrace the path from the goal to start
    def retrace(self, goal):
        print("retrace")
        #End Condition
        if goal.x_cd == self.tree.x_cd:
            return
        self.no_path_points += 1 #+1 if one point is traversed along the path
        current_point = np.array([goal.x_cd, goal.y_cd])
        self.path_points.insert(0, current_point)
        self.total_dist += self.step
        self.retrace(goal.par[0])


#Reading the image and converting it into grayscale.
image = cv.imread("Map_KGP.png")
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_gray = cv.resize(image_gray, dsize=(1200,800))
invert_image = cv.bitwise_not(image_gray)

#Converting the image into a binary numpy array
image_arr = np.zeros((800,1200), dtype=np.uint8)

#Only putting white pixels for roads and converting the image into a numpy array
#Only highlighting the roads
for k in range(invert_image.shape[0]):
    for j in range(invert_image.shape[1]):
        pixel_intensity = invert_image[k, j]
        if 13 <= pixel_intensity <= 18:
            image_arr[k, j] = 255
        else:
            continue

fig = plt.figure("Check")
plt.imshow(invert_image, cmap='binary')
plt.show()

end = (810, 138)
srt = (600, 136)

rrt_obj = RRT(srt, end, image_arr, 2000, 1)

for a in range(rrt_obj.time):
    #Resetting the parameters every iteration
    rrt_obj.changePara()
    #Sampling a new point in space
    samp_rand = rrt_obj.sample()
    #Finding a new node to the currently sampled point
    rrt_obj.nearest_node(rrt_obj.tree, samp_rand)
    #Steering towards the point
    new = rrt_obj.extend(rrt_obj.nearestNode[0], samp_rand)
    #Checking if inaccesible ares lies in between the node and the steered path
    obst = rrt_obj.obstinbet(rrt_obj.nearestNode[0], new)

    if obst==False: #If no object is in between it is added in the child list
        print("Not obstacle")
        rrt_obj.addchild(new[0], new[1])
        if rrt_obj.isgoal(new):
            rrt_obj.addchild(end[0], end[1])
            print("Reached")
            cv.destroyAllWindows()
            break

rrt_obj.retrace(rrt_obj.goal) #Calling the function to retrace the paths
rrt_obj.path_points.insert(0, srt)

#Drawing the points on the original map to show the path retraced by the algorithm
for i in range(len(rrt_obj.path_points)-1):
    cv.line(image_arr, (rrt_obj.path_points[i][0], rrt_obj.path_points[i][1]), (rrt_obj.path_points[i+1][0],
                                                                                rrt_obj.path_points[i+1][1]), [255, 0, 0], 1)
    cv.imshow("The Map", image_arr)
    k = cv.waitKey(1) & 0xFFF
    if k == 27:
        break

cv.destroyAllWindows()


