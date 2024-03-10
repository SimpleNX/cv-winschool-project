cv-winschool-project
------------
------------

For Winter School Projects
Contains documenation for the Projects.

Medial-Axis-Detection
---------------------
---------------------

 The code includes two major functions to do all the operations
 Process Function
 ----------------
 The variables and parameters for the loop are defined.
 Using video capture, an video is read.
 'subtract' is used to create a background subtraction object to use later for
 backk=ground subtraction process.
 'structureKernel' is used to create a Kernel for cleaning the image.

 using video.read, frames of the video are extracted every iteration

 ## Background subtraction
 A direct background subtraction method is applied from the OpenCV library.
 Other cleaning functions caught more noise than this method.

 Manual background subtraction using frame differences lead to worse quality and bad frames,       because in the frames the environment also is changing and it is also captured in addition to the desired object.

 ## Cleaning the Image
 A direct cleaning function 'cv.erode' is used to clean the image from the Open CV Library

 ## Detecting the edges on the Image
 To detect the edges, Sobel Filters were used first.
 The Sobel Filters were less accurate.

 For more accuracy, 
 Schurr Filters were used to detect the edges.
 'cv.Schurr' inbuilt function is used to calculate the derivatives along the x-axis and the y-axis.
 After, the derivatives are added with the weighted formula to get the edges.

 Observation : The Schurr Filter does not detect edges as accurate as the Canny Edge Detector.

 ## Hough Line Detection

 Note : The above code is able to detect and draw lines with the inbuilt Hough Transform functions.

 A separate function for the Hough Transform

 All the paramters and the variables are defined.
 Here (rho = x*sin(theta) + y*cos(theta)) 

 'diagonal' is defined to get the limits for the values of rho in the hough space.
 'theta', 'rho' are defined to create a certain thershold for the respective values.

 Now the entire image space dimension is divided into smaller cells.
 (So that the lines can vote for individual cells, which would yield points through which the line passes in the image.)

 In the print statement for the edges, 'NoneType' edges are also detected by the Schurr Filter.
 To bypass this, before the loop the edges with values are considered.

 Now to check for lines,
 Iterating through all the edges and then plotting the edges in the parametric Hough Space and voting to get the correct cell.
 and returning the lines.

 (The Hough Space detect function has some error to be debugged due to which it is not able to return the lines.
 There is a index out of range error on running the Hough Space function.)

 Features/Parts not working or not added.
  1. Hough Space function
  2. If the inbuilt functions are added, the edges of the object and the line(two in total) are drawn, taking the average of these two lines and marking the medial axis is not in the code. The average method would also not work with Hough Space because the order or the edges cannot be known without printing them out.

References :
[https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html]



Global Planner 
-------------
-------------

The code includes classes to define the RRT algorithm and the Tree data structure used here.

# 'class Node'
  It includes an implementation of a Tree data structure
  In this case, a tree node would have a x-coordinate and a y-coordinate and children nodes and parent node(s).
  ## Problem
  The implementation here did not work for _self.par = None_
  Because NoneType cannot have item assignment
  But online references used the above method and got the results for any class similiar to this structure.
  Also Tried _set_nextnode_ function(not inbuilt) to set the parent value to the previous node,
  but wasn't able to read through python documenatations on getting around this error.
Alternatively used _self.par=[]_ to assign an empty list and append the nearest node to it.

# 'class RRT'
 It includes the implementation of the RRT Algorithm and its functions

 ## _init_
 All of the required variables are variables are assigned in the constructor.

 ## Problem
 The _self.nearestNode_ had similiar issues as _self.par_ hence used an empty list to append the point which is the nearest node.

 ## _addchild()_
 The function is used to add a sampled point as a child to an node if it is checked that it is in the free space.
 Also there is a check for the condition that point is the goal itsef.

 ## Problem
 The parent of the goal node is assigned _NoneType_ even when none value is not used in the code. This leads to discrepancies in the retrace function.

 # _sample()_
 The function is to randomly sample points in the space, but because a basic random function is used it takes more time that it would have taken if a generator biased towards the free space was used.
 This returns a newly sampled point.

 `Works`

 # _extend()_
 This function is used move from the nearest_node to the newly sampled point using the step size provided.
 A boundary condition to check if the extension is out of the image or not.

 `Works`

 # _obstinbet()_
 This function is used to check if there is any obstacle in between the point.
 By moving with +1 step till _self.size_, it is check if the extension is on the free space(road), if not then there is a obstacle in between the point and the nearest node and it returns false.

 # _unitVec()_
 Returns a unit vector along the direction of nearest node to the newly sampled point.

 `Works`

 # _nearest_node()_
 The function is used to find the nearest node to the newly sampled point(returned from the sample function).
 If there lies a node, where the distance between the node and the sampled point is less than the step size, the node is nearest node to the point is added to _nearestNode_ list.
 The function is recursively called to check for the child of the root(start) which lies nearest to the sampled point.

 `Works`

 # _euc_dist()_
 Finds the euclidean distance between the point and a node.

 `Works`

 # _isgoal()_
 Checks if the current point is the goal node or not. Returns boolean value accordingly.

 `Works`

# _retrace()_
This is function is used to retrace the path by beginning from the goal and moving the goal's parent nodes to reach the root(start) node and everytime it does it adds the point to the _path_points_ list which would be later used to draw the lines on the map.
Also used recursively to move along the parents.

## Problems
Due to the _NoneType_ assignment to the goal.par the function is not able to retrace the path along the nodes.
Due to which the lines cannot be drawn.

# Outer definition

## Configuration space definition and finding the points
An image of the map is read and converted into grayscale.
A numpy array is defined to plot all the roads on the map.

Loop used to mark all the points with the lowest intensity to be roads/free space on the numpy array.
The lowest intensity range is used to mark the roads because the bitwise_not operator assigned the roads from the original with lowest intensities.

Used Matplotlib.pyplot to get the _start_ point and the _goal_ point.

`Works`

 ## Loop for RRT Class Implementation
 The loop is used to implement the RRT Class so that the RRT algorithm is applied onto the map.

## Loop to draw the lines

# Problems with the code
Mentioned with the functions which have problems.
Summary of the problems :
1. The problem with using empty lists or _NoneType_ with the nodes cause problems whenever there is node assignment.
2. The _sample()_ function is slow.
3. The _addchild()_ function assigns _NoneType_ value to the parent nodes which cause subsequent problems.
4. The _retrace()_ function does not work in its entirety.

The above codes do not work completely, parts of them work.

 
 
