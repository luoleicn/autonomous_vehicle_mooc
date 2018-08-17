# **Finding Lane Lines on the Road** 


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

+ convert the images to grayscale
+ use gaussian blur to remove some noise
+ find edges
+ mask region of interest
+ apply hough transforms to find lane lines

In order to draw a single line on the left and right lanes, I modified the hough_lines() function by 

+ filter the unreasonable gradient whose absolute value is larger than 0.8 or less than 0.5
+ calculate lane line making the height between 320 and 550
+ averge the left and right lines above 



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when some noise occurs in the interest region 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use momentum（An optimization method used in neutral network）to make the result more stable.
