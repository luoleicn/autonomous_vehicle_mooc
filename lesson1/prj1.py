#!/usr/bin/python
#coding=utf8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2

import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip

def imageProcess(img):
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    #edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = img.shape
    vertices = np.array([[(200,650),(600, 450), (700, 450), (1200,650)]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(img)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on a blank image
    pos_w, pos_w_count, neg_w, neg_w_count = 0, 0, 0, 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            w = 1.0 * (y2-y1) / (x2-x1)
            print x1, y1, x2, y2, w
            if w >= 0.5 and w <= 0.8:
                pos_w += w
                pos_w_count += 1
            elif w <= -0.5 and w >= -0.8:
                neg_w += w
                neg_w_count += 1

    if pos_w_count > 0 :
        pos_w = 1.0 * pos_w / pos_w_count
        right_bottom = get_nearest_point(lines, (1030, 650))
        right_top = (get_target_x(right_bottom[0], right_bottom[1], pos_w, 450), 450)
        cv2.line(line_image,right_bottom,right_top,(255,0,0),20)
    if neg_w_count > 0 :
        neg_w = 1.0 * neg_w / neg_w_count
        left_bottom = get_nearest_point(lines, (340, 650))
        left_top = (get_target_x(left_bottom[0], left_bottom[1], neg_w, 450), 450)
        cv2.line(line_image,left_bottom,left_top,(255,0,0),20)

    #vertices = np.array([[(200,650),(600, 450), (700, 450), (1200,650)]], dtype=np.int32)
    #right_top = right_bottom * neg_w

    #vertices = np.array([[(200,650),(600, 450), (700, 450), (1200,650)]], dtype=np.int32)
    #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    # Draw the lines on the edge image
    new_img = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    #return img
    #return gray
    #return blur_gray
    #return edges
    #return masked_edges
    return new_img

def get_target_x(x1, y1, w, target_y):
    # y = wx + b
    b = y1 - w * x1
    target_x = int(1.0 * (target_y - b) / w)
    return target_x

def get_nearest_point(lines, point):
    distance = -1
    for line in lines:
        for x1, y1, x2, y2 in line:
            if distance < 0:
                distance = abs(point[0] - x1) + abs(point[1] - y1)
                opt_point = (x1, y1)

            distance_p1 = abs(point[0] - x1) + abs(point[1] - y1)
            distance_p2 = abs(point[0] - x2) + abs(point[1] - y2)
            if distance > distance_p1:
                distance = distance_p1
                opt_point = (x1, y1)
            if distance > distance_p2:
                distance = distance_p2
                opt_point = (x2, y2)
    return opt_point
        



new_clip_output = 'test_output.mp4'
test_clip = VideoFileClip("test.mp4")
new_clip = test_clip.fl_image(lambda x: imageProcess(x)) 
new_clip.write_videofile(new_clip_output, audio=False)

#sample_frame = test_clip.get_frame(10)
#plt.imshow(imageProcess(sample_frame))
#plt.show()

