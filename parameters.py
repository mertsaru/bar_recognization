#Some parameters for the bar recognisation function

# Error function for probability calculation
from math import erf

epsilon = 10 # pixel density difference
deamplifier = 0.7
amplifier = 1.3

'''Summary: bar height, width
Make bar height and width smaller than the expected bar height and width. These parameters only help to
pass the frames in the image file. We have another implementation to find the real bar height and width
'''
bar_height = 10 # to eliminate small bars (aka frame  lines), we need to define the bar's dimensions(in pixels).
bar_width = 15

line_height = 5

'''Summary: edge multiplier
Edge multiplier makes contrast with background and the lines. It makes easier to detect the lines.
Edge multiplier should not be great, otherwise it creates noise near all the lines.
'''
#edge_multiplier = 500 # amplifies the edges in img_mx

# returns >= 0 as +1 and rest -1
def sign(x):
    if x<0:
        return -1
    return 1

    