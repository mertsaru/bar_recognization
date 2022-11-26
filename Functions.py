# This .py contains the functions we are using for bar recognition project

import numpy as np
from PIL import Image
from skimage import filters #(python -m pip install -U scikit-image)
from copy import deepcopy

from parameters import *

# Turns the image format and dimensions into a static format.
def reshape(img):
    pass

# Takes .jpeg file and turns into [n:m:1] matrix witn better border clearity
'''!Improve'''
def to_matrix(img, edge_multiplier = 500):
    '''Definition: edge multiplier
    Edge multiplier makes contrast with background and the lines. It makes easier to detect the lines.
    Edge multiplier should not be great, otherwise it creates noise near all the lines.
    '''

    ## Turning into [n:m:3] array
    img_mx = np.array(img)

    ## Making better contrast between borders and background
    edge_mx = filters.sobel(img_mx)

    ## Making edge more visible and saving as an RGB image
    '''!Idea
    Haven't checked if I can turn rgb into L in this step. if i can do it, the extra conversion to image and back to the matrix would be eliminated
    edge_mx values are between 0 and 1, that is why we are multiplying with edge_multiplier to make the range at least 0-255.
    But if we multiply with a bigger number such as 1000, then points that are close to 0s would be multiplied less than bigger ones
    and when we adjust everything to range 0-255 with astype(np.uint8), small numbers would be near 0 while big ones would be near 255. 
    '''
    '''Solution: Idea
    there is a calculation to turn rgb into L,
    if I can implement that, then we do not need to turn back into img and matrix again
    computation time saving'''
    '''Summary
    We still have [n:m:3] RGB matrix.
    That is why in this step we are transforming our image back to RGB after the border contrasting process.
    '''
    edge_img_rgb = Image.fromarray((edge_mx * edge_multiplier).astype(np.uint8),'RGB') 

    ## Changing RGB img into grayscale image 
    edge_img_gray = edge_img_rgb.convert('L') # convert library indicates we can transform while in matrix form. Maybe it would be faster? We can interfere in edge_mx *  multiplier.astype(uint8)

    ## Turning img into [n:m:1] array, values between 0-255
    edge_img_gray_mx = np.array(edge_img_gray) 

    return edge_img_gray_mx

# Takes the 'L' matrix form of the image and returns the matrix which the image only contains the bar.
def find_edge(img_mx):
    '''Summary
    First find left and right borders of the bar, then cut the bar with the given borders
    Then it would be easier to find top and bottom, since there would be less noise from the background.
    '''
    row_len, column_len = np.shape(img_mx)

    ## Finding the width of the bar
    img_mean_col = img_mx.mean(axis=0)

    threshold = amplifier * np.median(img_mean_col)

    ### Lefthand of the bar
    i = 0
    switch = False
    while (i < (column_len - bar_width)) and (not(switch)):
        col_l = img_mean_col[i]

        if col_l > threshold:
            switch = True
            for j in range(1 , bar_width):
                col_test = img_mean_col[i + j]
                '''!Idea
                Here we can also use derivative to find where the big change is, maximum change
                '''
                if col_test < threshold:
                    '''Summary
                    skips to the background since we know all of the col_l will fail until there.
                    '''
                    skip_step = j
                    i += skip_step + 1 
                    switch = False
                    break  # Turn back to while loop with index starting after i+j
        else:
            i += 1
    else:
        if switch:
            left_col_index = i        
        else:
            raise Exception('No left part of the bar found')

    ### Righthand of the bar
    '''Summary
    Starting from the bar width, where we left off, so we do not need to recalculate the whole process
    '''
    for j in range(left_col_index + bar_width, column_len): 
        col_test = img_mean_col[j]

        if (col_test < threshold):
            right_col_index = j-1
            break

    ## Finding the height of the bar
    '''Summary
    Cut the img for better mean performance.
    The cutted image give us less black values on the rows, so it would be less background noise.
    '''
    img_cutted_col_mx = img_mx[:,left_col_index:right_col_index]
    img_mean_row = img_cutted_col_mx.mean(axis=1)

    ### Top of the bar
    '''Summary
    We are using same threshold we used to find width of the bar, since threshold is a good number to divide black background and white lines.
    '''
    i = 0
    switch = False
    while (i < (row_len - bar_height)) and (not(switch)):
        row_u = img_mean_row[i]
        
        if row_u > threshold:
            switch = True
            for j in range(1 , bar_height):
                row_u_test = img_mean_row[i + j]
                
                if row_u_test < threshold:
                    skip_step = j
                    i += skip_step + 1
                    switch = False
                    break
        else:
            i +=1
    else:
        if switch:
            top_row_index = i
        else:
            raise Exception('No upper part of the bar found')
            
    ### Bottom of the bar
    i = 0
    switch = False
    while (i < row_len) and (not(switch)):
        row_b = img_mean_row[row_len -1 -i] # bottom i^th row
        
        if row_b > threshold:
            switch = True
            for j in range(1 , bar_height):
                row_b_test = img_mean_row[row_len -1 -i -j] # j above row of row_b

                if row_b_test < threshold:
                    skip_step = j
                    i += skip_step +1
                    switch = False
                    break
        else:
            i += 1
    else:
        if switch:
            bottom_row_index = row_len -1  -i
        else:
            raise Exception('No bottom part of the bar found')

    return [top_row_index , bottom_row_index , left_col_index , right_col_index]

# Finds the mid points of the sub-bars
'''!info!
We are using cutted version of the img. So if you want to see the mid-points on a whole img version,
you should add top_col_index(±1) to the line indices.'''
'''!Idea!
Use 5 bar finding functions:
smooth, dent, top_bottom, camel, sharp
Then put all into a list and use line eliminator to take the averaged lines
'''
'''?Question
How to take the border 0 to -0?
first find +0 then after is -0 coming?
if left part is + and after - comes?
if left part there is a 0 and some part later -(0) comes?
'''
'''Comment on find bars functions:
They have very poor performance regarding to smooth and dent with only corresponding to one example.'''
def find_bars_sharp(img_mx):
    lines = []
    row_vals = img_mx.mean(axis = 1)
    i=0
    while i < (len(row_vals) -1):
        delta_row = row_vals[i+1] - row_vals[i]
        if (delta_row >= 0) and (delta_row <= 1):
            for j in range(i, len(row_vals) -1):
                delta_row_j = row_vals[j+1] - row_vals[j]
                if delta_row_j < 0:
                    lines.append(j-1)
                    i = j+1
                    break
                else:
                    i = len(row_vals)
        else:
            i += 1    
    return lines

def find_bars_sharp_v2(img_mx):
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    i = 0
    while i < len(delta_row)-1:
        if (delta_row[i] > 0 ) and (delta_row[i+1] < 0):
            lines.append(i)
            i += 1
        else:
            i += 1
    return lines

def find_bars_sharp_v3(img_mx):
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    i = 0
    while i < len(delta_row)-1:
        if (delta_row[i] > 0) and (delta_row[i] < 1) and (delta_row[i+1] > -1) and (delta_row[i+1] < 0) :
            lines.append(i)
            i += 1
        else:
            i += 1
    return lines

def find_bars_smooth(img_mx, n=5):
    '''Improvement
    You can easily eliminate sign function. its unnecessary.
    '''
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    '''!Cleaning Idea!
    we can check sign of i+1 in the first if and take it out from the k loop. 
    Will it be faster?'''
    i = 0
    while i < len(delta_row) -n:
        if (delta_row[i] >0) and (delta_row[i] <1):
            sgn_i = sign(delta_row[i]) # sgn_i is +1 always
            switch = False
            
            '''description n:
            n chooses how far we want the function monotonically increasing or decreasing.
            '''
            ## Front loop
            for k in range(i+1,i+n+1):
                if sign(delta_row[k]) == sgn_i: # if sign i+n and i are same
                    switch = True
                    if k == i+1:
                        i += 1 
                    else:
                        i = k +n # start from k+n since we know k is positive but k-1 is negative, and Back-loop will fail until k
                    break
            if switch: # Turn back to while loop with i = k+n
                continue

            '''explanation: Why range j is taken like this
            If we took the range j like Front-loop range(i-n,i),
            then we would be checking signs of j from i-n to i-1,
            which I think, it would be slower, since there are some positives next to negative.
            '''
            ## Back Loop
            for j in range(1,n+1):
                if sign(delta_row[i-j]) != sgn_i: # if sign i-j and i are not same
                    switch = True
                    i = i -j +n+1 # start from there since until that point everything will fail with back-loop because of i-jth point
                    break
            if switch: # Turns back to while loop with i = i-j+n+1
                continue
            
            lines.append(i)
            i = i+n+1 # If we passed until here, we know i+1 until i+n are all negative so they will fail anyways

        else:
            i += 1
    return lines
    
def find_bars_dent(img_mx, n=15):
    lines = []
    row_vals = img_mx.mean(axis=1)

    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])
    
    i = 0
    while i< len(delta_row)-n:
        if (delta_row[i] >0) and (delta_row[i+1] <0):
            left_mean = np.mean(delta_row[i-n:i+1])
            right_mean = np.mean(delta_row[i+1:i+n+1])
            if (left_mean >0) and (right_mean <0):
                lines.append(i)
            i += 1
        else:
            i +=1
    return lines

# Eliminate close lines 
def line_elimenator_v2(lines):
    '''!Bug
    Last line is skipped
    Probably while j condition is not running
    '''
    i = 0
    new_lines = []
    while i < len(lines):
        first_line = lines[i]
        nhood_lines = [first_line]

        j = i+1
        while j < len(lines):
            next_line = lines[j]
            if (next_line - nhood_lines[-1]) < 10:
                nhood_lines.append(next_line)
                j += 1
                continue
            else:
                new_lines.append(int(np.round(np.mean(nhood_lines))))
                i = j
                break
        else:
            new_lines.append(int(np.round(np.mean(nhood_lines))))
            i +=1
            break

    return new_lines
 
# Line drawer
def show_line(img, lines):
    '''!Improvement
    Make the function dont change the original img_cut but creates a copy and shows that
    '''
    img_copy = deepcopy(img)
    for line in lines:
        img_copy[line,:] = 255
    img_viz = Image.fromarray(img_copy)
    img_viz.show()

# Check which line is reliable to be the middle point
'''Not working efficiently'''
def bar_reliability(img_mx, lines, error_function ='linear', max_dist =14):
    '''Parameters:
        error_function:
            has 2 values: linear, tanh(gauss)
            describes which function we are using to calculate the error
        max_dist:
            describes how far we want to calculate our error. If the difference between top(bottom) of the line to the end,
            it chooses that distance instead
    '''

    '''!Cleaning!
    If it would be a class, I wouldn't need to re-calculate row_vals and 
    it would be easier to implement lines, img_mx, img_name
    and it would be easier to store reliability of the lines
    '''
    '''Cleaning
    you are repeating line_names with distances, put them into parameters.'''
    line_name ={
        0 : 'albumin',
        1 : 'alpha1',
        2 : 'alpha2',
        3 : 'beta',
        4 : 'gamma'
        }    

    row_vals = img_mx.mean(axis=1)
    col_len = len(row_vals)
    
    # Finding the distance value
    if (lines[0] < max_dist) or (lines[-1] +max_dist > col_len -1): #if line ±n passes the border.
        distance = min(lines[0], col_len - lines[-1] -1)
    else:
        distance = max_dist

    '''!!IDEA!!
    Maybe it would be better if lines are too close to top or bottom,
    we can decrease it's reliability to a constant or with a multiplier.

    If it is not bright enough, we can decrease the reliability
    '''
    # Finding all the reliability
    validity_dict = {}
    for i in range(len(lines)):
        brightness = row_vals[lines[i]]
        # Finding individual error
        err = 0
        for dist in range(1,distance+1):
            new_err = abs(row_vals[lines[i] -dist] - row_vals[lines[i] +dist])
            err += new_err
        
        '''!Idea
        dividing 255 is taking every individual bar into same ground,
        but not all bars have same luminosity, error detection would not be equal
        try dividing into the line color
        
        Somehow better performance, works good with large max_dist
        but not enough, also error gets out of the 0-1 boundary

        Since you are using luminosity, dont use the amplified version,
        since amplified version is not correct interpretation.
        Use original grayscale version.
        '''
        '''Summary: mean error
        We are dividing error into distance to take the mean of the error,
        then we also divide it to 255 to normalize the error between 0-1,
        after that we also divide the error to the luminosity of the given line, since all lines are not in same luminosity, so the differences would be greater for lines that have bigger luminosity than others.'''
        mean_error = err /(brightness*255*distance) 
        
        # Turn error into reliability
        '''!Small error: Gauss
        not reliable
        erf(gauss) function is <~1 when it is at 1, we are not calculating error fully'''
        if (error_function == 'gauss') or (error_function == 'tanh') or (error_function == 'erf'):
            reliability = 1-erf(mean_error)
        elif (error_function == 'linear'):
            reliability = 1- mean_error
        
        # Add reliability to the dict
            validity_dict[line_name[i]] = reliability        
    
    return validity_dict

def bar_reliability_nodict(img_mx, lines, error_function = 'linear', max_dist =14):
    row_vals = img_mx.mean(axis=1)
    col_len = len(row_vals)
    
    # Finding the distance value
    if (lines[0] < max_dist) or (lines[-1] +max_dist > col_len -1): #if line ±n passes the border.
        distance = min(lines[0], col_len - lines[-1] -1)
    else:
        distance = max_dist

    validity_list = []
    # Finding all the reliability
    for i in range(len(lines)):
        brightness = row_vals[lines[i]]
        # Finding individual error
        err = 0
        for dist in range(1,distance+1):
            new_err = abs(row_vals[lines[i] -dist] - row_vals[lines[i] +dist])
            err += new_err
        
        mean_error = err /(brightness*distance) 
        
        # Turn error into reliability
        if (error_function == 'gauss') or (error_function == 'tanh') or (error_function == 'erf'):
            reliability = 1-erf(mean_error)
        elif (error_function == 'linear'):
            reliability = 1- mean_error
        
        # Add reliability to the list
        validity_list.append(reliability)
    
    return validity_list

# Returns distances of the bars
def line_dist(lines):
    '''!Idea
    For further calculations, we might include bar lenght to the calculation,
    and take the ratio of the distance to full bar length'''
    line_name ={
        0 : 'albumin',
        1 : 'alpha1',
        2 : 'alpha2',
        3 : 'beta',
        4 : 'gamma'
        }    
    
    dist_dict ={}
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            d_ij = f'{line_name[i]} - {line_name[j]}'
            distance = lines[j]-lines[i]
            dist_dict[d_ij] = distance
    
    return dist_dict

# If data is eligable for calculation, put it into a dataset for further calculation.
'''Not usable yet!!!'''
def validate_data(img_name, list):
    '''
    If you find more then 5 maybe take first 5
    if you find less then 5 then dont use the example
    '''
    validated = True

    if len(list) > 5:
        print(f'{img_name} is not usable for bar approximation! Too many bars!')
        validated = False
    
    elif len(list) < 5:
        print(f'{img_name} is not usable for bar approximation! Not enough bars!')
        validated = False

    return validated
