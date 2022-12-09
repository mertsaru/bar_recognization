# This .py contains the functions we are using for bar recognition project

import numpy as np
from PIL import Image
from skimage import filters #(python -m pip install -U scikit-image)
from copy import deepcopy


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

# SPEP finding
### extracts albumin dimension of the mask img
def find_albumin(img, bar_height=20, bar_width =15, amplifier=1.3):
    row_len, column_len = np.shape(img)

    ## Finding the width of the albumin
    img_mean_col = img.mean(axis=0)
    global threshold
    threshold = amplifier * np.median(img_mean_col)
    print(threshold)
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

    ## Finding the height of the albumin
    '''Summary
    Cut the img for better mean performance.
    The cutted image give us less black values on the rows, so it would be less background noise.
    '''
    img_cutted_col_mx = img[:,left_col_index:right_col_index]
    img_mean_row = img_cutted_col_mx.mean(axis=1)

    ### Top of the albumin
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

    # Bottom of the albumin

    for j in range(top_row_index + bar_height, row_len): 
        row_test = img_mean_row[j]

        if (row_test < threshold):
            bottom_row_index = j
            break
    return [top_row_index, bottom_row_index, left_col_index, right_col_index]
### creates mask from find_albumin
def create_mask(list,buffer =5):
    top,bottom,left,right = list
    height = bottom - top +buffer
    width = right - left +buffer
    return np.ones((height,width))
### use mask to find spep
def locate_spep(img_mx,mask,step=3):
    row_len,col_len = np.shape(img_mx)
    col_len_q = col_len/4
    row_len_q = row_len/4

    cutout_height,cutout_width = np.shape(mask)
    folder_max = 0
    i=0
    while i<col_len_q:
        j=0
        while j< row_len_q:
            cutout = img_mx[i: i+cutout_height, j: j+cutout_width]
            folder = np.multiply(cutout,mask)
            folder_value = folder.sum()
            if folder_value > folder_max:
                folder_max = folder_value
                top_row_index = i
                left_col_index = j
                j += step
            else:
                j += step
        else:
            i += step
    
    right_col_index = left_col_index + cutout_width

    ## Open the folder
    img_cut = img_mx[:,left_col_index:right_col_index]
    img_mean_row_left = img_cut[:,0:round(cutout_width/2)].mean(axis=1)
    img_mean_row_right = img_cut[:,round(cutout_width/2):-1].mean(axis=1)

    i = 0
    switch = False
    while (i < row_len) and (not(switch)):
        left_row = img_mean_row_left[row_len -1 -i] # bottom i^th row's left half
        right_row = img_mean_row_right[row_len -1 -i] # bottom i^th row's right half
        if (left_row > threshold) and (right_row > threshold):
            switch = True
            for j in range(1 , bar_height=10):
                left_row_test = img_mean_row_left[row_len -1 -i -j] # j above row of row_b
                right_row_test = img_mean_row_right[row_len -1 -i -j]
                if (left_row_test < threshold) or (right_row_test < threshold):
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
            print('no bottom of the bar found')
    return [top_row_index,bottom_row_index,left_col_index,right_col_index]
### cut the spep part of the img
def cut_spep(img,boundaries):
    img_copy = deepcopy(img)
    t,b,l,r = boundaries
    img_copy = img_copy[t:b,l:r]
    return img_copy

# Finding lines
## Finds the mid points of SPEP lines to analyze the bar
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
def find_peak_sharp(img_mx):
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

def find_peak_sharp_v2(img_mx):
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

def find_peak_sharp_v3(img_mx):
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

def find_peak_smooth(img_mx, n=5):
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
            switch = False
            
            '''description n:
            n chooses how far we want the function monotonically increasing or decreasing.
            '''
            ## Front loop
            for k in range(i+1,i+n+1):
                if delta_row[k] >= 0: # if sign i+n and i are same
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
                if delta_row[i-j] < 0: # if sign i-j and i are not same
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
    
def find_peak_dent(img_mx, n=15):
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


def find_trough_sharp(img_mx):
    lines = []
    row_vals = img_mx.mean(axis = 1)
    i=0
    while i < (len(row_vals) -1):
        delta_row = row_vals[i+1] - row_vals[i]
        if (delta_row <= 0) and (delta_row >= -1):
            for j in range(i, len(row_vals) -1):
                delta_row_j = row_vals[j+1] - row_vals[j]
                if delta_row_j > 0:
                    lines.append(j-1)
                    i = j+1
                    break
                else:
                    i = len(row_vals)
        else:
            i += 1    
    return lines

def find_trough_sharp_v2(img_mx):
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    i = 0
    while i < len(delta_row)-1:
        if (delta_row[i] < 0 ) and (delta_row[i+1] >= 0):
            lines.append(i)
            i += 1
        else:
            i += 1
    return lines

def find_trough_sharp_v3(img_mx):
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    i = 0
    while i < len(delta_row)-1:
        if (delta_row[i] < 0) and (delta_row[i] > -1) and (delta_row[i+1] < 1) and (delta_row[i+1] > 0) :
            lines.append(i)
            i += 1
        else:
            i += 1
    return lines

def find_trough_smooth(img_mx, n=5):
    '''Improvement
    You can easily eliminate sign function. its unnecessary.
    '''
    lines = []
    row_vals = img_mx.mean(axis=1)

    # Creating derivative list
    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])

    i = 0
    while i < len(delta_row) -n:
        if (delta_row[i] <0) and (delta_row[i] >-1):
            switch = False
            
            ## Front loop
            for k in range(i+1,i+n+1):
                if delta_row[k] <= 0: # if sign i+n and i are same
                    switch = True
                    if k == i+1:
                        i += 1 
                    else:
                        i = k +n # start from k+n since we know k is positive but k-1 is negative, and Back-loop will fail until k
                    break
            if switch: # Turn back to while loop with i = k+n
                continue

            ## Back Loop
            for j in range(1,n+1):
                if delta_row[i-j] > 0: # if sign i-j and i are not same
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

def find_trough_dent(img_mx, n=15):
    lines = []
    row_vals = img_mx.mean(axis=1)

    delta_row = []
    for i in range(len(row_vals) -1):
        delta_row.append(row_vals[i+1] - row_vals[i])
    
    i = 0
    while i< len(delta_row)-n:
        if (delta_row[i] <0) and (delta_row[i+1] >0):
            left_mean = np.mean(delta_row[i-n:i+1])
            right_mean = np.mean(delta_row[i+1:i+n+1])
            if (left_mean <0) and (right_mean >0):
                lines.append(i)
            i += 1
        else:
            i +=1
    return lines


## Eliminate close lines 
def line_elimenator_v2(lines):
    ### getting rid of duplicate lines
    lines.sort()
    lines = list(dict.fromkeys(lines))

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

# Visualization 
## Mid line drawer
def show_line(img, lines):
    img_viz = deepcopy(img)
    for line in lines:
        img_viz[line,:] = 255
    img_viz = Image.fromarray(img_viz)
    img_viz.show()

## Show any matrix
def show_matrix(img):
    img_viz = Image.fromarray(img)
    img_viz.show()

## Shows the bar with the approx. lenght drawn
def draw_bar(img,bar):
    drawn_img = deepcopy(img)
    img_height, img_width = np.shape(drawn_img)
    img_center = round(img_width/2)
    if img_height < bar:
        difference = bar - img_height
        buffer_zone = np.zeros((difference,img_width))
        drawn_img = np.vstack((drawn_img,buffer_zone))
    drawn_img[0 : bar , img_center-5 : img_center+5] = 255
    drawn_img = Image.fromarray(drawn_img)
    drawn_img.show()

def find_bar_dist(img,height=[1/2,1],width=[0,1/2], bar_width = 15): # read all the bars top point
    row_len, col_len = np.shape(img)
    u = col_len*height[0]
    d = col_len*height[1]
    l = row_len*width[0]
    r = row_len*width[1]
    mean_col_vals = np.mean(img[u:d,l:r], axis=0)
    '''if it passes the threshold for quite a while then take it as left,
    if it goes under the threshold for quite a while then take it as right'''
    list_left = []
    list_right = []
    i = 0
    switch = False
    while (i < (row_len - bar_width)) and (not(switch)):
        col_l = mean_col_vals[i]

        if col_l > threshold:
            switch = True
            for j in range(1 , bar_width):
                col_test = mean_col_vals[i + j]
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

    ### Righthand of the bar
    '''Summary
    Starting from the bar width, where we left off, so we do not need to recalculate the whole process
    '''
    for j in range(left_col_index + bar_width, column_len): 
        col_test = img_mean_col[j]

        if (col_test < threshold):
            right_col_index = j-1
            break

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
        if (error_function == 'linear'):
            reliability = 1- mean_error
            
        #elif (error_function == 'gauss') or (error_function == 'tanh') or (error_function == 'erf'):
        #    reliability = 1-erf(mean_error)
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
        if (error_function == 'linear'):
            reliability = 1- mean_error
        #elif (error_function == 'gauss') or (error_function == 'tanh') or (error_function == 'erf'):
        #    reliability = 1-erf(mean_error)
        
        # Add reliability to the list
        validity_list.append(reliability)
    
    return validity_list

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
