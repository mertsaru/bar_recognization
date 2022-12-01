import numpy as np
from PIL import Image
from skimage import filters

from parameters import *

class read():

    # Functions

    ## Changing image to matrix
    def to_matrix(self, edge_multiplier = 500):
        '''Summary: edge multiplier
        Edge multiplier makes contrast with background and the lines. It makes easier to detect the lines.
        Edge multiplier should not be great, otherwise it creates noise near all the lines.
        '''

        ## Open Img
        img = Image.open(self.name)

        ## Turning into [n:m:3] array
        img_mx = np.array(img)

        ## Turn into grayscale and then a matrix to save in self
        img_gray = img.convert('L')
        self.img_orj = np.array(img_gray)

        ## Making better contrast between borders and background
        edge_mx = filters.sobel(img_mx)

        ## Making edge more visible and saving as an RGB image
        edge_img_rgb = Image.fromarray((edge_mx * edge_multiplier).astype(np.uint8),'RGB') 

        ## Changing RGB img into grayscale image 
        edge_img_gray = edge_img_rgb.convert('L') # convert library indicates we can transform while in matrix form. Maybe it would be faster? We can interfere in edge_mx *  multiplier.astype(uint8)

        ## Turning img into [n:m:1] array, values between 0-255
        self.matrix = np.array(edge_img_gray) 
    
    ## Finding SPEP bar
    def find_SPEP(self, amplifier = 1.3):
        '''Summary
        First find left and right borders of the bar, then cut the bar with the given borders
        Then it would be easier to find top and bottom, since there would be less noise from the background.
        '''
        row_len, column_len = np.shape(self.matrix)

        ## Finding the width of the bar
        img_mean_col = self.matrix.mean(axis=0)

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
                self.left_col = i        
            else:
                raise Exception('No left part of the bar found')

        ### Righthand of the bar
        '''Summary
        Starting from the bar width, where we left off, so we do not need to recalculate the whole process
        '''
        for j in range(self.left_col + bar_width, column_len): 
            col_test = img_mean_col[j]

            if (col_test < threshold):
                self.right_col = j-1
                break

        ## Finding the height of the bar
        '''Summary
        Cut the img for better mean performance.
        The cutted image give us less black values on the rows, so it would be less background noise.
        '''
        img_cutted_col_mx = self.matrix[:,self.left_col:self.right_col]
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
                self.top_row = i
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
                self.bottom_row = row_len -1  -i
            else:
                raise Exception('No bottom part of the bar found')

    ## Creating derivative list
    def derivative(self):
        delta_row = []
        for i in range(len(self.row_vals_alter) -1):
            delta_row.append(self.row_vals_alter[i+1] - self.row_vals_alter[i])
        self.delta_row = delta_row

    ## Functions to find middle points of the lines
    def find_bars_sharp(self):
        lines = []
        i=0
        while i < (len(self.row_vals_alter) -1):
            if (self.delta_row[i] >= 0) and (self.delta_row[i] <= 1):
                for j in range(i, len(self.row_vals_alter) -1):
                    if self.delta_row[j] < 0:
                        self.lines.append(j-1)
                        lines.append(j-1)
                        i = j+1
                        break
                    else:
                        i = len(self.row_vals_alter)
            else:
                i += 1    

    def find_bars_sharp_v2(self):
        i = 0
        while i < len(self.delta_row)-1:
            if (self.delta_row[i] >= 0 ) and (self.delta_row[i+1] <= 0):
                self.lines.append(i)
                i += 1
            else:
                i += 1

    def find_bars_sharp_v3(self):
        i = 0
        while i < len(self.delta_row)-1:
            if (self.delta_row[i] > 0) and (self.delta_row[i] < 1) and (self.delta_row[i+1] > -1) and (self.delta_row[i+1] < 0) :
                self.lines.append(i)
                i += 1
            else:
                i += 1

    def find_bars_smooth(self, n=5):
        '''description n:
        n chooses how far we want the function monotonically increasing or decreasing.
        '''
        i = 0
        while i < len(self.delta_row) -n:
            if (self.delta_row[i] >0) and (self.delta_row[i] <1): # sign i >= 0
                switch = False
                
                ## Front loop
                for k in range(i+1,i+n+1):
                    if self.delta_row[k] >= 0: # if sign i+n and i are same
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
                    if self.delta_row[i-j] < 0: # if sign i-j and i are not same
                        switch = True
                        i = i -j +n+1 # start from there since until that point everything will fail with back-loop because of i-jth point
                        break
                if switch: # Turns back to while loop with i = i-j+n+1
                    continue
                
                self.lines.append(i)
                i = i+n+1 # If we passed until here, we know i+1 until i+n are all negative so they will fail anyways
            else:
                i += 1
        
    def find_bars_dent(self, n=10):
        i = n
        while i< len(self.delta_row)-n:
            if (self.delta_row[i] >0) and (self.delta_row[i+1] <0):
                left_mean = np.mean(self.delta_row[i-n:i+1])
                right_mean = np.mean(self.delta_row[i+1:i+n+1])
                if (left_mean >0) and (right_mean <0):
                    self.lines.append(i)
                i += 1
            else:
                i +=1

    def __init__(self, name, functions = []):
        self.name = name
        self.lines = []

        self.to_matrix()
        self.find_SPEP()

        self.img_alter_cut = self.matrix[self.top_row: self.bottom_row, self.left_col: self.right_col]
        self.img_orj_cut = self.img_orj[self.top_row: self.bottom_row, self.left_col: self.right_col]
        self.row_vals_alter = self.img_alter_cut.mean(axis = 1)
        self.derivative()

        if 'sharp' in functions:
            self.find_bars_sharp()
        if 'sharp_v2' in functions:    
            self.find_bars_sharp_v2()
        if 'sharp_v3' in functions:
            self.find_bars_sharp_v3()
        if 'smooth' in functions:
            self.find_bars_smooth()
        if 'dent' in functions:
            self.find_bars_dent()
    