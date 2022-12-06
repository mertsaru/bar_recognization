import numpy as np
from PIL import Image
from skimage import filters
from copy import deepcopy

line_names ={
            0 : 'albumin',
            1 : 'alpha1',
            2 : 'alpha2',
            3 : 'beta',
            4 : 'gamma'
            }    

class open:

    def __init__(self, name):
        self.name = name

    # Functions

    ## Creating matrix from the image
    def read(self, edge_multiplier = 500, amplifier = 1.3):
        '''Summary: edge multiplier
        Edge multiplier makes contrast with background and the lines. It makes easier to detect the lines.
        Edge multiplier should not be great, otherwise it creates noise near all the lines.
        '''

        ## Open Img
        img = Image.open(self.name)

        ## Turning into [n:m:3] array
        self.matrix_color = np.array(img)

        ## Turn into grayscale and then a matrix to save in self
        img_gray = img.convert('L')
        self.matrix_orj = np.array(img_gray)
        
        ## Making edge more visible and saving as an RGB image and turning into grayscale matrix
        edge_mx = filters.sobel(self.matrix_color)
        edge_img_rgb = Image.fromarray((edge_mx * edge_multiplier).astype(np.uint8),'RGB') 
        edge_img_gray = edge_img_rgb.convert('L') # convert library indicates we can transform while in matrix form. Maybe it would be faster? We can interfere in edge_mx *  multiplier.astype(uint8)
        self.matrix_test = np.array(edge_img_gray)

        ## Saving threshold for distinguishing the background from the bar to find bottom of the bar
        img_col_mean = self.matrix_test.mean(axis=0)
        self.threshold = np.median(img_col_mean) *amplifier # increasing threshold for eliminating black-space better 

        ## Taking the shape of the matrix for finding bars
        self.row, self.col = np.shape(self.matrix_test)

    ## Creating albumin mask
    @classmethod
    def create_albumin_mask(cls,self, bar_height=20, bar_width =15, buffer = 5):

        ## Finding the width of the albumin
        img_col_mean = self.matrix_test.mean(axis=0)

        ### Lefthand of the bar
        i = 0
        switch = False
        while (i < (self.col - bar_width)) and (not(switch)):
            col_l = img_col_mean[i]

            if col_l > self.threshold:
                switch = True
                for j in range(1 , bar_width):
                    col_test = img_col_mean[i + j]
                    '''!Idea
                    Here we can also use derivative to find where the big change is, but not for the bottom
                    '''
                    if col_test < self.threshold:
                        '''Summary
                        skips since we know all of the col_l will fail until there.
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
                print('No left part of the albumin found')

        ### Righthand of the bar
        '''Summary
        Starting from the bar width, where we left off, so we do not need to recalculate the whole process
        '''
        for j in range(left_col_index + bar_width, self.col): 
            col_test = img_col_mean[j]

            if (col_test < self.threshold):
                right_col_index = j-1
                break

        ## Finding the height of the albumin
        img_cutted_col_mx = self.matrix_test[:,left_col_index:right_col_index] # Cutting left and right sides to find top of the albumin better
        img_row_mean = img_cutted_col_mx.mean(axis=1)

        ### Top of the albumin
        '''Summary
        We are using same threshold we used to find width of the bar, since threshold is a good number to divide black background and white lines.
        '''
        i = 0
        switch = False
        while (i < (self.row - bar_height)) and (not(switch)):
            row_u = img_row_mean[i]
            
            if row_u > self.threshold:
                switch = True
                for j in range(1 , bar_height):
                    row_u_test = img_row_mean[i + j]
                    
                    if row_u_test < self.threshold:
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
                print('No upper part of the bar found')

        ### Bottom of the albumin
        for j in range(top_row_index + bar_height, self.row): 
            row_test = img_row_mean[j]

            if (row_test < self.threshold):
                bottom_row_index = j
                break
        try:
            cls.cutout_h = bottom_row_index - top_row_index +buffer
            cls.cutout_w = right_col_index - left_col_index +buffer
            cls.mask = np.ones((cls.cutout_h,cls.cutout_w))
            
            return matrix(self.matrix_orj[top_row_index:bottom_row_index,left_col_index:right_col_index])
        except:
            print('Couldn\'t find the mask! Try a different image.' )

    ## Creating derivative list for searching peaks with find_peak_* functions
    def derivative(self):
        row_vals = self.matrix_test.mean(axis=1)
        delta_row = []
        for i in range(len(row_vals) -1):
            delta_row.append(row_vals[i+1] - row_vals[i])
        self.delta_row = delta_row

    def locate_spep(self,step=3):
    
        col_len_part = self.col/4
        row_len_part = self.row/4

        folder_max = 0
        i=0
        while i<col_len_part:
            j=0
            while j< row_len_part:
                cutout = self.matrix_test[i: i+self.cutout_h, j: j+self.cutout_w]
                folder = np.multiply(cutout,self.mask)
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
        
        right_col_index = left_col_index + self.cutout_w

        ## Open the folder

        ### Dividing left and right parts to check row values seperately, to avoid a mark on one of the sides.
        img_cut = self.matrix_test[:,left_col_index:right_col_index]
        img_mean_row_left = img_cut[:,0:round(self.cutout_w/2)].mean(axis=1)
        img_mean_row_right = img_cut[:,round(self.cutout_w/2):-1].mean(axis=1)

        i = 0
        switch = False
        while (i < self.row) and (not(switch)):
            left_row = img_mean_row_left[self.row -1 -i] # bottom i^th row's left half
            right_row = img_mean_row_right[self.row -1 -i] # bottom i^th row's right half
            if (left_row > self.threshold) and (right_row > self.threshold):
                switch = True
                for j in range(1 , bar_height):
                    left_row_test = img_mean_row_left[self.row -1 -i -j] # j above row of row_b
                    right_row_test = img_mean_row_right[self.row -1 -i -j]
                    if (left_row_test < self.threshold) or (right_row_test < self.threshold):
                        skip_step = j
                        i += skip_step +1
                        switch = False
                        break
            else:
                i += 1
        else:
            if switch:
                bottom_row_index = self.row -1  -i
            else:
                print('no bottom of the bar found')
        
        try:
            self.top, self.bottom, self.left, self.right = top_row_index, bottom_row_index, left_col_index, right_col_index
            self.matrix_cut = self.matrix_orj[self.top:self.bottom, self.left:self.right]
            self.matrix_test = self.matrix_test[self.top:self.bottom, self.left:self.right]
            self.derivative()

            return matrix(self.matrix_cut)
        except:
            print('Couldn\'t find the SPEP bar! File will be skipped!')

    ### Finding the middle point of the lines by searching peaks of the bar
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
        '''Description n:
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

                '''Explanation: Why range j is taken like this
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

    ## Eliminate close lines 
    def line_elimenator(lines):
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
        if len(lines)==5:
            dist_dict ={}
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    d_ij = f'{line_names[i]} - {line_names[j]}'
                    distance = lines[j]-lines[i]
                    dist_dict[d_ij] = distance
            
            return dist_dict
        else:
            print('The line number is not 5')


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


    ## Visualization
    def test(self):
        return matrix(self.matrix_test)

    def draw_lines(self,img = 'orj'):
        if img == 'orj': # orj grayscale fullscale 
            img_viz = deepcopy(self.matrix_orj)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right] = 255
            return matrix(img_viz)
        
        elif img == 'cut': # orj grayscale spep part
            img_viz = deepcopy(self.matrix_cut)
            for line in self.lines:
                img_viz[line,:] = 255
            return matrix(img_viz)
        
        elif img == 'color': # orj color fullscale
            img_viz = deepcopy(self.matrix_color)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right,:] = 0
            return matrix(img_viz)
        
        elif img == 'test': # reader img spep part
            img_viz = deepcopy(self.matrix_test)
            for line in self.lines:
                img_viz[line,:] = 255
            return matrix(img_viz)
        
        else:
            print('!Wrong input in draw_lines!')

# Using Matrix to visualize
class matrix:

    def __init__(self,matrix):
        self.matrix = matrix
    
    def show(self):
        img_viz = Image.fromarray(self.matrix)
        img_viz.show()