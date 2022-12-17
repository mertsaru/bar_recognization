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

class spep:

    bar_distance ={}

    def __init__(self, name):
        self.name = name
        self.is_ref = False
        self.is_mask = False

    # Creating matrix from the image
    def read(self, edge_multiplier = 500, amplifier = 1.3):
        '''Summary: edge multiplier
        Edge multiplier makes contrast with background and the lines. It makes easier to detect the lines.
        Edge multiplier should not be great, otherwise it creates noise near all the lines.
        '''

        ## Open Img
        img = Image.open(self.name)

        ## Turning into [n:m:3] array
        self.img_color = np.array(img)

        ## Turn into grayscale and then a matrix to save in self
        img_gray = img.convert('L')
        self.img_orj = np.array(img_gray)
        
        ## Making edge more visible and saving as an RGB image and turning into grayscale matrix
        edge_mx = filters.sobel(self.img_color)
        edge_img_rgb = Image.fromarray((edge_mx * edge_multiplier).astype(np.uint8),'RGB') 
        edge_img_gray = edge_img_rgb.convert('L') # convert library indicates we can transform while in matrix form. Maybe it would be faster? We can interfere in edge_mx *  multiplier.astype(uint8)
        self.img_subject = np.array(edge_img_gray)

        ## Saving threshold for distinguishing the background from the bar to find bottom of the bar
        img_col_mean = self.img_subject.mean(axis=0)
        self.threshold = np.median(img_col_mean) *amplifier # increasing threshold for eliminating black-space better 

        ## Taking the shape of the matrix for finding bars
        self.img_height, self.img_width = np.shape(self.img_subject)

    # Finding bars   

    ## Creating albumin mask
    @classmethod
    def create_albumin_mask(cls,self, bar_height=20, bar_width =15, buffer_top=0, buffer_side = -10):
        
        ## Finding the width of the albumin
        img_col_mean = self.img_subject.mean(axis=0)

        ### Lefthand of the bar
        i = 0
        switch = False
        while (i < (self.img_width - bar_width)) and (not(switch)):
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
        for j in range(left_col_index + bar_width, self.img_width): 
            col_test = img_col_mean[j]

            if (col_test < self.threshold):
                right_col_index = j-1
                break

        ## Finding the height of the albumin
        img_cutted_col_mx = self.img_subject[:,left_col_index:right_col_index] # Cutting left and right sides to find top of the albumin better
        img_row_mean = img_cutted_col_mx.mean(axis=1)

        ### Top of the albumin
        '''Summary
        We are using same threshold we used to find width of the bar, since threshold is a good number to divide black background and white lines.
        '''
        i = 0
        switch = False
        while (i < (self.img_height - bar_height)) and (not(switch)):
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
        for j in range(top_row_index + bar_height, self.img_height): 
            row_test = img_row_mean[j]

            if (row_test < self.threshold):
                bottom_row_index = j
                break
        try:
            cls.cutout ={
                'height': bottom_row_index - top_row_index +buffer_top,
                'width': right_col_index - left_col_index +buffer_side
            }
            cls.albumin_finder_name = self.name
            return show(self.img_color[top_row_index:bottom_row_index,left_col_index:right_col_index,:])
        except:
            print('Couldn\'t find the mask! Try a different image.' )

    ## Finding bar lenght
    @classmethod
    def bar_lenght_finder(cls,self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25], min_bar_height = 10):
        
        if hasattr(self,'test'):
            pass
        else:
            self.locate_lru(step, search_width, search_lenght)
        
        img_mx = deepcopy(self.img_subject)
        img_cut = img_mx[:,self.test['left'] : self.test['right']]
        img_mean_row_left = img_cut[:,0:round(self.cutout['width']/2)].mean(axis=1)
        img_mean_row_right = img_cut[:,round(self.cutout['width']/2):-1].mean(axis=1)

        i = 0
        switch = False
        while (i < self.img_height) and (not(switch)):
            left_row = img_mean_row_left[self.img_height -1 -i] # bottom i^th row's left half
            right_row = img_mean_row_right[self.img_height -1 -i] # bottom i^th row's right half
            if (left_row > self.threshold) and (right_row > self.threshold):
                switch = True
                for j in range(1 , min_bar_height):
                    left_row_test = img_mean_row_left[self.img_height -1 -i -j] # j above row of row_b
                    right_row_test = img_mean_row_right[self.img_height -1 -i -j]
                    if (left_row_test < self.threshold) or (right_row_test < self.threshold):
                        skip_step = j
                        i += skip_step +1
                        switch = False
                        break
            else:
                i += 1
        else:
            if switch:
                bottom_row_index = self.img_height -1  -i
            
        try:
            cls.bar = bottom_row_index - self.test['top']
            cls.bar_lenght_finder_name = self.name
            self.test_color = self.img_color[self.test['top'] : self.test['top'] + self.bar , self.test['left'] : self.test['right'],:]

            return show(self.test_color)
        except:
            print('the lenght of the bar couldn\'t be found')
    
    ## Finding positions

    ### bar distance finder
    def find_bar_dist(self, left_bar, search_width):
        '''Uses mask to find bars'''
        starting_point = left_bar['right']
        ending_point = starting_point +(search_width*self.cutout['width'])
        if starting_point >= 0:
            i = starting_point
            folder_max = 0
            while i < min(ending_point, self.img_width - self.cutout['width']):
                cutout = self.img_subject[self.test['top'] : self.test['top'] +self.bar , i:i+self.cutout['width']]
                folder_value = cutout.sum()
                if folder_value > folder_max:
                    folder_max = folder_value
                    next_bar_left = i
                i += 3
            
            distance = next_bar_left - left_bar['left']
            return distance
    
    ### Test
    def locate_test(self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25]):
       
        if hasattr(self, 'test'):
            pass
        else:
            self.locate_lru(step, search_width, search_lenght)

        self.test['bottom'] = self.test['top'] + self.bar
        ## Open the folder
        ### If image size is not enough
        if (self.img_height - self.test['top']) < (self.bar): 
            difference = self.bar - (self.img_height - self.test['top'])
            buffer_zone = np.zeros((difference,self.img_width))
            buffer_zone_color =np.array([[[255 for rgb in range(3)]for row in range(difference)]for column in range(self.img_width)])
            self.img_orj = np.vstack((self.img_orj,buffer_zone))
            self.img_subject = np.vstack((self.img_subject,buffer_zone))
            self.img_color = np.vstack((self.img_color,buffer_zone_color))
        
        self.test_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.test['left'] : self.test['right']] 
        
        self.test_color = self.img_color[self.test['top'] : self.test['bottom'], self.test['left'] : self.test['right'],:] 

        self.test_subject = self.img_subject[self.test['top'] : self.test['bottom'], self.test['left'] : self.test['right']] 
        
        return show(self.test_color)
    
    ### Finds the distance between two bars next to each other
    ### Gamma
    @classmethod
    def locate_gamma(cls,self, search_width=1):
        
        if hasattr(self,'test'):
            pass
        else:
            self.locate_test()

        distance = self.find_bar_dist(self.test, search_width)
        left = self.test['left'] + distance
        right = left + self.cutout['width']
        self.gamma = {
            'left': left,
            'right': right
        }
        cls.bar_distance['gamma'] = distance

    ### Alpha
    @classmethod
    def locate_alpha(cls,self, search_width=1):

        if hasattr(self,'gamma'):
            pass
        else:
            cls.locate_gamma(self)

        distance = self.find_bar_dist(self.gamma, search_width)
        left = self.gamma['left'] + distance
        right = left + self.cutout['width']
        self.alpha = {
            'left': left,
            'right': right
        } 
        cls.bar_distance['alpha'] = distance

    ### Mu 
    @classmethod
    def locate_mu(cls,self, search_width=1):

        if hasattr(self,'alpha'):
            pass
        else:
            cls.locate_alpha(self)

        distance = self.find_bar_dist(self.alpha, search_width)
        left = self.alpha['left'] + distance
        right = left + self.cutout['width']
        self.mu = {
            'left': left,
            'right': right
        } 
        cls.bar_distance['mu'] = distance
  
    ### Kappa
    @classmethod
    def locate_kappa(cls,self, search_width=1):
      
        if hasattr(self,'mu'):
            pass
        else:
            cls.locate_mu(self)

        distance = self.find_bar_dist(self.mu, search_width)
        left = self.mu['left'] + distance
        right = left + self.cutout['width']
        self.kappa = {
            'left': left,
            'right': right
        } 
        cls.bar_distance['kappa'] = distance

    ### Lambda
    @classmethod
    def locate_lambda(cls,self, search_width=1):

        if hasattr(self,'kappa'):
            pass
        else:
            cls.locate_kappa(self)

        distance = self.find_bar_dist(self.kappa, search_width)
        left = self.kappa['left'] +distance
        right = left + self.cutout['width']
        self.Lambda= {
            'left': left,
            'right': right
        }
        cls.bar_distance['lambda'] = distance

    ### All
    @classmethod
    def bar_finder(cls,self):
        try:

            cls.bar_finder_name = self.name
            cls.locate_lambda(self)
        except:
            print('Invalid input!')

    ## Using one reference to create bar finder
    def reference(self):
        self.create_albumin_mask(self)
        self.bar_lenght_finder(self)
        self.bar_finder(self)

    ## locates left right and top of the test bar
    def locate_lru(self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25]):
    
        width_cut = round(self.img_width *search_width[1])
        column_cut = round(self.img_height *search_lenght[1])

        folder_max = 0
        i = round(self.img_width*search_width[0])
        while i<width_cut:
            j = round(self.img_height*search_lenght[0])
            while j< column_cut:
                cutout = self.img_subject[i: i+self.cutout['height'], j: j+self.cutout['width']]
                folder_value = cutout.sum()
                if folder_value > folder_max:
                    folder_max = folder_value
                    top_row_index = i
                    left_col_index = j
                    j += step
                else:
                    j += step
            else:
                i += step
        
        right_col_index = left_col_index + self.cutout['width']
    
        self.test ={
            'left': left_col_index,
            'right': right_col_index,
            'top': top_row_index
        }

    ## locates all the bars in an image 
    def locate_bars(self):

        if hasattr(self,"test['bottom']"):
            pass
        else:
            self.locate_test()


        self.gamma = {
            'left': self.test['left'] + self.bar_distance['gamma'],
            'right': self.test['right'] + self.bar_distance['gamma']
        }

        self.alpha = {
            'left': self.gamma['left'] + self.bar_distance['alpha'],
            'right': self.gamma['right'] + self.bar_distance['alpha']
        }
        
        self.mu = {
            'left': self.alpha['left'] + self.bar_distance['mu'],
            'right': self.alpha['right'] + self.bar_distance['mu']
        }

        self.kappa = {
            'left': self.mu['left'] + self.bar_distance['kappa'],
            'right': self.mu['right'] + self.bar_distance['kappa']
        }

        self.Lambda = {
            'left': self.kappa['left'] + self.bar_distance['lambda'],
            'right': self.kappa['right'] + self.bar_distance['lambda']
        }


        self.gamma_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.gamma['left'] : self.gamma['right']] 
        self.gamma_color = self.img_color[self.test['top'] : self.test['bottom'], self.gamma['left'] : self.gamma['right'],:]
        
        self.alpha_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.alpha['left'] : self.alpha['right']] 
        self.alpha_color = self.img_color[self.test['top'] : self.test['bottom'], self.alpha['left'] : self.alpha['right'],:]
        
        self.mu_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.mu['left'] : self.mu['right']] 
        self.mu_color = self.img_color[self.test['top'] : self.test['bottom'], self.mu['left'] : self.mu['right'],:]
        
        self.kappa_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.kappa['left'] : self.kappa['right']] 
        self.kappa_color = self.img_color[self.test['top'] : self.test['bottom'], self.kappa['left'] : self.kappa['right'],:]
        
        self.lambda_orj = self.img_orj[self.test['top'] : self.test['bottom'], self.Lambda['left'] : self.Lambda['right']] 
        self.lambda_color = self.img_color[self.test['top'] : self.test['bottom'], self.Lambda['left'] : self.Lambda['right'],:]
        
    # For bar analysis
    ## Creating derivative list for searching peaks with find_peak_* functions
    '''Automaticly called in the class'''
    #? why it is working on img_subject
    def derivative(self):
        row_vals = self.img_subject.mean(axis=1)
        delta_row = []
        for i in range(len(row_vals) -1):
            delta_row.append(row_vals[i+1] - row_vals[i])
        self.delta_row = delta_row

    ## Finding the middle point of the lines by searching peaks of the bar
    # ! need to transform to obj.var
    '''All find_bars are called automatically in find_lines'''
    def find_bars_sharp(self):
        lines = []
        i=0
        while i < (len(self.img_height_vals_alter) -1):
            if (self.delta_row[i] >= 0) and (self.delta_row[i] <= 1):
                for j in range(i, len(self.img_height_vals_alter) -1):
                    if self.delta_row[j] < 0:
                        self.lines.append(j-1)
                        lines.append(j-1)
                        i = j+1
                        break
                    else:
                        i = len(self.img_height_vals_alter)
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

    # ! need to be written
    def find_lines(self,function = 'exclusive', range = 10):
        self.lines = []

        if function == 'exclusive':
            pass
        elif function == 'inclusive':
            pass
        
        try:
            self.line_cleaner(range)
        except:
            print('Wrong Input!')

    ### Eliminate close lines 
    '''Called automatically in find_lines'''
    def line_cleaner(self, range =10):
        '''Idea:
        I would suggest to find a static-dynamic range. depends on the bar lenght. like 1/5th of the bar or smthg'''
        ### getting rid of duplicate lines
        lines = self.lines
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
                if (next_line - nhood_lines[-1]) < range:
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

            self.lines = new_lines

    ## Returns distances of the lines
    # ! hasnt transformed to obj, check if it is working
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
            print('The number of lines is not 5')

    ## defines the bars reliability of position
    #! not working
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

    # Visualization
    ## Shows the bar with the approx. lenght drawn
    # ! Will read the bars and create a cls/var graph, which gives the graphs
    def bar_value(self):
        pass

    @staticmethod
    def show(matrix):
        return show(matrix)

    def img(self, bar, type = 'color'):
        try:
            if type == 'color':
                if bar == 'img':
                    return show(self.img_color)
                elif bar == 'test':
                    return show(self.test_color)
                elif bar == 'alpha':
                    return show(self.alpha_color)
                elif bar == 'gamma':
                    return show(self.gamma_color)
                elif bar == 'mu':
                    return show(self.mu_color)
                elif bar == 'kappa':
                    return show(self.kappa_color)
                elif bar == 'lambda':
                    return show(self.lambda_color)
                
            elif type == 'gray':
                if bar == 'img':
                    return show(self.img_orj)
                elif bar == 'test':
                    return show(self.test_orj)
                elif bar == 'alpha':
                    return show(self.alpha_orj)
                elif bar == 'gamma':
                    return show(self.gamma_orj)
                elif bar == 'mu':
                    return show(self.mu_orj)
                elif bar == 'kappa':
                    return show(self.kappa_orj)
                elif bar == 'lambda':
                    return show(self.lambda_orj)
                
            elif type == 'subject':
                if bar == 'img':
                    return show(self.img_subject)
                elif bar == 'test':
                    return show(self.test_subject)
                elif bar == 'alpha':
                    return show(self.alpha_subject)
                elif bar == 'gamma':
                    return show(self.gamma_subject)
                elif bar == 'mu':
                    return show(self.mu_subject)
                elif bar == 'kappa':
                    return show(self.kappa_subject)
                elif bar == 'lambda':
                    return show(self.lambda_subject)
                
        except:
            print('Wrong Input!')

    def matrix(self, bar, type = 'subject'):
        try:
            if type == 'color':
                if bar == 'img':
                    return self.img_color
                elif bar == 'test':
                    return self.test_color
                elif bar == 'alpha':
                    return self.alpha_color
                elif bar == 'gamma':
                    return self.gamma_color
                elif bar == 'mu':
                    return self.mu_color
                elif bar == 'kappa':
                    return self.kappa_color
                elif bar == 'lambda':
                    return self.lambda_color
                
            elif type == 'gray':
                if bar == 'img':
                    return self.img_orj
                elif bar == 'test':
                    return self.test_orj
                elif bar == 'alpha':
                    return self.alpha_orj
                elif bar == 'gamma':
                    return self.gamma_orj
                elif bar == 'mu':
                    return self.mu_orj
                elif bar == 'kappa':
                    return self.kappa_orj
                elif bar == 'lambda':
                    return self.lambda_orj
                
            elif type == 'subject':
                if bar == 'img':
                    return self.img_subject
                elif bar == 'test':
                    return self.test_subject
                elif bar == 'alpha':
                    return self.alpha_subject
                elif bar == 'gamma':
                    return self.gamma_subject
                elif bar == 'mu':
                    return self.mu_subject
                elif bar == 'kappa':
                    return self.kappa_subject
                elif bar == 'lambda':
                    return self.lambda_subject
                
        except:
            print('Wrong Input!')

    # ! need to be reworked. check if everything is okay or not
    def draw_lines(self,img = 'orj'):
        if img == 'orj': # orj grayscale fullscale 
            img_viz = deepcopy(self.img_orj)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right] = 255
            return show(img_viz)
        
        elif img == 'cut': # orj grayscale spep part
            img_viz = deepcopy(self.test_orj)
            for line in self.lines:
                img_viz[line,:] = 255
            return show(img_viz)
        
        elif img == 'color': # orj color fullscale
            img_viz = deepcopy(self.img_color)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right,:] = 0
            return show(img_viz)
        
        elif img == 'test': # reader img spep part
            img_viz = deepcopy(self.test_subject)
            for line in self.lines:
                img_viz[line,:] = 255
            return show(img_viz)
        
        else:
            print('!Wrong input in draw_lines!')

# Using show class to visualize
class show:

    def __init__(self,matrix):
        self.matrix = matrix
    
    def show(self):
        if len(np.shape(self.matrix))==2: # grayscale
            img_viz = Image.fromarray(self.matrix)
            img_viz.show(self.matrix)
        elif len(np.shape(self.matrix))==3: # RGB
            img_viz = Image.fromarray(self.matrix,'RGB')
            img_viz.show()
        else:
            print("Image is not readable")
