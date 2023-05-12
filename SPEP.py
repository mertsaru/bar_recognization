import numpy as np
from PIL import Image
from skimage import filters
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools

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

    
    # Creating matrix from the image
    def read(self, edge_multiplier = 500, amplifier = 1.3):
        
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
 
    ## Evaluate find bars
    def evaluate(self):
        if hasattr(self,'Lambda'):
            pass
        else:
            self.find_graphs()
    
    ## Creating albumin mask
    @classmethod
    def albumin_reference(cls,self, give_information = True, bar_height=20, bar_width =15, buffer_top=0, buffer_side = -10):
        if hasattr(self,'img_subject'):
            pass
        else:
            self.read()

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
                    if col_test < self.threshold:
                        skip_step = j
                        i += skip_step + 1 
                        switch = False
                        break  # Turn back to while loop with index starting after i+j
            else:
                i += 1
        else:
            if switch:
                left_col_index = i        
            elif give_information:
                print('No left part of the albumin found')

        ### Righthand of the bar
        for j in range(left_col_index + bar_width, self.img_width): 
            col_test = img_col_mean[j]

            if (col_test < self.threshold):
                right_col_index = j-1
                break

        ## Finding the height of the albumin
        img_cutted_col_mx = self.img_subject[:,left_col_index:right_col_index] # Cutting left and right sides to find top of the albumin better
        img_row_mean = img_cutted_col_mx.mean(axis=1)

        ### Top of the albumin
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
            elif give_information:
                print('No upper part of the bar found')

        ### Bottom of the albumin
        for j in range(top_row_index + bar_height, self.img_height): 
            row_test = img_row_mean[j]

            if (row_test < self.threshold):
                bottom_row_index = j
                break

        cls.cutout ={
            'height': bottom_row_index - top_row_index +buffer_top,
            'width': right_col_index - left_col_index +buffer_side,
            'mid': round((right_col_index - left_col_index +buffer_side)/2)           
        }
        cls.albumin_refname = self.name

        return self.img_color[top_row_index:bottom_row_index,left_col_index:right_col_index,:]
    
    ## Finding bar lenght
    @classmethod
    def barLenght_reference(cls,self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25], min_bar_height = 10):
        
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
            
        
        cls.bar = bottom_row_index - self.test['top']
        cls.bar_lenght_finder_name = self.name
        self.test_color = self.img_color[self.test['top'] : self.test['top'] + self.bar , self.test['left'] : self.test['right'],:]
        
    ## Finding positions

    ### Test
    def locate_test(self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25]):
        if hasattr(self,'bar'):
            pass
        else:
            print('first determine bar lenght by using spep.barLenght_reference')
            return
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
    
    ### Finds the distance between two bars next to each other
    def find_bar_dist(self, left_bar, search_width): # Uses mask to find bars
        
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
        
        cls.bar_finder_name = self.name
        cls.locate_lambda(self)

    ## Using one reference to create bar finder
    def reference(self):
        self.albumin_reference(self) # Finding albumin mask
        self.barLenght_reference(self) # Finding height of the bars
        self.bar_finder(self) # Finding distances of the bars

    ## locates left right and top of the test bar
    def locate_lru(self, step =3, search_width = [0, 0.25], search_lenght = [0, 0.25]):
    
        if hasattr(self,'cutout'):
            pass
        else:
            print('first determine albumin mask by using spep.albumin_reference')
            return
        if hasattr(self,'img_subject'):
            pass
        else:
            self.read()

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
        if hasattr(self,'bar_finder_name'):
            pass
        else:
            print('first determine bar distance by using spep.bar_finder')
            return
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

    # Bar values
    ## Bar graph
    def graph_creator(self,bar,range = 5):
        if bar == 'test':
            bar_value = deepcopy(self.test_orj)
        elif bar == 'gamma':
            bar_value = deepcopy(self.gamma_orj)
        elif bar == 'alpha':
            bar_value = deepcopy(self.alpha_orj)
        elif bar == 'mu':
            bar_value = deepcopy(self.mu_orj)
        elif bar == 'kappa':
            bar_value = deepcopy(self.kappa_orj)
        elif bar == 'lambda':
            bar_value = deepcopy(self.lambda_orj)

        bar_value = bar_value[:,self.cutout['mid'] - range: self.cutout['mid'] + range]
        graph_value = bar_value.mean(axis=1)
        
        return graph_value

    ## Bars' graph values
    def find_graphs(self):
        if hasattr(self,'Lambda'):
            pass
        else:
            self.locate_bars()

        self.graphs ={
        'test': 255 - self.graph_creator('test'),
        'gamma': 255 - self.graph_creator('gamma'),
        'alpha': 255 - self.graph_creator('alpha'),
        'mu': 255 - self.graph_creator('mu'),
        'kappa': 255 - self.graph_creator('kappa'),
        'lambda': 255 - self.graph_creator('lambda')
        }

    # Bar analysis
    ## Creating derivative list for searching peaks with find_peak_* functions
    #self.graphs['test']
    def derivative(self,img):
        
        if img == 'subject':
            row_vals = self.test_subject.mean(axis=1)
        elif img == 'grayscale':
            row_vals = self.graphs['test']    
        
        delta_row = []
        for i in range(len(row_vals) -1):
            delta_row.append(row_vals[i+1] - row_vals[i])
        self.delta_row = delta_row

    ## Finding the middle point of the lines by searching peaks of the bar
    def find_peak_sharp(self):
        i=0
        while i < (self.bar -1):
            if (self.delta_row[i] >= 0) and (self.delta_row[i] <= 1):
                for j in range(i, self.bar -1):
                    if self.delta_row[j] < 0:
                        self.lines.append(j-1)
                        i = j+1
                        break
                    else:
                        i = self.bar
            else:
                i += 1    

    def find_peak_sharp_v2(self):
        i = 0
        while i < len(self.delta_row)-1:
            if (self.delta_row[i] >= 0 ) and (self.delta_row[i+1] <= 0):
                self.lines.append(i)
                i += 1
            else:
                i += 1

    def find_peak_sharp_v3(self):
        i = 0
        while i < len(self.delta_row)-1:
            if (self.delta_row[i] > 0) and (self.delta_row[i] < 1) and (self.delta_row[i+1] > -1) and (self.delta_row[i+1] < 0) :
                self.lines.append(i)
                i += 1
            else:
                i += 1

    def find_peak_smooth(self, smoothRange):
        i = 0
        while i < len(self.delta_row) -smoothRange:
            if (self.delta_row[i] >0) and (self.delta_row[i] <1): # sign i >= 0
                switch = False
                
                ## Front loop
                for k in range(i+1,i+smoothRange+1):
                    if self.delta_row[k] >= 0: # if sign i+smoothRange and i are same
                        switch = True
                        if k == i+1:
                            i += 1 
                        else:
                            i = k +smoothRange # start from k+smoothRange since we know k is positive but k-1 is negative, and Back-loop will fail until k
                        break
                if switch: # Turn back to while loop with i = k+smoothRange
                    continue

                ## Back Loop
                for j in range(1,smoothRange+1):
                    if self.delta_row[i-j] < 0: # if sign i-j and i are not same
                        switch = True
                        i = i -j +smoothRange+1 # start from there since until that point everything will fail with back-loop because of i-jth point
                        break
                if switch: # Turns back to while loop with i = i-j+smoothRange+1
                    continue
                
                self.lines.append(i)
                i = i+smoothRange+1 # If we passed until here, we know i+1 until i+smoothRange are all negative so they will fail anyways
            else:
                i += 1
        
    def find_peak_dent(self, dentRange):
        i = dentRange
        while i< len(self.delta_row)-dentRange:
            if (self.delta_row[i] >0) and (self.delta_row[i+1] <0):
                left_mean = np.mean(self.delta_row[i-dentRange:i+1])
                right_mean = np.mean(self.delta_row[i+1:i+dentRange+1])
                if (left_mean >0) and (right_mean <0):
                    self.lines.append(i)
                i += 1
            else:
                i +=1

    ## finds lines
    def line_method(self,function, smoothRange = 5, dentRange = 10, combine_lines = True, increase_spec = True):
        
        if 'all' in function:
            function = ['sharp','sharp_v2','sharp_v3','smooth','dent']

        self.derivative('grayscale')
        self.lines = []

        if 'sharp' in function:
            self.find_peak_sharp()
        if 'sharp_v2' in function:
            self.find_peak_sharp_v2()
        if 'sharp_v3' in function:
            self.find_peak_sharp_v3()
        if 'smooth' in function:
            self.find_peak_smooth(smoothRange)
        if 'dent' in function:
            self.find_peak_dent(dentRange)
        
        if combine_lines:
            self.line_cleaner()

        if (increase_spec) and (len(self.lines) != 5):
            
            self.derivative('subject')

            self.lines = []

            if 'sharp' in function:
                self.find_peak_sharp()
            if 'sharp_v2' in function:
                self.find_peak_sharp_v2()
            if 'sharp_v3' in function:
                self.find_peak_sharp_v3()
            if 'smooth' in function:
                self.find_peak_smooth(smoothRange)
            if 'dent' in function:
                self.find_peak_dent(dentRange)
            
            if combine_lines:
                self.line_cleaner()

    def find_lines(self,function: list, smoothRange = 5, dentRange = 10, combine_lines = True, increase_spec = True, use_combinations = False):

        if 'all' in function:
            function = ['smooth','dent','sharp_v2','sharp_v3','sharp']

        if use_combinations:

            combinations = []
            for r in range(1 , len(function)+1):
                for combination in itertools.combinations(function,r):
                    combinations.append(combination)
            combinations = combinations[::-1]
            for combination in combinations:
                self.line_method(function = combination, smoothRange = smoothRange, dentRange = dentRange, combine_lines = combine_lines, increase_spec = increase_spec)
                if len(self.lines) == 5:
                    break
        
        else:
            self.line_method(function = function, smoothRange = smoothRange, dentRange = dentRange, combine_lines = combine_lines, increase_spec = increase_spec)

    ### Eliminate close lines 
    def line_cleaner(self, range =10):
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
    def line_dist(self, give_information = False):
        if len(self.lines)==5:
            self.dist_dict ={}
            for i in range(5):
                for j in range(i+1, 5):
                    d_ij = f'{line_names[i]} - {line_names[j]}'
                    distance = self.lines[j] - self.lines[i]
                    self.dist_dict[d_ij] = distance
            
        elif give_information:
            print('The number of lines is not 5')

    # Visualization
    ## Shows the bar with the approx. lenght drawn
    @staticmethod
    def show_img(matrix):
        if len(np.shape(matrix))==2: # grayscale
            img_viz = Image.fromarray(matrix)
            img_viz.show(matrix)
        elif len(np.shape(matrix))==3: # RGB
            img_viz = Image.fromarray(matrix,'RGB')
            img_viz.show()
        else:
            print("Matrix is not transformable")

    def show_graph(self,bar):
        if hasattr(self,'graphs'):
            pass
        else:
            self.find_graphs()

        bar_height = range(self.bar)

        if 'all' in bar:
            bar = ['test','gamma','alpha','mu','kappa','lambda']

        lenght = 0
        if 'test' in bar:
            lenght +=1
        if 'gamma' in bar:
            lenght +=1
        if 'alpha' in bar:
            lenght +=1
        if 'mu' in bar:
            lenght +=1
        if 'kappa' in bar:
            lenght +=1
        if 'lambda' in bar:
            lenght +=1
        
        plt.figure(figsize=(15,5))
        count = 1
        if 'test' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['test'],bar_height[::-1])
            plt.title(f'SP')
            plt.xticks(np.arange(0,255,50))
            plt.xlabel('value')
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        if 'gamma' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['gamma'],bar_height[::-1])
            plt.title(f'ɣ')
            plt.xlabel('value')
            plt.xticks(np.arange(0,255,50))
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        if 'alpha' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['alpha'],bar_height[::-1])
            plt.title(f'ɑ')
            plt.xlabel('value')
            plt.xticks(np.arange(0,255,50))
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        if 'mu' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['mu'],bar_height[::-1])
            plt.title(f'μ')
            plt.xlabel('value')
            plt.xticks(np.arange(0,255,50))
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        if 'kappa' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['kappa'],bar_height[::-1])
            plt.title(f'ĸ')
            plt.xlabel('value')
            plt.xticks(np.arange(0,255,50))
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        if 'lambda' in bar:
            plt.subplot(1,lenght,count)
            count +=1
            plt.plot(self.graphs['lambda'],bar_height[::-1])
            plt.title(f'λ')
            plt.xlabel('value')
            plt.xticks(np.arange(0,255,50))
            plt.axis([0,255,0,self.bar])
            plt.yticks([])

        plt.suptitle(self.name)
        plt.show()

    def draw_lines(self,img = 'orj'):
        if img == 'orj': # orj grayscale fullscale 
            img_viz = deepcopy(self.img_orj)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right] = 255
            return img_viz
        
        elif img == 'cut': # orj grayscale spep part
            img_viz = deepcopy(self.test_orj)
            for line in self.lines:
                img_viz[line,:] = 255
            return img_viz
        
        elif img == 'color': # orj color fullscale
            img_viz = deepcopy(self.img_color)
            for line in self.lines:
                img_viz[line+self.top,self.left:self.right,:] = 0
            return img_viz
        
        elif img == 'test': # reader img spep part
            img_viz = deepcopy(self.test_subject)
            for line in self.lines:
                img_viz[line,:] = 255
            return img_viz
        
        else:
            print('!Wrong input in draw_lines!')
