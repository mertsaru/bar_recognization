from PIL import Image
import os

from Functions import *

directory = 'C:\Users\merts\Desktop\Bar Approx\dataset_1\A\A-kappa'

#Lists for distances collected from all the samples in the loop
jpeg_dict = {"bar_length" : [], "reliability": {}, "distances": {}}
count = 0
for file in os.listdir(directory):
    '''Runs the whole jpg files in folder'''
    file_dir = os.path.join(directory, file)
    if file_dir.endswith('.jpg'):
        
        # Read the image
        img = Image.open(file_dir)

        # Turn image into [wxh] matrix form
        img_mx = to_matrix(img)

        # Cuts the bar part only and viewing
        edges = find_edge(img_mx)
        img_cut = img_mx[edges[0]:edges[1], edges[2]:edges[3]]

        #img_cut_jpg = Image.fromarray(img_cut)
        #img_cut_jpg.show()
        #img_cut_rgb = img(edges)
        #img_cut_rgb.show()



        # Returns a list of mid points of albumin, alpha-1, alpha-2, beta, and gamma
        '''!Idea
        If it is + and near 0 or 1 and transitioning to - it is a middle point.
        The sign needs to repeat couple of times

        This function didnt work
        try to do sign differenatiation, but sign should repeat itself couple of times.
        Which signs to take? every 2nd or positive to negative ones?

        Maybe after reading +0 we can seek -0 and take the mid point
        '''

        lines_sharp, lines_smooth, lines_dent= [], [], []
        #lines_sharp = find_bars_sharp(img_cut)
        lines_smooth = find_bars_smooth(img_cut)
        lines_dent = find_bars_dent(img_cut)
        #print(f"line sharp :{lines_sharp}\n line smooth :{lines_smooth} \n line dent :{lines_dent}")

        #add lists
        lines = lines_sharp + lines_smooth + lines_dent 
        lines.sort()
        lines = list(dict.fromkeys(lines))
        #print(f'combined lines: {lines}\n')

        #Eliminate close lines
        new_lines = line_elimenator_v2(lines)
        #print(f'refined lines :  {new_lines}')

        # Draw lines into the jpeg
        #show_line(img_cut, new_lines)

        #Line distance
        '''Improvement!
        We are only taking the files which we found 5 lines on it.
        Need a better line validator'''
        if len(new_lines)==5:
            print('--------------------------\n'+file+'\n--------------------------\n')
            count += 1

            # Adding bar lenght to output
            bar_length = edges[1] - edges[0]
            jpeg_dict['bar_length'].append(bar_length)
            print(bar_length)
            
            '''Error:
            This should be for line picking. It should not be added to print. needs to eliminate lines'''
            # reliability of each line
            reliability = bar_reliability(img_cut, new_lines)          
            for key,value in reliability.items():
                if f'{key}_reliability' in jpeg_dict["reliability"][key]:
                    jpeg_dict["reliability"][key].append(value)
                else:
                    jpeg_dict["reliability"][key] = []
                    jpeg_dict["reliability"][key].append(value)
                print(f'{key} reliability : {value}')
                    
            # Distances of the lines    
            line_distance = line_dist(new_lines)
            for key,value in line_distance.items():
                if key in jpeg_dict['distances']:
                    jpeg_dict['distances'][key].append(value)
                else:
                    jpeg_dict['distances'][key] = []
                    jpeg_dict['distances'][key].append(value)
                print(f'{key} : {value}')

#Overall                        
print('----------------------------------')                
print(f'#Accepted files: {count}')
for value in jpeg_dict['bar_length']:
    print('----------------------------------\nbar lenght:')
    print(f'mean: {round(np.mean(value),2)} \nstd: {round(np.std(value),2)}')
print('\n')
for key, value in jpeg_dict['distances'].items():
    print(f'----------------------------------\n{key}:')
    print(f'mean: {round(np.mean(value),2)} \nstd: {round(np.std(value),2)}')
print('\n')
for key, value in jpeg_dict['reliability'].items():
    print(f'----------------------------------\n{key}:')
    print(f'mean: {round(np.mean(value),2)} \nstd: {round(np.std(value),2)}')
print('\n')