from PIL import Image
import os

from Functions import *

directory = 'dataset_1'
jpeg_dict = {"bar_length" : [], "reliability": {}, "distances": {}, "distances %":{}, "mid%":{}, "ratio":[], 'ratio2':[]}
count = 0
count_all = 0
# inside Main directory
for AGM in os.listdir(directory):
    dirAGM = os.path.join(directory, AGM)
    
    # inside A, G, M directories
    for kappa_lambda in os.listdir(dirAGM):
        dirKL = os.path.join(dirAGM, kappa_lambda)

        # inside Kappa, Lambda directiories, the files
        for file in os.listdir(dirKL):
            file_dir = os.path.join(dirKL, file)
            
            #Runs the whole jpg files in folder
            if file_dir.endswith('.jpg'):
                count_all +=1 # number of jpeg files we are running

                # Read the image
                img = Image.open(file_dir)

                # Turn image into [wxh] matrix form with alterating the density
                img_mx = to_matrix(img)

                # Cuts the SPEP part
                edges = find_edge(img_mx)
                img_cut = img_mx[edges[0]:edges[1], edges[2]:edges[3]]


                # Finds spikes
                lines_sharp, lines_smooth, lines_dent= [], [], []
                #lines_sharp = find_bars_sharp_v3(img_cut)
                lines_smooth = find_bars_smooth(img_cut)
                lines_dent = find_bars_dent(img_cut)

                # Line Organizer
                ## Combine findings
                lines = lines_sharp + lines_smooth + lines_dent 
                lines.sort()
                lines = list(dict.fromkeys(lines))

                ## Eliminate close lines
                new_lines = line_elimenator_v2(lines)

                # Eliminate unfit data
                '''Improvement!
                We are only taking the files which we found 5 lines on it.
                Need a better line validator'''
                '''Broken Code: reliability
                
                # Original Cut, for reliability use
                img_grayscale = img.convert('L')
                img_orj = np.array(img_grayscale)
                img_orj_cut = img_orj[edges[0]:edges[1], edges[2]:edges[3]]

                reliability = bar_reliability(img_orj_cut, new_lines)          
                for key,value in reliability.items():
                    if f'{key}_reliability' in jpeg_dict["reliability"][key]:
                        jpeg_dict["reliability"][key].append(value)
                    else:
                        jpeg_dict["reliability"][key] = []
                        jpeg_dict["reliability"][key].append(value)
                    print(f'{key} reliability : {value}')
                '''
                
                #Line distance
                if len(new_lines)==5:
                    print('--------------------------\n'+file+'\n--------------------------')
                    count += 1

                    # Adding bar lenght to output
                    bar_length = edges[1] - edges[0]
                    jpeg_dict['bar_length'].append(bar_length)
                    print(f'bar length: {bar_length}\n')
                    
                    # which percentage stays the mid points
                    obj = {
                        'albumin mid%': 100*new_lines[0]/bar_length,
                        'alpha_1 mid%': 100*new_lines[1]/bar_length,
                        'alpha_2 mid%': 100*new_lines[2]/bar_length,
                        'beta mid%': 100*new_lines[3]/bar_length,
                        'gamma mid%': 100*new_lines[4]/bar_length
                        }
                    for key, value in obj.items():
                        if key in jpeg_dict['mid%']:
                            jpeg_dict['mid%'][key].append(value)
                        else:
                            jpeg_dict['mid%'][key] =[]
                            jpeg_dict['mid%'][key].append(value)
                    for key,value in obj.items():
                        print(f'{key}: {round(value,2)}')
                    print('\n')

                    line_distance = line_dist(new_lines)       
                    # Distances of the lines    
                    for key,value in line_distance.items():
                        if key in jpeg_dict['distances']:
                            jpeg_dict['distances'][key].append(value)
                        else:
                            jpeg_dict['distances'][key] = []
                            jpeg_dict['distances'][key].append(value)
                        print(f'{key} : {value}')
                    print('\n')

                    # albumin-alpha1 : alpha1-alpha2
                    jpeg_dict['ratio'].append(line_distance["albumin - alpha1"]/line_distance["alpha1 - alpha2"])
                    jpeg_dict['ratio2'].append(line_distance["albumin - alpha1"]/line_distance["albumin - alpha2"])

                    # % of distances on the SPEP
                    for key,value in line_distance.items():
                        if f'{key} %' in jpeg_dict['distances %']:
                            jpeg_dict['distances %'][f'{key} %'].append(100*value/bar_length)
                        else:
                            jpeg_dict['distances %'][f'{key} %'] = []
                            jpeg_dict['distances %'][f'{key} %'].append(100*value/bar_length)
                        print(f'{key} % : {round(100*value/bar_length,2)}')
                    print('\n')
# Overall                        
print('---------------------\n       Overall\n---------------------')                
print(f'   #Accepted files\n   {count}/{count_all} ~ {round(100*count/count_all,2)}%')

# Bar Lenght
print('---------------------\n       bar lenght')
print(f'mean: {round(np.mean(jpeg_dict["bar_length"]),2)} \nstd: {round(np.std(jpeg_dict["bar_length"]),2)}')

# Distances
for key, value in jpeg_dict['distances'].items():
    print(f'---------------------\n   {key}')
    print(f'mean: {round(np.mean(value),2)} \nstd: {round(np.std(value),2)}')
print('\n')

# Distances %, how many % of bar is the given distance 
for key, value in jpeg_dict['distances %'].items():
    print(f'---------------------\n {key}')
    print(f'mean: {round(np.mean(value),2)}% \nstd: {round(np.std(value),2)}%')
print('\n')

for key, value in jpeg_dict['mid%'].items():
    print(f'---------------------\n {key}')
    print(f'mean: {round(np.mean(value),2)}% \nstd: {round(np.std(value),2)}%')
print('\n')

print(f'albumin-alpha1 : alpha1-alpha2\n mean: {round(np.mean(jpeg_dict["ratio"]),2)}\n std: {round(np.std(jpeg_dict["ratio"]),2)}')
print(f'albumin-alpha1 : albumin-alpha2\n mean: {round(np.mean(jpeg_dict["ratio2"]),2)}\n std: {round(np.std(jpeg_dict["ratio2"]),2)}')
print('\n')