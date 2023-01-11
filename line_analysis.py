from SPEP import spep
import os
import numpy as np

# Selecting mask and reference images
ref_name = 'datasets/dataset_1/A/A-kappa/Z_80080808_0230_IFIX1.jpg'
mask_name = 'datasets/dataset_1/A/A-kappa/Z_80090997_0230_IFIX1.jpg'

ref = spep(ref_name)
mask = spep(mask_name)

spep.albumin_reference(mask)
spep.barLenght_reference(ref)
spep.bar_finder(ref)

directory = 'datasets'
count_all = 0
count_valid = 0
distance_ratio = {}
for root, dir, files in os.walk(directory):
    for file in files:
        file_dir = os.path.join(root,file)

        if file_dir.endswith('.jpg'):
            count_all += 1

            obj = spep(file_dir)

            obj.evaluate()
            obj.find_lines('dent','smooth',smoothRange= 5, dentRange= 10)
            
            if len(obj.lines) == 5:
                count_valid += 1
                obj.line_dist()
                for key1,value1 in obj.dist_dict.items():
                    for key2,value2 in obj.dist_dict.items():
                        if key1 == key2:
                            pass
                        else:
                            if f'{key1}:{key2}' in distance_ratio:
                                distance_ratio[f'{key1}:{key2}'].append(value1/value2)
                            else:
                                distance_ratio[f'{key1}:{key2}'] =[]
                                distance_ratio[f'{key1}:{key2}'].append(value1/value2)

# Writing the findings
print('valid images: ',round(100*count_valid/count_all,2),'%')
#for key,value in distance_ratio.items():
    #print('--------------------\n''mean ', key,' = ', np.mean(value),'\n')
    #print('std ', key,' = ', np.std(value),'\n--------------------\n')