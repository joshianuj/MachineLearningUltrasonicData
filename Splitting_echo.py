import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOISE_SIZE = 350
ECHO_SIZE = 350
THRESHOLD = 0.26
NOISE_END_LENGTH = 1500 #this is a hack
SAMPLE_RATE_HZ = 200
LOW_PASS_HZ = 65

def get_data_points(file_location):
    dataframe = pd.read_csv(file_location, skiprows=[0], header=None)
    return dataframe.iloc[:,9:]

def check_data_points(file_name):
    data_points = get_data_points(file_name)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 20))
    for index, data in enumerate(data_points.values):
        ax = fig.add_subplot(7,7,index+1)
        ax.plot(data)

def check_peak(data_points_without_noise, old_index_list = []):
    peak_value = 0
    peak_index = 0
    for i,b in enumerate(data_points_without_noise):
        if i not in old_index_list:
            if (abs(b)> abs(peak_value)):
                peak_value = abs(b)
                peak_index=i
    if peak_value < THRESHOLD:
        return None
    print(peak_value, peak_index, 'hello')
    peak_index_value = peak_index - 150
    if peak_index_value > 0:
        return peak_index_value
    else: 
        old_index_list.append(peak_index)
        check_peak(data_points_without_noise, old_index_list)
                
def seperate_echos(data_point):
    data_points_with_noise = data_point.tolist()
    #cut signal first noise length
    data_points_without_noise = data_points_with_noise[NOISE_SIZE:]
    data_points_without_noise = data_points_without_noise[:NOISE_END_LENGTH]
    #find peaks
#     peak_value = data_points_without_noise[0]
#     peak_index = 0
#     for i,b in enumerate(data_points_without_noise):
#         if (abs(b)> abs(peak_value)):
#             peak_value = b
#             peak_index=i
#     echo_data_points = data_points_without_noise
#     print(peak_index, peak_value)
#     peak_value = peak_index - 100
#     check_peak(p)
            
    peak_value = check_peak(data_points_without_noise)
    if peak_value == None:
        return None
    echo_data_points = data_points_without_noise[peak_value:]
    echo_data_points = echo_data_points[:ECHO_SIZE]
    return echo_data_points

def get_data_points_without_offset(data_points):
    data_points_without_offset = []
    for data in data_points.values:
        mean = np.mean(data)
        data_points_without_offset.append(data - mean)
    return data_points_without_offset
        
def plot_echos(file_name):
    data_points = get_data_points(file_name)
    data_points_without_offset = get_data_points_without_offset(data_points)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 20))
    echo_set = []
    for index, data in enumerate(data_points_without_offset):
        echo = seperate_echos(data)
        if echo:
            ax = fig.add_subplot(8,8,index+1)
            echo_set.append(echo)
            # ax.set_xlim([600,1500])
            ax.plot(echo)
    return echo_set

plot_echos('./data_set/car_new/Human_A/Human_80/1.csv')

# file_set = [1,2,3,4,5,6,7,8,9]
# folders = [180,190,200,210,220,230,240,250]
# # folders =[,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250]
# wall_data_set = []
# for folder in folders:
#     for distance in file_set:
#         file_name = './data_set/car_new/pillar/Pillar_{}/{}.csv'.format(folder, distance)
#         echo_set = plot_echos(file_name)
#         df = pd.DataFrame(echo_set)
#         df.to_csv('./data_set/car_new/pillar/Pillar_{}/echo/{}.csv'.format(folder, distance), index=False)