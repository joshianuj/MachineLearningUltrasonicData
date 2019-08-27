import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOISE_SIZE = 350
ECHO_SIZE = 350
THRESHOLD = 0.26
SAMPLE_RATE_HZ = 200
LOW_PASS_HZ = 65
ECHO_PEAK_LEFT = 150

from scipy import signal

def butter_lowpass(cutoff=LOW_PASS_HZ, nyq_freq=SAMPLE_RATE_HZ, order=5):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq=LOW_PASS_HZ, nyq_freq=SAMPLE_RATE_HZ, order=5):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def get_data_points(file_location):
    time_domain_data = pd.read_csv(file_location, skiprows=[0], header=None)
    time_domain_data = time_domain_data.iloc[:,9:]
    time_domain_data_after_low_pass = []
    for data in time_domain_data.values:
        mean = np.mean(data)
        # remote offset
        x = data - mean
        # use low pass
        y = butter_lowpass_filter(x, cutoff_frequency, SAMPLE_RATE_HZ/2)
        diff = np.array(x)-np.array(y)
        time_domain_data_after_low_pass.append(diff)
    return time_domain_data_after_low_pass

def check_data_points(file_name):
    data_points = get_data_points(file_name)
    fig = plt.figure(figsize=(20, 20))
    for index, data in enumerate(data_points):
        ax = fig.add_subplot(7,7,index+1)
        ax.plot(data)

def check_peak(data_points_without_noise, old_index_list = []):
    peak_value = np.array(data_points_without_noise).max()
    peak_index = np.array(data_points_without_noise).argmax()
    print(np.array(data_points_without_noise).max())
#     for i,b in enumerate(data_points_without_noise):
#         if i not in old_index_list:
#             if (abs(b)> abs(peak_value)):
#                 peak_value = abs(b)
#                 peak_index=i
    if peak_value < THRESHOLD:
        return None
    peak_index_value = peak_index - ECHO_PEAK_LEFT
    if peak_index_value > 0:
        return peak_index_value
    else: 
#         old_index_list.append(peak_index)
#         check_peak(data_points_without_noise, old_index_list)
        return None
                
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
