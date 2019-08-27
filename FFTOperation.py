
from numpy.fft import fft, fftfreq, ifft
import numpy as np


def fill_by_average(prev_data):
    final_data=[]
    for i,item in enumerate(prev_data):
        final_data.append(item)
        if i<len(prev_data)-1:
            final_data.append((item+prev_data[i+1])/2)
    return final_data


def fft_with_filter(data, low_pass=30000, high_pass = 50000, fs= 1e6):
	signal_set = []

	for index, row in data.iterrows():
		new_fs = fs
		fft_data = np.fft.fft(row, n=row.size)/row.size
		if (row.size<3000):
  			new_fs = fs * 2
		freq = fftfreq(fft_data.size, d=1/int(new_fs))

		cut_low_signal = fft_data.copy()
		cut_low_signal[(freq<int(low_pass))] = 0

		signal = ifft(cut_low_signal)

		cut_high_signal = cut_low_signal.copy()
		cut_high_signal[(freq>int(high_pass))] = 0

		signal = ifft(cut_high_signal)
		d = np.abs(cut_high_signal)
		d  = list(filter(lambda a: a != 0, d))

		while len(d) < 80:
			d = fill_by_average(d) 
		if len(d) == 81:
			d.insert(41,(d[0]+d[len(d)-1])/2)
		signal_set.append(d)

	return signal_set
