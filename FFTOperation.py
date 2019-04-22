
from numpy.fft import fft, fftfreq, ifft
import numpy as np
def fft_with_filter(data, low_pass=30000, high_pass = 50000, fs= 1e6):
	signal_set = []

	for index, row in data.iterrows():
		fft_data = np.fft.fft(row, n=row.size)/row.size
		freq = fftfreq(fft_data.size, d=1/fs)

		cut_low_signal = fft_data.copy()
		cut_low_signal[(freq<30000)] = 0

		signal = ifft(cut_low_signal)

		cut_high_signal = cut_low_signal.copy()
		cut_high_signal[(freq>50000)] = 0

		signal = ifft(cut_high_signal)

		# abs_signal = []
		# for n in np.abs(cut_high_signal):
		# 	if n != 0:
		# 		abs_signal.append([abs(n)])
		signal_set.append(np.abs(cut_high_signal))
	return signal_set
