import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly
from numpy.fft import fft, fftfreq, ifft
df = pd.read_csv('./data/predict/Stopper/_80cm_AS_00008.csv', header=None)
print(df.values[0])
data = df.values[0]
# fft = np.fft.fft(data,n=data.size)/data.size
# print(fft)


fs= 1e6
fft_data = np.fft.fft(data, n=data.size)/data.size
freq = fftfreq(fft_data.size, d=1/fs)

cut_low_signal = fft_data.copy()
cut_low_signal[(freq<30000)] = 0

signal = ifft(cut_low_signal)

cut_high_signal = cut_low_signal.copy()
cut_high_signal[(freq>50000)] = 0

signal = ifft(cut_high_signal)
freq = fftfreq(signal.size, d=1/fs)
print(signal)
final_signal = []
for n in np.abs(cut_high_signal):
    if n != 0:
        final_signal.append([abs(n)])
fig, ax = plt.subplots(2, 1)

ax[0].plot(freq, np.abs(signal))
ax[0].set_xlabel('frequency(hz)')
ax[0].set_ylabel('absolute fft')

plot_url = plotly.offline.plot_mpl(fig, filename='mpl-basic-fft.html')