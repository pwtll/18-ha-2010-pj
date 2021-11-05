import scipy.io
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import numpy as np

path_to_csv = r"C:\Users\Philipp Witulla\PycharmProjects\training"
Tk().withdraw()
file_directory = askopenfilename(initialdir=path_to_csv, filetypes =((".mat File", "*.mat"),(".csv File", "*.csv"),("All Files","*.*")))
filename = os.path.basename(file_directory)

data = scipy.io.loadmat(file_directory)

fig, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False, figsize=(12,6))
fig.suptitle('ECG time series and FFT of ' + filename, fontsize=12)
params = {'mathtext.default': 'regular',
          'axes.titlesize': 16,
          'axes.labelsize': 14,
          'font.family' : 'sans-serif',
          'font.sans-serif' : 'Tahoma'
          }        # mathematische Achsenbeschriftungen
plt.rcParams.update(params)

number_of_data_points = range(data['val'].size)
time = []
for data_point in number_of_data_points:
    time.append(data_point/300)            # 300 Hz sampling Frequency

ax1.plot(time, data['val'][0])        # ax1.plot(time[1500:], data['val'][0][1500:])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('ECG Amplitude')
ax1.grid(b=True, which='both', axis='y')

# Number of sample points
N = data['val'].size    # -1500
# sample spacing
T = 1.0 / 300.0
x = np.linspace(0.0, N*T, N, endpoint=False)

yf = fft(data['val'][0])    # fft(data['val'][0][1500:])
xf = fftfreq(N, T)[:N//2]

ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('FFT Amplitude')
plt.grid(b=True, which='both', axis='y')
ax2.set_xscale('log')

fig.tight_layout()
plt.show()