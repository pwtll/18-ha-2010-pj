"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = input("Enter the path of the csv file: ")    # c:\Users\Philipp Witulla\PycharmProjects\training\train_ecg_00001.mat
image_directory = '../../training_images'


def main(path):
    def create_stft(path):
        sampling_rate = 300

        filename = os.path.basename(path).split('.')[0]
        csv_data = loadmat(path)
        data = np.array(csv_data['val'][0])


        fig = plt.figure()
        # ToDo: define correct sized Hamming window (see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html)
        plt.specgram(data, cmap='nipy_spectral', Fs=sampling_rate,
                     sides='twosided', scale='dB') #, NFFT=64, noverlap=32)  # , NFFT=128, noverlap=64

        #f, t, Sxx = signal.spectrogram(data, sampling_rate) # , nperseg=64, noverlap=32 #, nperseg=sampling_rate)
        #plt.pcolormesh(t, f, Sxx, shading='auto', norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='nipy_spectral') # , norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())  # , shading='gouraud'
        ##plt.ylim(f.min(), f.max())
        #plt.yscale('symlog')
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')

        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.show()

        if not os.path.exists(image_directory + '/' + 'stft_test/'):
            os.makedirs(image_directory + '/' + 'stft_test/')

        filepath = image_directory + '/' + 'stft_test/' + filename + '.png'
        fig.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.0)


    create_stft(path)


main(path)
