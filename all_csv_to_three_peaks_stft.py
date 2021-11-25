"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy

# c:\Users\Philipp Witulla\PycharmProjects\training\train_ecg_00001.mat
directory = 'c:/Wettbewerb/picture/training__images_stft_try2/'


def main(directory):
    def create_stft(path):
        sampling_rate = 300

        filename = os.path.basename(path).split('.')[0]
        csv_data = loadmat(path)
        data = np.array(csv_data['val'][0])
        signals = []
        count = 2
        peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=300)[0]
        # TODO Divide ECG data into segments of 3 peaks per image
        for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
            diff1 = abs(peaks[count - 2] - i)
            diff2 = abs(peaks[count + 2] - h)
            x = peaks[count - 2] + diff1 // 2
            y = peaks[count + 2] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 3

        fig = plt.figure()

        # # ToDo: define correct sized Hamming window (see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html)
        # plt.specgram(np.array(signals), cmap='nipy_spectral', Fs=sampling_rate,
        #              sides='twosided', scale='dB') #, NFFT=64, noverlap=32)  # , NFFT=128, noverlap=64

        ''' 
        # Alternative spectrogram
        f, t, Sxx = signal.spectrogram(data, sampling_rate) # , nperseg=64, noverlap=32 #, nperseg=sampling_rate)
        plt.pcolormesh(t, f, Sxx, shading='auto', norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='nipy_spectral') # , norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())  # , shading='gouraud'
        #plt.ylim(f.min(), f.max())
        plt.yscale('symlog')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        '''

        if not os.path.exists(directory + filename):
            os.makedirs(directory + filename)

        for count, i in enumerate(signals):
            fig = plt.figure(frameon=False)
            # TODO creat segement_stft of "3 peaks per image"
            plt.specgram(np.array(signals)[count], cmap='nipy_spectral', Fs=sampling_rate,
                         sides='twosided', scale='dB')  # , NFFT=64, noverlap=32)  # , NFFT=128, noverlap=64
            # plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            new_filepath = directory + filename + '\\' + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

        print(filename)

        return directory

    folder = '../training'
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei
            directory = create_stft(os.path.join(folder, row[0] + '.mat'))
    return directory

# filepath = main()
directory = main(directory)
