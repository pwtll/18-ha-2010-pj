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
train_path = '../training_complete_6000/'
image_directory = train_path + 'stft/'
sampling_rate = 300


def main(directory):
    def segmentation(path_):
        csv_data = loadmat(path_)
        data = np.array(csv_data['val'][0])
        signals = []
        count = 2
        peaks = biosppy.signals.ecg.hamilton_segmenter(signal=data, sampling_rate=sampling_rate)[0]
        for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
            diff1 = abs(peaks[count - 2] - i)
            diff2 = abs(peaks[count + 2] - h)
            x = peaks[count - 2] + diff1 // 2
            y = peaks[count + 2] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 3
        return signals

    def create_stft(signals_ , directory_, filename_, label_):
        new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
        if not os.path.exists(new_file_directory):
            os.makedirs(new_file_directory)

        for count, i in enumerate(signals_):
            fig = plt.figure(frameon=False)
            # TODO: create segment_stft of "3 peaks per image"
            # ToDo: define correct sized Hamming window (see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html)
            # ToDo: add logrithmic y-axis
            # ToDo: define correct time&frequency resolution
            # ToDo: compare interpolated STFT with blockwise STFT
            plt.specgram(np.array(signals_)[count], cmap='nipy_spectral', Fs=sampling_rate, NFFT=128, noverlap=64,
                         sides='onesided', scale='dB')  # , NFFT=64, noverlap=32)  # , sides='twosided', NFFT=128, noverlap=64

            # plot spectrogram
            # plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.close(fig)
            ''' 
            # Alternative spectrogram
            f, t, Sxx = signal.spectrogram(data, sampling_rate) # , nperseg=64, noverlap=32 #, nperseg=sampling_rate)
            plt.pcolormesh(t, f, Sxx, shading='auto', norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='nipy_spectral') # , norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())  # , shading='gouraud'
            #plt.ylim(f.min(), f.max())
            plt.yscale('symlog')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            '''

            new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

        print(filename)

    with open(os.path.join(train_path, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei
            filename = row[0]
            label = row[1]

            # skip recreating already existing images
            if os.path.exists(directory + filename):
                continue

            # Lade MatLab Datei
            filepath = os.path.join(train_path, filename + '.mat')
            ecg_segments = segmentation(filepath)
            create_stft(ecg_segments, image_directory, filename, label)
            print(str(row[0]))


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    main(image_directory)
