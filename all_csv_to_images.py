"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""
import os
import numpy as np
from scipy.io import loadmat
import biosppy
import csv


# directory = input("Enter the directory where you want to save the images: ")
directory = 'c:/Users/Philipp Witulla/PycharmProjects/training_images/'
image_size = 512


def main(directory):
    def segmentation(path):
        filename = os.path.basename(path).split('.')[0]
        csv_data = loadmat(path)
        data = np.array(csv_data['val'][0])
        signals = []
        count = 1
        peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=300)[0]
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1] - i)
            x = peaks[count - 1] + diff1 // 2
            y = peaks[count + 1] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        return signals, filename

    def signal_to_img(array, directory, filename):
        import os
        import cv2
        import matplotlib.pyplot as plt

        if not os.path.exists(directory + filename):
            os.makedirs(directory + filename)

        for count, i in enumerate(array):
            fig = plt.figure(frameon=False)
            plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            new_filepath = directory + filename + '\\' + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

            # downsampling images to desired image_size
            im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
            cv2.imwrite(new_filepath, im_gray)
        return directory

    folder = '../training'
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei
            array, filename = segmentation(os.path.join(folder, row[0] + '.mat'))
            directory = signal_to_img(array, directory, filename)
            print(str(row[0]))

    return directory


directory = main(directory)
