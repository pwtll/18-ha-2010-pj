"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy
import cv2

# ToDo: Research: normalization of data necessary?
def main():
    def segmentation(path):
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
        return signals

    def signal_to_img(array, image_directory, filename, label):
        if not os.path.exists(image_directory + '/' + label + '/' + filename):
            os.makedirs(image_directory + '/' + label + '/' + filename)

            for count, i in enumerate(array):
                fig = plt.figure(frameon=False)
                plt.plot(i)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                new_filepath = image_directory + '/' + label + '/' + filename + '/' + '{:05d}'.format(count) + '.png'
                fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
                plt.close(fig)

                # downsampling images to desired image_size
                im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.resize(im_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
                cv2.imwrite(new_filepath, im_gray)

    training_folder_csv = '../training'
    image_directory = '../training_images'
    image_size = 128  # 512

    with open(os.path.join(training_folder_csv, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            filename = row[0]
            label = row[1]
            # Lade MatLab Datei
            ecg_segments = segmentation(os.path.join(training_folder_csv, filename + '.mat'))
            signal_to_img(ecg_segments, image_directory, filename, label)
            print(str(row[0]))

if __name__ == '__main__':
    main()
