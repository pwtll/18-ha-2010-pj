"""
Source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/csv_to_image.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import biosppy
import cv2


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

    def signal_to_img(array, image_directory, filename):
        if not os.path.exists(image_directory + '/' + filename):
            os.makedirs(image_directory + '/' + filename)

        for count, i in enumerate(array):
            fig = plt.figure(frameon=False)
            plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            new_filepath = image_directory + '/' + filename + '/' + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

            # downsampling images to desired image_size
            im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (image_size, image_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4)  # ToDo: resize to 256x256 with correct interpolation method
            cv2.imwrite(new_filepath, im_gray)

    path = input("Enter the path of the csv file: ")  # c:\Users\Philipp Witulla\PycharmProjects\training\train_ecg_00001.mat
    image_directory = '../../single_test'
    image_size = 128  # 512

    filename = os.path.basename(path).split('.')[0]
    ecg_segments = segmentation(path)
    signal_to_img(ecg_segments, image_directory, filename)


main()
