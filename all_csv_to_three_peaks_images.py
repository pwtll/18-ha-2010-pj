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
from joblib import Parallel, delayed


train_path = '../training_complete_6000/'
image_directory = train_path + 'images_256/'
image_size = 256  # 256
sampling_rate = 300


def main(directory):
    def segmentation(path_):
        csv_data = loadmat(path_)
        data = np.array(csv_data['val'][0])
        signals = []
        count = 2
        peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=sampling_rate)[0]
        for i, h in zip(peaks[1:-1:3], peaks[3:-1:3]):
            diff1 = abs(peaks[count - 2] - i)
            diff2 = abs(peaks[count + 2] - h)
            x = peaks[count - 2] + diff1 // 2
            y = peaks[count + 2] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 3
        return signals

    def segment_to_img(array_, directory_, filename_, label_):
        new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
        if not os.path.exists(new_file_directory):
            os.makedirs(new_file_directory)

        for count, i in enumerate(array_):
            fig = plt.figure(frameon=False)
            plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            new_filepath = new_file_directory + '{:05d}'.format(count) + '.png'
            fig.savefig(new_filepath, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)

            # downsampling images to desired image_size
            im_gray = cv2.imread(new_filepath, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (image_size, image_size)) #, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4) # ToDo: resize to 256x256 with correct interpolation method
            cv2.imwrite(new_filepath, im_gray)

    def process_single_img(row):
        filename = row[0]
        label = row[1]

        # Lade MatLab Datei
        filepath = os.path.join(train_path, filename + '.mat')
        ecg_segments = segmentation(filepath)
        segment_to_img(ecg_segments, image_directory, filename, label)
        print(str(row[0]))

    with open(os.path.join(train_path, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile, parallele Ausführung
        Parallel(n_jobs=8)(delayed(process_single_img)(row) for row in csv_reader)  # number of cpus here?

        # # Iteriere über jede Zeile
        # for row in csv_reader:
        #     filename = row[0]
        #     label = row[1]

        #     # skip recreating already existing images
        #     if os.path.exists(directory + filename):
        #         continue

        #     # Lade MatLab Datei
        #     filepath = os.path.join(train_path, filename + '.mat')
        #     ecg_segments = segmentation(filepath)
        #     segment_to_img(ecg_segments, image_directory, filename, label)
        #     print(str(row[0]))


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    main(image_directory)
