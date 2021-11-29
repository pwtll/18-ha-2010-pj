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

image_size = 256  # 256
sampling_rate = 300

train_path = '../training/'
image_directory = train_path + 'single_images_' + str(image_size) + '/'     # Enter the directory for the training images seperated in their classes


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

    def single_segment_to_img(array_, directory_, filename_, label_):
        new_file_directory = directory_ + '/' + label_ + '/' + filename_ + '/'
        if not os.path.exists(new_file_directory):
            os.makedirs(new_file_directory)

            fig = plt.figure(frameon=False)
            plt.plot(array_)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            count = 1

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
        #
        if len(ecg_segments) > 0:
            middle_segment = int(len(ecg_segments) // 2)

            # processes just a single three r-peak ecg segment from the middle of the signal
            single_segment_to_img(ecg_segments[middle_segment], image_directory, filename, label)
        else:
            # save the filename of the errorneous ecg_lead in a logging file
            # open the file in the write mode
            with open('dataset/errorneous_ecg_leads.csv', 'a') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow(row)

                # close the file
                f.close()

        print(str(row[0]))

    with open(os.path.join(train_path, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile, parallele Ausführung
        Parallel(n_jobs=8)(delayed(process_single_img)(row) for row in csv_reader)  # number of cpus here?

        # Iteriere über jede Zeile
        #for row in csv_reader:
        #    filename = row[0]
        #    label = row[1]

        #    # skip recreating already existing images
        #    if os.path.exists(directory + filename):
        #        continue

        #    # Lade MatLab Datei
        #    filepath = os.path.join(train_path, filename + '.mat')
        #    ecg_segments = segmentation(filepath)

        #    if len(ecg_segments) > 0:
        #        middle_segment = int(len(ecg_segments) // 2)
        #
        #        # processes just a single three r-peak ecg segment from the middle of the signal
        #        single_segment_to_img(ecg_segments[middle_segment], image_directory, filename, label)
        #        # segment_to_img(ecg_segments, image_directory, filename, label)
        #    else:
        #        # save the filename of the errorneous ecg_lead in a logging file
        #        # open the file in the write mode
        #        with open('dataset/errorneous_ecg_leads.csv', 'a') as f:
        #            # create the csv writer
        #            writer = csv.writer(f)
        #
        #            # write a row to the csv file
        #            writer.writerow(row)
        #
        #            # close the file
        #            f.close()
        #
        #        continue

        #    print(str(row[0]))


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    main(image_directory)
