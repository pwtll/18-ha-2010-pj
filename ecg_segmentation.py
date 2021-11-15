def segmentation(filename):
    """ Gets the ECG segment (wave) and saves to image """

    # reads raw signal (ECG I)
    record = wfdb.rdsamp(filename)
    # get the first column signal
    data = record[0][:, 1]
    signals = []
    count = 1

    # apply Christov Segmenter to get separate waves
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=record[1]['fs'])[0]
    for i in (peaks[1:-1]):
        diff1 = abs(peaks[count - 1] - i)
        diff2 = abs(peaks[count + 1] - i)
        x = peaks[count - 1] + diff1 // 2
        y = peaks[count + 1] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 1

    # save the waves as the images
    for signal in signals:
        save_fig(signal)