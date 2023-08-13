# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
from collections import Counter
import h5py
import pyedflib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

from utils import compute_fft, create_output_dirs
from montage import INCLUDED_CHANNELS, bipolar_tusz


def get_edf_files(resampled_data_dir):
    """
    Get .edf file names and corresponding paths

    Returns:
    --------
    dict
        key: edf file name
        val: corresponding path
        file names are in format {official patient number}_{session number}_{segment number}, e.g., 00000258_s002_t000
    """
    edf_files = {}
    for root, _, files in os.walk(resampled_data_dir):
        for name in files:
            if '.h5' in name:
                # some patients have sessions listed under multiple montage folders, only the last one will be saved
                edf_files[name] = os.path.join(root, name)

    return edf_files


def get_label_file_names(version, raw_data_dir, annotation_type):
    """
    Get label file names

    Returns:
    --------
    dict
        label file names and corresponding paths
    """
    # postfix of label files
    if annotation_type == 'term-based':
        postfix = 'tse_bi' if version == '1.5.2' else 'csv_bi'
    else:
        postfix = 'tse' if version == '1.5.2' else 'csv'
    label_files = {}
    for root, _, files in os.walk(raw_data_dir):
        for name in files:
            if postfix in name:
                # some patients have sessions listed under multiple montage folders, only the last one will be saved
                label_files[name] = os.path.join(root, name)

    return label_files


def get_ordered_channels(file_name, verbose, labels_object, channel_names):
    """
    Some person may missing necessary channels, these persons will be excluded
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/data_utils.py
    """
    labels = list(labels_object)
    for i, label in enumerate(labels):
        labels[i] = label.split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def get_edf_signals(edf, ordered_channels):
    """
    Get EEG signal in edf file

    Parameters:
    -----------
    edf:
        edf object
    ordered_channels: list
        list of channel indexes

    Returns:
    --------
    numpy.ndarray
        shape (num_channels, num_data_points)
    """
    signals = np.zeros((len(ordered_channels), edf.getNSamples()[0]))
    for i, index in enumerate(ordered_channels):
        try:
            signals[i, :] = edf.readSignal(index)
        except:
            raise Exception("Get edf signals failed")
    return signals


def resample_data(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freqency
    refer to https://github.com/tsy935/eeg-gnn-ssl/blob/main/data/resample_signals.py

    Parameters:
    -----------
    signals: numpy.ndarray
        EEG signal slice, (num_channels, num_data_points)
    to_freq: int
        Re-sampled frequency in Hz
    window_size: int
        time window in seconds

    Returns:
    --------
    numpy.ndarray
        shape (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)

    return resampled


def resample_all(raw_edf_dir, to_freq, save_dir):
    """
    Resample all edf files in raw_edf_dir to to_freq and save to save_dir

    Returns:
    --------
    dict
        edf file names and corresponding paths after resampling
    """
    raw_edfs = []
    for root, _, files in os.walk(raw_edf_dir):
        for file in files:
            if ".edf" in file:
                raw_edfs.append(os.path.join(root, file))
    print(f"Number of raw edf files: {len(raw_edfs)}")

    resampled_edfs = {}
    failed_files = []
    for _, edf_fn in enumerate(tqdm(raw_edfs)):
        new_file_name = f"{edf_fn.split('/')[-1].split('.edf')[0]}.h5"
        resampled_edf = os.path.join(save_dir, new_file_name)
        if os.path.exists(resampled_edf):
            resampled_edfs[new_file_name] = resampled_edf
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)

            ordered_channels = get_ordered_channels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signal_array = get_edf_signals(f, ordered_channels)
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != to_freq:
                signal_array = resample_data(
                    signal_array,
                    to_freq=to_freq,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(resampled_edf, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
            resampled_edfs[new_file_name] = resampled_edf

        except Exception:
            # pepole may missing some channels
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))

    return resampled_edfs


def get_train_test_ids_v1(file_markers_dir, train_meta, test_meta):
    """
    Get train and test ids from file markers in v1.5.2

    Returns:
    --------
    list
        official patient numbers of train set
    list
        official patient numbers of test set
    """
    def __get_ids(meta):
        ids = set()
        for line in open(meta, 'r').readlines():
            ids.add(line.split('.')[0].split('_')[0])
        return ids

    train_ids, test_ids = set(), set()
    for meta in train_meta:
        train_ids.update(__get_ids(os.path.join(file_markers_dir, meta)))
    for meta in test_meta:
        test_ids.update(__get_ids(os.path.join(file_markers_dir, meta)))

    return list(train_ids), list(test_ids)


def get_train_test_ids_v2(raw_data_dir):
    """
    Get train and test ids in v2.0.0

    Returns:
    --------
    list
        official patient numbers of train set
    list
        official patient numbers of test set
    """
    train_ids = os.listdir(os.path.join(raw_data_dir, 'edf', 'train'))
    test_ids = os.listdir(os.path.join(raw_data_dir, 'edf', 'eval'))

    return train_ids, test_ids


def __get_term_based_features(identifier, edf, frequency, duration, montage_type):
    """
    generate features in dataframe format

    Returns:
    --------
    pandas.core.frame.DataFrame
        feature, each row is a sample, each column is a channel

    Example:
    features:
                        FP1-F7      F7-T3 ...
    id
    aaaaadse_s001_t003  -26.855483   0.305176 ...
    aaaaadse_s001_t003  -22.583020   1.831056 ...
    ...
    aaaaadse_s001_t003   107.421931   9.155278 ...
    """
    num_samples = len(edf) - len(edf) // frequency % duration * frequency  # truncate the additional seconds
    if num_samples == 0:
        return None
    features = pd.DataFrame(index=np.arange(num_samples))
    channels_map = {k: v for v, k in enumerate(INCLUDED_CHANNELS)}

    if montage_type == 'bipolar':
        for bp in bipolar_tusz:
            start, end = bp.split('-')
            start = f"EEG {start}"
            end = f"EEG {end}"
            bp_value = edf[:num_samples, channels_map[start]] - edf[:num_samples, channels_map[end]]
            features.insert(len(features.columns), bp, bp_value)
    else:
        for channel in INCLUDED_CHANNELS:
            features.insert(len(features.columns), channel, edf[:num_samples, channels_map[channel]])
    features.insert(0, 'id', identifier)
    features.set_index('id', inplace=True)

    return features


def __get_event_based_features(identifier, edf, frequency, duration, montage_type):
    """
    generate features in dataframe format

    Returns:
    --------
    pandas.core.frame.DataFrame
        feature, each row is a sample, each column is a channel

    Example:
    features:
                                val
    id
    aaaaadse_s001_t003_FP1-F7   -26.855483
    aaaaadse_s001_t003_FP1-F7   -22.583020
    ...
    aaaaadse_s001_t003_FP1-F7   107.421931
    aaaaadse_s001_t003_F7-T3    0.305176
    aaaaadse_s001_t003_F7-T3    1.831056
    ...
    aaaaadse_s001_t003_F7-T3    9.155278
    ...
    """
    features = None
    channels_map = {k: v for v, k in enumerate(INCLUDED_CHANNELS)}
    num_samples = len(edf) - len(edf) // frequency % duration * frequency  # truncate the additional seconds
    if num_samples == 0:
        return None

    if montage_type == 'bipolar':
        for bp in bipolar_tusz:
            start, end = bp.split('-')
            start = f"EEG {start}"
            end = f"EEG {end}"
            bp_value = edf[:num_samples, channels_map[start]] - edf[:num_samples, channels_map[end]]

            df = pd.DataFrame(index=np.arange(num_samples), columns=['val'])
            df['val'] = bp_value
            df.insert(0, 'id', f"{identifier}_{bp}")
            df.set_index('id', inplace=True)
            features = pd.concat([features, df])
    else:
        for channel in INCLUDED_CHANNELS:
            df = pd.DataFrame(index=np.arange(len(edf)), columns=['val'])
            df['val'] = edf[:num_samples, channels_map[channel]]
            df.insert(0, 'id', f"{identifier}_{channel}")
            df.set_index('id', inplace=True)
            features = pd.concat([features, df])

    return features


def __get_term_based_labels(version, identifier, edf, frequency, duration, label_files):
    """
    Get bi-class labels of a given edf file

    Returns:
    --------
    pandas.core.frame.DataFrame
        label, each row is a sample

    Example:
        label
    0   0.0
    1   0.0
    ...
    104 0.0
    105 0.0
    """
    postfix = '.tse_bi' if version == '1.5.2' else '.csv_bi'
    num_samples = len(edf) // frequency  # in seconds
    num_samples -= num_samples % duration  # truncate the additional labels
    if num_samples == 0:
        return None

    labels = pd.DataFrame(index=np.arange(num_samples), columns=['label'])
    labels['label'] = 0
    annotations_raw = pd.read_csv(label_files[identifier + postfix], comment='#')
    for _, row in annotations_raw.iterrows():
        if row['label'] != 'seiz':
            start = round(row['start_time'])
            end = min(round(row['stop_time']), num_samples)
            labels['label'][start:end] = 1

    return labels


def __get_event_based_labels(version, identifier, edf, frequency, duration, label_files):
    """
    Get multi-class labels of a given edf file

    Returns:
    --------
    pandas.core.frame.DataFrame
        label, each row is a sample

    Example:
        label
    0   'bckg'
    1   'fnsz'
    ...
    2308  'gnsz'
    2309  'bckg'
    """
    if version != '2.0.0':
        raise Exception("Multi-class labels are only supported in v2.0.0")

    postfix = '.csv'
    num_samples = len(edf) // frequency  # in seconds
    num_samples -= num_samples % duration  # truncate the additional labels
    if num_samples == 0:
        return None

    labels = pd.DataFrame(index=np.arange(num_samples*len(bipolar_tusz)), columns=['label'])
    labels['label'] = 'bckg'
    annotations_raw = pd.read_csv(label_files[identifier + postfix], comment='#')
    for _, row in annotations_raw.iterrows():
        if row['label'] != 'bckg':
            channel_idx = bipolar_tusz.index(row['channel'])
            start = round(row['start_time']) + channel_idx * num_samples
            end = min(round(row['stop_time']) + channel_idx * num_samples, (channel_idx+1)*num_samples)
            labels['label'][start:end] = row['label']
    labels['label'] = labels['label'].astype('category')

    return labels


def get_feature_and_label(version, annotation_type, edf_file, edf_file_path, label_files, frequency, duration, montage_type,
                          feature_dir, label_dir):
    """
    Get features and labels of a given edf file
    if serialized file exists, read from disk, otherwise calculate and serialize in DataFrame format

    Returns:
    --------
    pandas.core.frame.DataFrame
        feature, each row is a sample, each column is a channel
    pandas.core.frame.DataFrame
        label, each row is a sample, each column is a label
    """
    identifier = edf_file.split('.')[0]
    # resume from serialized file if exists
    if os.path.exists(f"{feature_dir}/{identifier}.pkl"):
        feature = pd.read_pickle(f"{feature_dir}/{identifier}.pkl")
        label = pd.read_pickle(f"{label_dir}/{identifier}.pkl")
    else:
        identifier = edf_file.split('.')[0]
        edf = np.array(h5py.File(edf_file_path, 'r')['resampled_signal'][()]).T
        assert len(edf) % frequency == 0, f"{edf_file} EDF file shape error."

        if annotation_type == 'term-based':
            feature = __get_term_based_features(identifier, edf, frequency, duration, montage_type)
            label = __get_term_based_labels(version, identifier, edf, frequency, duration, label_files)
        else:
            feature = __get_event_based_features(identifier, edf, frequency, duration, montage_type)
            label = __get_event_based_labels(version, identifier, edf, frequency, duration, label_files)

        try:
            if feature is not None and label is not None:
                feature.to_pickle(f"{feature_dir}/{identifier}.pkl")
                label.to_pickle(f"{label_dir}/{identifier}.pkl")
        except:
            raise Exception(f"{edf_file} IO error in serializating raw feature and label dataframe.")

    return feature, label


def get_linewise_feature_and_label(annotation_type, edf_file, nedf, label, frequency, duration, feature_linewise_dir, label_linewise_dir):
    """
    Calculate and serialize line-wise feature and label
    Format: One sample each line, duration * frequency * feature_num
    sample_num =  int(len(nedf) / frequency / duration)
    """
    identifier = edf_file.split('.')[0]
    if os.path.exists(f"{feature_linewise_dir}/{identifier}.pkl"):
        nedf = pd.read_pickle(f"{feature_linewise_dir}/{identifier}.pkl")
        label = pd.read_pickle(f"{label_linewise_dir}/{identifier}.pkl")
    else:
        nedf, label = nedf.copy(deep=True), label.copy(deep=True)
        # shape: (sample_num, (duration * frequency * feature_num))

        # for features, merge every duration*frequency rows into one row
        # for labels, merge every duration rows into one row
        if annotation_type == 'term-based':
            nedf = pd.DataFrame(index=nedf.index.values.reshape(-1, duration * frequency)[:, 0],
                                data=nedf.values.reshape(-1, duration * frequency * len(nedf.columns)),
                                columns=np.tile(np.array(nedf.columns), duration * frequency))
            label = pd.DataFrame(index=nedf.index,
                                 data=np.any(label.values.reshape(-1, duration), axis=1).astype(int))
        else:
            nedf = pd.DataFrame(index=nedf.index.values.reshape(-1, duration * frequency)[:, 0],
                                data=nedf.values.reshape(-1, duration * frequency))
            label = pd.DataFrame(index=nedf.index, data=np.apply_along_axis(lambda x: Counter(
                x).most_common(1)[0][0], axis=1, arr=label['label'].values.reshape(-1, duration)))
        # Serialize to disk
        try:
            nedf.to_pickle(f"{feature_linewise_dir}/{identifier}.pkl")
            label.to_pickle(f"{label_linewise_dir}/{identifier}.pkl")
        except:
            raise Exception(f"{edf_file} IO error in serialization for LINED feature and label dataframe.")


def get_train_test_edf_files(edf_files, train_ids, test_ids):
    """
    Get train and test edf files
    """
    edf_files_train = []
    edf_files_test = []
    for edf_file in edf_files:
        patient_id = edf_file.split('.')[0].split('_')[0]
        if patient_id in train_ids:
            edf_files_train.append(edf_file)
        elif patient_id in test_ids:
            edf_files_test.append(edf_file)

    return edf_files_train, edf_files_test


def get_meta(montage_type, frequency, edf_files, train_ids, test_ids, classes, fft, feature_dir, meta_dir):
    """
    Calculate and serialize meta information to pickle, in format:
        {
            'classes': list of all classes available,
            'train': list of train edf file names,
            'test': list of test edf file names,
            'mean@train': channel-wise mean of train,
            'std@train': channel-wise std of train,
            'mean@test': channel-wise mean of test,
            'std@test': channel-wise std of test,
            'mean_fft@train': channel-wise mean of train fft,
            'std_fft@train': channel-wise std of train fft,
        }
    """
    meta = {
        'classes': classes
    }

    edf_files_train, edf_files_test = get_train_test_edf_files(edf_files, train_ids, test_ids)
    print(f'Number of train / test edf files: {len(edf_files_train)} / {len(edf_files_test)}')
    meta['train'] = edf_files_train
    meta['test'] = edf_files_test

    def concate_edfs(ids):
        edfs = [pd.read_pickle(feature_dir+f) for f in os.listdir(feature_dir) if f.split('.')[0].split('_')[0] in ids]
        return pd.concat(edfs, axis='index')

    print("Processing test meta...")
    test_df = concate_edfs(test_ids)
    pd.to_pickle(test_df, f"{meta_dir}/test_df.pkl")
    meta['mean@test'] = test_df.mean(axis='index')
    meta['std@test'] = test_df.std(axis='index')
    del test_df

    print("Processing train meta...")
    train_df = concate_edfs(train_ids)
    pd.to_pickle(train_df, f"{meta_dir}/train_df.pkl")
    meta['mean@train'] = train_df.mean(axis='index')
    meta['std@train'] = train_df.std(axis='index')

    if fft:
        print("Calculating mean/std of train fft...")
        num_polars = len(bipolar_tusz) if montage_type == 'bipolar' else len(INCLUDED_CHANNELS)
        ffted = compute_fft(train_df.values.reshape(-1, frequency, num_polars).transpose(0, 2, 1),  # second_num * num_channels * frequency
                            frequency)
        train_df_fft = pd.DataFrame(index=train_df.index,
                                    columns=train_df.columns,
                                    data=ffted.transpose(0, 2, 1).reshape(-1, num_polars))
        meta["mean_fft@train"] = train_df_fft.mean(axis='index')
        meta["std_fft@train"] = train_df_fft.std(axis='index')
    print(meta)
    pd.to_pickle(meta, f"{meta_dir}/meta.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tusz-version', type=str, default="2.0.0", choices=['1.5.2', '2.0.0'],
                        help='tusz version')
    parser.add_argument('--annotation-type', type=str, default='term-based', choices=['term-based', 'event-based'],
                        help='annotation types, term-based bi-class annotations & event-based multi-class annotations')
    parser.add_argument('--duration', type=int, default=60,
                        help='window size, in seconds')
    parser.add_argument('--frequency', type=int, default=200,
                        help='resample frequency')
    parser.add_argument('--montage-type', type=str, default="bipolar", choices=['unipolar', 'bipolar'],
                        help='unipolar or bipolar')
    parser.add_argument('--raw-data-dir', type=str, default="/xxx/TUSZ_V2.0.0/EEGs/",
                        help='path to raw data')
    parser.add_argument('--resampled-data-dir', type=str, default="/xxx/tusz_resampled_v2.0.0",
                        help='path to resampled h5py files, the folder can be empty. If empty, the resampling process will be performed')
    parser.add_argument('--file-markers-dir', type=str, default="data/file_markers_detection",
                        help='path to file markers, only required in v1.5.2 for train/test splitting')
    parser.add_argument('--train-meta', type=str, nargs='+', default=['trainSet_seq2seq_12s_sz.txt', 'trainSet_seq2seq_12s_nosz.txt'],
                        help='file markers of train set, only required for v1.5.2.')
    parser.add_argument('--test-meta', type=str, nargs='+', default=['devSet_seq2seq_12s_nosz.txt', 'devSet_seq2seq_12s_sz.txt'],
                        help='file markers of train set, only required for v1.5.2.')
    parser.add_argument("--fft", action='store_true', default=False)
    parser.add_argument('--output-dir', type=str, default="/xxx/tusz_processed_v2.0.0",
                        help='root path for output data.')
    args = parser.parse_args()

    print("Creating output directories...")
    meta_dir, feature_dir, label_dir, feature_linewise_dir, label_linewise_dir = create_output_dirs(
        args.output_dir, args.montage_type, args.annotation_type)
    if not os.path.isdir(args.resampled_data_dir):
        os.mkdir(args.resampled_data_dir)

    print("Reading resampled edf and label file names...")
    edf_files = get_edf_files(args.resampled_data_dir)
    print(f"Number of resampled edf files: {len(edf_files)}")

    # process resampleing if no files found in resampled path
    if len(edf_files) == 0:
        print("Resampling...")
        edf_files = resample_all(args.raw_data_dir, args.frequency, args.resampled_data_dir)
        print(f"Number of resampled edf files: {len(edf_files)}")

    print("Getting train/test ids...")
    if args.tusz_version == '1.5.2':
        train_ids, test_ids = get_train_test_ids_v1(args.file_markers_dir, args.train_meta, args.test_meta)
    else:
        train_ids, test_ids = get_train_test_ids_v2(args.raw_data_dir)
    print(f"Number of train / test ids: {len(train_ids)} / {len(test_ids)}")

    label_files = get_label_file_names(args.tusz_version, args.raw_data_dir, args.annotation_type)
    print(f"Number of label files: {len(label_files)}")

    print("Calculating and serializing features and labels...")
    classes = set() # all classes in label
    for edf_file in tqdm(edf_files):
        patient_id = edf_file.split('.')[0].split('_')[0]
        if patient_id not in train_ids and patient_id not in test_ids:
            continue
        # feature and label
        nedf, label = get_feature_and_label(args.tusz_version, args.annotation_type, edf_file, edf_files[edf_file], label_files,
                                            args.frequency, args.duration, args.montage_type, feature_dir, label_dir)
        if nedf is not None and label is not None:
            classes.update(label['label'].unique())
            # line-wise feature and label
            get_linewise_feature_and_label(args.annotation_type, edf_file, nedf, label, args.frequency,
                                           args.duration, feature_linewise_dir, label_linewise_dir)

    print("Calculating and serializing meta...")
    # classes = {1}
    get_meta(args.montage_type, args.frequency, edf_files, train_ids, test_ids, list(classes), args.fft, feature_dir, meta_dir)

if __name__ == '__main__':
    main()