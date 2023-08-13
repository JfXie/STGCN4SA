import numpy as np
import pandas as pd
import scipy.io as scio
from os import path
from scipy import signal

path_Data   = './data/TUSZ_v2/feature/'
path_output    = './data/TUSZ_v2/'


'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''

file_names = pd.read_csv('./data/TUSZ_v2/file_list.csv')


fold_label = []
fold_eeg = []
fold_len = 0

for _, row in file_names.iterrows():
    file_name = row['file_name']
    label = row['lable']
    # print('Read subject', file_name)
    
    eeg = pd.read_pickle(f"{path_Data}/{file_name}.pkl")
    eeg = eeg.values
    
    fold_label.append(label)
    fold_eeg.append(eeg)
    fold_len += 1
print('Preprocess over.')

np.savez(path.join(path_output, 'TUSZ_v2.npz'),
    Fold_data = fold_eeg,
    Fold_label = fold_label,
    Fold_len = fold_len
)
print('Saved to', path.join(path_output, 'TUSZ_v2.npz'))
