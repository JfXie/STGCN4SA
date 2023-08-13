# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# INCLUDED_CHANNELS = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
#                      'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6',
#                      'EEG FZ', 'EEG CZ', 'EEG PZ', 'EEG A1', 'EEG A2']
INCLUDED_CHANNELS = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4',
                     'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6',
                     'EEG FZ', 'EEG CZ', 'EEG PZ']

# bipolar_tusz = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3",
#                 "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
#                 "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
bipolar_tusz = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
                "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

bipolar_neonate = ["FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F8",
                   "F8-T4", "T4-T6", "T6-O2", "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FZ-CZ", "CZ-PZ"]