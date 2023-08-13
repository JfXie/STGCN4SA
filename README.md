[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Exploring Brain Connectivity with Spatial-Temporal Graph Neural Networks for Improved EEG Seizure Analysis


## Quick Staty

### Datasets

We evaluate our model on the  Temple University Seizure Corpus (TUSZ) v2.0.0 dataset. The `TUSZ v2.0.0` dataset can be downloaded from the [Open Source EEG Resources](https://isip.piconepress.com/projects/tuh_eeg/index.shtml).

After you have registered and downloaded the data, you will see a subdirectory called 'edf' which contains all the EEG signals and their associated labels. 

### Requirements

#### Setting up the environment
- All the development work is done using `Python 3.6`
- Install all the necessary dependencies using `requirement.txt` file. 

### How to run
- **1. Data preparation:**

 To resamples all EEG signals to 200Hz, and saves the resampled signals in 19 EEG channels as h5 files.
 ```shell
 python ./data_preprocessing/TUSZ_v2.py --duration <window size> --raw_edf_dir <tusz-data-dir> --resampled-data-dir <resampled-dir> --output-dir <output-dir>
 ```

 To facilitate reading, we preprocess the dataset into a single .npz file:

  ```shell
  python preprocess_eeg.py
  ```
  
  In addition, distance based adjacency matrix is provided at `./electrode_graph/adj_mx_3d.pkl`.
  
- **3. Configuration:**

  Write the config file in the format of the example.

  We provide a config file at `/config/TUSZ.config`

- **4. Feature extraction:**

  Run `python train_FeatureNet_eeg.py` with -c and -g parameters. After this step, the features learned by a feature net will be stored.

  + -c: The configuration file.
  + -g: The number of the GPU to use. E.g.,`0`,`1,3`. Set this to`-1` if only CPU is used.


- **5. Training:**

  Run `python train_STGCN_eeg.py` with -c and -g parameters. This step uses the extracted features directly. 



- **6. Evaluate**

  Run `python evaluate.py` with -c and -g parameters.

