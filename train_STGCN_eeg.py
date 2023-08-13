import os
import numpy as np
import argparse
import shutil
import gc

import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from model.STGCN import build_STGCN
from model.DataGenerator import DominGenerator
from model.Utils import *

print(128 * '#')
print('Start to train STGCN.')

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use, set '-1' to use CPU", required = True)
args = parser.parse_args()
Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
if args.g != "-1":
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #"+args.g)
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

# [train] parameters ('_f' means FeatureNet)
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lambda_GRL   = float(cfgTrain["lambda_GRL"])

# [model] parameters
dense_size            = np.array(str.split(cfgModel["Globaldense"],','),dtype=int)
GLalpha               = float(cfgModel["GLalpha"])
num_of_chev_filters   = int(cfgModel["cheb_filters"])
num_of_time_filters   = int(cfgModel["time_filters"])
time_conv_strides     = int(cfgModel["time_conv_strides"])
time_conv_kernel      = int(cfgModel["time_conv_kernel"])
num_block             = int(cfgModel["num_block"])
cheb_k                = int(cfgModel["cheb_k"])
l1                    = float(cfgModel["l1"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save']+"last.config")


# # 2. Read data and process data

# ## 2.1. Read data
# Each fold corresponds to one subject's data (ISRUC-S3 dataset)
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold

# ## 2.2. Read adjacency matrix
# Prepare Chebyshev polynomial of G_DC
Dis_Conn = np.load(Path['disM'], allow_pickle=True)  # shape:[V,V]
L_DC = scaled_Laplacian(Dis_Conn)                    # Calculate laplacian matrix
cheb_poly_DC = cheb_polynomial(L_DC, cheb_k)         # K-order Chebyshev polynomial

print("Read data successfully")
print('Number of samples: ', Fold_Num)
# Fold_Num_c  = Fold_Num + 1 - context
# print('Number of samples: ',np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# ## 2.3. Build kFoldGenerator or DominGenerator
# Dom_Generator = DominGenerator(Fold_Num_c)


# # 3. Model training (cross validation)


print(128*'_')

# Instantiation optimizer
opt = Instantiation_optim(optimizer, learn_rate)
    # Instantiation l1, l2 regularizer
regularizer = Instantiation_regularizer(l1, l2)
    
    # get i th-fold feature and label
Features = np.load(Path['Save']+'Feature_'+'.npz', allow_pickle=True)
train_feature = Features['train_feature']
val_feature   = Features['val_feature']
train_targets = Features['train_targets']
val_targets   = Features['val_targets']

## Use the feature to train STGCN

print('Feature',train_feature.shape, val_feature.shape)
# train_feature, train_targets  = AddContext_SingleSub(train_feature, train_targets,context)
# val_feature, val_targets    = AddContext_SingleSub(val_feature, val_targets, context)


sample_shape = (val_feature.shape[1:])
    
print('Feature with context:',train_feature.shape, val_feature.shape)
model = build_STGCN(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_poly_DC,
                                  time_conv_kernel, sample_shape, num_block, dense_size, opt, GLalpha, regularizer, 
                                  dropout, lambda_GRL, num_classes=1) # '_p' model is without GRL
        
    # train
history = model.fit(
    x = train_feature,
    y = train_targets,
        epochs = num_epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = (val_feature, val_targets),
        verbose = 2,
        callbacks=[keras.callbacks.ModelCheckpoint(Path['Save']+'STGCN_Best'+'.h5', 
                                                   monitor='val_Label_acc', 
                                                   verbose=0, 
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1 )])
    
    # save the final model
model.save(Path['Save']+'STGCN_Final'+'.h5')
    
    # Save training information

fit_loss = np.array(history.history['loss'])*Fold_Num
fit_acc = np.array(history.history['Label_acc'])*Fold_Num
fit_val_loss = np.array(history.history['val_loss'])*Fold_Num
fit_val_acc = np.array(history.history['val_Label_acc'])*Fold_Num
    
saveFile = open(Path['Save'] + "Result_STGCN.txt", 'a+')
print('Fold #', file=saveFile)
print(history.history, file=saveFile)
saveFile.close()

    # Fold finish
keras.backend.clear_session()
del model, train_feature, train_targets, val_feature, val_targets
gc.collect()

# # 4. Final results

# Average training performance
fit_acc      = fit_acc/Fold_Num
fit_loss     = fit_loss/Fold_Num
fit_val_loss = fit_val_loss/Fold_Num
fit_val_acc  = fit_val_acc/Fold_Num

# Draw ACC / loss curve and save
VariationCurve(fit_acc, fit_val_acc, 'Acc', Path['Save'], figsize=(9, 6))
VariationCurve(fit_loss, fit_val_loss, 'Loss', Path['Save'], figsize=(9, 6))

saveFile = open(Path['Save'] + "Result_STGCN.txt", 'a+')
print(history.history, file=saveFile)
saveFile.close()

print(128 * '_')
print('End of training STGCN.')
print(128 * '#')
