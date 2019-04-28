
import os
import time
import datetime
import random
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
import data_loader
from keras.models import model_from_json

im_size = 320



model = densenet.DenseNet(nb_classes=1, img_dim=(320,320,1), depth=22, nb_dense_block=4, growth_rate=12, nb_filter=16, dropout_rate=0.2, weight_decay=1E-4)

model.load_weights('./save_models/MURA_modle@epochs52.h5')

X_valid_path, Y_valid = data_loader.load_path(root_path = './valid/XR_HUMERUS', size = im_size)
X_valid = data_loader.load_image(X_valid_path,im_size)
y1 = model.predict(X_valid, batch_size=None, verbose=0, steps=None)

j = len(y1)

for i in range (0, j):
	if y1[i]>0.5 :
		print(X_valid_path[i],":\t","Positive\t", y1[i])
	else:
		print(X_valid_path[i],":\t","Negative\t", y1[i])




