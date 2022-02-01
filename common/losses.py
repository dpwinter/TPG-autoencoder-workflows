import keras
import keras.backend as K
import tensorflow as tf
from keras.losses import *

def tf_diff_axis0(a):
	return a[1:] - a[:-1]   # calculate difference between all modules of the sample

def mase(y_true, y_pred):
	n = y_true.shape[0]  # M: (sample, 30, 19, 3)
	d = K.sum( K.abs( tf_diff_axis0(y_true) )) / (n-1)
	mae = K.mean( K.abs(y_true - y_pred) )
	return mae/d

def rev_mase(y_true, y_pred):
	n = y_true.shape[0]  # M: (sample, 30, 19, 3)
	d = K.sum( K.abs( tf_diff_axis0(y_true) )) / (n-1)
	mae = K.mean( K.abs(y_true - y_pred) )
	return mae*d
