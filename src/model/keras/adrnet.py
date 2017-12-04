# -*- coding: utf-8 -*-
import sys
sys.path.append('./../')

import os
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from base_deepnet import BaseDeepNet
from keras import regularizers
from custom_loss import WeightedBinaryCrossEntropy

current_path = os.path.dirname(os.path.realpath(__file__))
# checkpoint_path = "/media/onur/LinuxHDD/ONUR/checkpoints/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
checkpoint_path = "/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/src/model/keras/checkpoints/weights.hdf5"
weights_path = current_path + "/weights/adr_weights.npy"
csv_log_path = current_path + "/../../log/csv_log/training.log"
tensorboard_log_path = current_path + "/../../log/tensorboard_log"


class ADRNet(BaseDeepNet):
	def __init__(self, input_shape, output_shape, class_weights=None):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.weights_path = weights_path
		self.class_weights = class_weights
	
	def create_model(self):
		inputs = Input(shape=(self.input_shape[1],))
		x = Dense(self.input_shape[1], activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(inputs)
		x = Dropout(0.75)(x)
		x = Dense(9000, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(x)
		x = Dropout(0.75)(x)
		x = Dense(9000, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(x)
		x = Dropout(0.75)(x)
		x = Dense(9000, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(x)
		x = Dropout(0.75)(x)
		x = Dense(self.output_shape[1], activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')(x)

		self.model = Model(inputs=inputs, outputs=x)
		sgd = SGD(lr=0.01)
		#rmsprop = RMSprop(lr = 0.1)

		if self.class_weights is not None:
			loss = WeightedBinaryCrossEntropy(self.class_weights)
		else:
			loss = 'binary_crossentropy'

		self.model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])
		return self.model
	
	def fit(self, x, y, epochs = 50, batch_size = 512, verbose=1):
		# csv_logger = CSVLogger(csv_log_path, append=True)
		modelcheckpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
		tensorboard = TensorBoard(log_dir=tensorboard_log_path, histogram_freq=0, write_graph=True, write_images=True)
		callbacks_list = [tensorboard, modelcheckpoint]
		self.model.fit(x, y, validation_split=0.3, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)
		return self.model