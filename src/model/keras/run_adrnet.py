#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
from math import floor
import numpy as np
from datetime import datetime
from adrnet import ADRNet
from keras.models import load_model
from sklearn.metrics import roc_auc_score

phase = "phase1"

train_data_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/'+ phase +'/train_data.pkl'
train_label_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/'+ phase +'/train_label.pkl'

test_data_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/'+ phase +'/test_data.pkl'
test_label_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/'+ phase +'/test_label.pkl'
checkpoint_path = "/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/src/model/keras/checkpoints/weights.hdf5"

prune_count = 5
train_size = 0.9


def calc_class_weights(Y):
	weights = dict()
	for i in range(Y.shape[1]):
		counts = np.bincount(Y[:, i].astype(int))
		num_neg = float(counts[0])
		num_pos = float(counts[1])
		weights[i] = num_pos / (num_pos+num_neg)
	return weights


def calc_sample_weights(Y):
	weights = []
	sample_weights = []
	for i in range(Y.shape[1]):
		counts = np.bincount(Y[:, i].astype(int))
		num_neg = float(counts[0])
		num_pos = float(counts[1])
		weights.append(num_pos / (num_pos+num_neg))
	for i in range(Y.shape[0]):
		sample_weights.append(weights)

	return np.array(sample_weights)


def create_dataset():
	train_data_df = pd.read_pickle(train_data_path)
	train_label_df = pd.read_pickle(train_label_path)

	test_data_df = pd.read_pickle(test_data_path)
	test_label_df = pd.read_pickle(test_label_path)

	x_train = train_data_df.values
	y_train = train_label_df.values
	x_test = test_data_df.values
	y_test = test_label_df.values

	return x_train, y_train, x_test, y_test, train_data_df, train_label_df, test_data_df, test_label_df


#def run():
if __name__ == '__main__':
	x_train, y_train, x_test, y_test, x_train_df, y_train_df, x_test_df, y_test_df = create_dataset()

	# net = ADRNet(x_train.shape, y_train.shape)
	# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	#
	# print "Model creating..."
	# net.create_model()
	# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	#
	# print "Model start to train..."
	# net.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1)
	# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	
	
	net = load_model(checkpoint_path)
	print "Model evaluation..."
	accuracy = net.evaluate(x_test, y_test)[1]
	y_probs = net.predict(x_test)
	scores = roc_auc_score(y_test, y_probs, average=None)
	print "MEAN AUC:", np.mean(scores)
	
#if __name__ == '__main__':
#	run()