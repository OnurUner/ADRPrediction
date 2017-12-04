# -*- coding: utf-8 -*-
import pickle
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from math import ceil
import sys
sys.path.append('../../data/')

import dataset_generator


class LincsDataset(Dataset):
	
	def __init__(self, dataset, fold):		
		self.dataset = dataset
		self.fold = fold
		self.x_data = self.get_data()
		self.labels = dataset["label_df"]
		self.len = self.x_data.shape[0]
		
	def get_data(self):
		drug_data = self.dataset["drug_expression"]
		self.data_ids = []
		drug_ids = drug_data.keys()
		data_list = []
		for i in self.fold:
			data = drug_data[drug_ids[i]]
			ids = [drug_ids[i]]*len(data)
			
			self.data_ids += ids
			data_list += data
		
		return torch.from_numpy(np.vstack(data_list))
			
	def __len__(self):
		return self.len
	
	def __getitem__(self, index):
		x = self.x_data[index]
		drug_id = self.data_ids[index]
		return x, torch.from_numpy(self.labels.loc[drug_id, :].values).type(torch.FloatTensor)
	
	def get_drug_id(self, index):
		return self.data_ids[index]

	def calc_class_weights(self):
		Y = self.labels.values
		neg_weights = []
		pos_weights = []
		weights = []
		for i in range(Y.shape[1]):
			counts = np.bincount(Y[:,i].astype(int))
			num_neg = float(counts[0])
			num_pos = float(counts[1])
			neg_weights.append(1.0/num_neg)
			pos_weights.append(1.0/num_pos)
			weights.append(num_neg / num_pos)
		return Variable(torch.from_numpy(np.array(weights)).type(torch.FloatTensor), requires_grad=False).cuda()


def load_dataset(path):
	dataset = None
	with open(path, 'rb') as handle:
		dataset = pickle.load(handle)
	return dataset


def generate_folds(drug_list, validation_rate):
	sample_order = range(len(drug_list))
	fold_len = int(ceil(len(sample_order)*validation_rate))
	test_folds = []
	train_folds = []
	begin_index = 0
	end_index = begin_index + fold_len
	while end_index <= len(sample_order):
		train_set = sample_order[0:begin_index] + sample_order[end_index:]
		test_set = sample_order[begin_index:end_index]
		
		train_folds.append(train_set)
		test_folds.append(test_set)
		begin_index += fold_len
		end_index += fold_len
	return train_folds, test_folds


def cross_validation(phase, validation_rate, only_landmark_genes=False):
	dataset_path = dataset_generator.get_all_data_path(phase, only_landmark_genes=only_landmark_genes)
	print dataset_path
	dataset = load_dataset(dataset_path)
	drug_data = dataset["drug_expression"]
	drug_list = drug_data.keys()
	train_folds, test_folds = generate_folds(drug_list, validation_rate)
	for fold_i in range(len(train_folds)):
		train_dataset = LincsDataset(dataset, train_folds[fold_i])
		validation_dataset = LincsDataset(dataset, test_folds[fold_i])
		yield train_dataset, validation_dataset
