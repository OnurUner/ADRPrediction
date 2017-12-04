# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from lincsdataset import cross_validation
from adrnet import AdrNet
from sklearn.metrics import roc_auc_score


phase = "phase1"


def save_mean_auc(auc_scores, save_path):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
		
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	f = open(save_path, "w")
	for i in sorted_means:
		f.write(str(i) + " " + str(mean_auc_scores[i]) + "\n")
	f.close()


def calc_auc_without_drug(y_prob, y_true):
	y_pred = []
	y_target = []
	for drug_id in y_prob:
		y_pred.append(np.vstack(y_prob[drug_id]))
		y_target.append(np.vstack([y_true[drug_id]]*len(y_prob[drug_id])))
	
	y_pred = np.vstack(y_pred)
	y_target = np.vstack(y_target)
	
	auc_scores = dict()
	for column_index in range(y_pred.shape[1]):
		t_y = y_target[:,column_index]
		p_y = y_pred[:,column_index]
		if np.sum(t_y) == 0:
			continue		
		score = roc_auc_score(t_y, p_y, average=None)
		auc_scores[column_index] = score
	return np.mean(auc_scores.values())
	

def calc_auc_scores(y_prob, y_true):
	y_pred = []
	y_target = []
	for drug_id in y_prob:
		drug_pred = np.vstack(y_prob[drug_id])
		drug_pred = np.max(drug_pred, axis=0)
		y_pred.append(drug_pred)
		y_target.append(y_true[drug_id])
	
	y_pred = np.vstack(y_pred)
	y_target = np.vstack(y_target)
	
	auc_scores = dict()
	for column_index in range(y_pred.shape[1]):
		t_y = y_target[:,column_index]
		p_y = y_pred[:,column_index]
		if np.sum(t_y) == 0:
			continue		
		score = roc_auc_score(t_y, p_y, average=None)
		auc_scores[column_index] = score
	return np.mean(auc_scores.values())


if __name__ == '__main__':
	epochs = 20
	learning_rate = 0.1
	
	mean_auc_scores_with_drug = []
	mean_auc_scores_without_drug = []
	
	for fold_i, (train_dataset, test_dataset) in enumerate(cross_validation(phase, validation_rate=0.33, only_landmark_genes=False)):
		print "Cross Validation Fold:", fold_i
		train_loader = DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)
		validation_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
		test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

		nnet = AdrNet(train_dataset.x_data.shape[1], train_dataset.labels.shape[1])
		optimizer = optim.SGD(nnet.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
		# optimizer = optim.Adam(nnet.parameters(), lr=0.001)
		criterion = nn.BCELoss(weight=train_dataset.calc_class_weights(), size_average=True)
		# criterion = nn.MultiLabelSoftMarginLoss(weight=train_dataset.calc_class_weights(), size_average=True)
		# criterion = nn.BCELoss()
		nnet.fit(train_loader, epochs, criterion, optimizer, validation_loader=validation_loader, verbose=2)
#		nnet.plot_loss_graph("Loss Graph Fold " + str(fold_i))
#		nnet.plot_acc_graph("Accuracy Graph Fold " + str(fold_i))
		y_prob, y_true = nnet.test(test_loader)
		mean_auc_scores_with_drug.append(calc_auc_scores(y_prob, y_true))
		mean_auc_scores_without_drug.append(calc_auc_without_drug(y_prob, y_true))
	print "with drug:", np.mean(mean_auc_scores_with_drug)
	print "without drug:", np.mean(mean_auc_scores_without_drug)

