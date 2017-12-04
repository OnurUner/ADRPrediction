#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from cmapPy.pandasGEXpress import parse
from os import path as osp
import numpy as np
from numpy import genfromtxt
import pandas as pd
from math import floor
import pickle

phase = 'phase1'
prune_count = 10
train_size = 0.75

# --------- LOAD PATHS ---------
working_dir = osp.dirname(osp.abspath(__file__))
data_path = osp.join(working_dir, '..', '..', 'data')

# phase 1
gctx_path = osp.join(data_path, phase, 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx')
sig_path = osp.join(data_path, phase, 'GSE92742_Broad_LINCS_sig_info.txt')
gene_path = osp.join(data_path, phase, 'GSE92742_Broad_LINCS_gene_info.txt')

# phase 2
#gctx_path = osp.join(data_path, phase, 'GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx')
#sig_path = osp.join(data_path, phase, 'GSE70138_Broad_LINCS_sig_info.txt')
#gene_path = osp.join(data_path, phase, 'GSE92742_Broad_LINCS_gene_info.txt')

label_path = osp.join(data_path, 'SIDER_PTs.csv')
# ------------------------------

# --------- SAVE PATHS ---------
pickle_all_data_path = osp.join(data_path, phase, 'drug_data_landmark_genes.pkl')
pickle_train_data_path = osp.join(data_path, phase, 'train_data.pkl')
pickle_train_label_path = osp.join(data_path, phase, 'train_label.pkl')

pickle_test_data_path = osp.join(data_path, phase, 'test_data.pkl')
pickle_test_label_path = osp.join(data_path, phase, 'test_label.pkl')
# ------------------------------

def get_all_data_path(phase, only_landmark_genes=False):
	if only_landmark_genes:
		filename = "drug_data_landmark_genes.pkl"
	else:
		filename = "drug_data.pkl"
	return osp.join(data_path, phase, filename)

def get_dataset():
	expression_df = parse(gctx_path).data_df.transpose()
	label_df = pd.DataFrame.from_csv(label_path)
	sig_info_df = pd.DataFrame.from_csv(sig_path, sep='\t')
	print "Expression shape:", expression_df.shape
	label_pert_ids = label_df.index.values
	del_sig_list = []
	# fill list with perturbation ids to be deleted
	for sig_id in expression_df.index.values:
		if sig_info_df.loc[sig_id, "pert_id"] not in label_pert_ids:
			del_sig_list.append(sig_id)
	# delete perturbations from data frame
	expression_df = expression_df.drop(del_sig_list)
	expression_df = expression_df.sample(frac=1)
	# collect signature side effect labels
	sig_label_data = []
	for sig_id in expression_df.index.values:
		pert_id = sig_info_df.loc[sig_id, "pert_id"]
		sig_labels = label_df.loc[pert_id]
		sig_label_data.append(sig_labels.values)

	sig_label_df = pd.DataFrame(data=sig_label_data, index=expression_df.index.values, columns=label_df.columns.values)
	
	print "Before column pruning y shape:", sig_label_df.shape
	adr_names = list(sig_label_df)
	del_adr_names = list()
	for adr_name in adr_names:
		if np.sum(sig_label_df.loc[:,adr_name].values) < prune_count:
			del_adr_names.append(adr_name)
		
	sig_label_df = sig_label_df.drop(del_adr_names, axis=1)
	print "After column pruning y shape:", sig_label_df.shape
	
	train_cnt = int(floor(expression_df.shape[0] * train_size))
	x_train = expression_df.iloc[0:train_cnt]
	y_train = sig_label_df.iloc[0:train_cnt]
	x_test = expression_df.iloc[train_cnt:]
	y_test = sig_label_df.iloc[train_cnt:]

	print "Before train/test column pruning y shape:", y_train.shape, y_test.shape
	adr_names = list(sig_label_df)
	del_adr_names = list()
	for adr_name in adr_names:
		if np.sum(y_train.loc[:,adr_name].values) < 1 or np.sum(y_test.loc[:,adr_name].values) < 1 :
			del_adr_names.append(adr_name)
	
	y_train = y_train.drop(del_adr_names, axis=1)
	y_test = y_test.drop(del_adr_names, axis=1)
	print "After train/test column pruning y shape:", y_train.shape, y_test.shape	
	return x_train, y_train, x_test, y_test

def get_drug_data(expression_df, signatures_df, label_df):
	drug_perturbations = dict()
	drug_expression = dict()
	
	for drug_id in label_df.index.values:
		drug_perturbations[drug_id] = []
		drug_expression[drug_id] = []
	
	for sig_id in signatures_df.index.values:
		pert_id = signatures_df.loc[sig_id, "pert_id"]
		if pert_id in label_df.index.values:
			drug_perturbations[pert_id].append(sig_id)
			
	for drug_id in label_df.index.values:
		for pert_id in drug_perturbations[drug_id]:
			expression = expression_df.loc[pert_id,:].values
			drug_expression[drug_id].append(expression)
			
	keys = drug_perturbations.keys()
	for drug_id in keys:
		if len(drug_perturbations[drug_id]) == 0:
			drug_perturbations.pop(drug_id, None)
			drug_expression.pop(drug_id, None)
	
	return drug_expression, drug_perturbations	

def get_drug_labels(drug_ids, label_df):
	delete_pert_ids = [pert_id for pert_id in label_df.index.tolist() if pert_id not in drug_ids]
	delete_adr_names = []
	print "Before column pruning y shape:", label_df.shape
	for adr_name in label_df.columns.tolist():
		if np.sum(label_df.loc[:,adr_name].values) < prune_count:
			delete_adr_names.append(adr_name)
	label_df = label_df.drop(delete_pert_ids)
	label_df = label_df.drop(delete_adr_names, axis=1)
	print "After column pruning y shape:", label_df.shape
	return label_df
	
def save_drug_dataset(only_landmark_genes=False):
	expression_df = parse(gctx_path).data_df.transpose()
	label_df = pd.DataFrame.from_csv(label_path)
	signatures_df = pd.DataFrame.from_csv(sig_path, sep='\t')
	gene_info_df = pd.DataFrame.from_csv(gene_path, sep='\t')
	print "Expression DataFrame Shape:", expression_df.shape
	if only_landmark_genes:
		del_gene_list = []
		landmark_genes = gene_info_df.loc[gene_info_df['pr_is_lm'] == 1].index.values.tolist()
		for gene_id in expression_df.columns.values.tolist():
			if gene_id not in landmark_genes:
				del_gene_list.append(gene_id)
		expression_df = expression_df.drop(del_gene_list, axis=1)
		print "Expression DataFrame Only Landmark Genes Shape:", expression_df.shape		
	
	drug_expression, drug_perturbations = get_drug_data(expression_df, signatures_df, label_df)
	label_df = get_drug_labels(drug_perturbations.keys(), label_df)
	dataset = dict()
	dataset["drug_expression"] = drug_expression
	dataset["drug_perturbations"] = drug_perturbations
	dataset["label_df"] = label_df
	with open(pickle_all_data_path, 'wb') as handle:
		pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return drug_perturbations, drug_expression, label_df
	
def save_dataset():
	x_train, y_train, x_test, y_test = get_dataset()
	x_train.to_pickle(pickle_train_data_path)
	y_train.to_pickle(pickle_train_label_path)
	
	x_test.to_pickle(pickle_test_data_path)
	y_test.to_pickle(pickle_test_label_path)


def load_dataset(phase):
	pkl_train_data_path = osp.join(data_path, phase, 'train_data.pkl')
	pkl_train_label_path = osp.join(data_path, phase, 'train_label.pkl')
	pkl_test_data_path = osp.join(data_path, phase, 'test_data.pkl')
	pkl_test_label_path = osp.join(data_path, phase, 'test_label.pkl')
	return pd.read_pickle(pkl_train_data_path), pd.read_pickle(pkl_train_label_path), pd.read_pickle(pkl_test_data_path), pd.read_pickle(pkl_test_label_path)


if __name__ == '__main__':
	save_drug_dataset(only_landmark_genes=True)
