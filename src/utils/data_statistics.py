# -*- coding: utf-8 -*-
from cmapPy.pandasGEXpress import parse
from os import path as osp
import numpy as np
from numpy import genfromtxt
import pandas as pd
from math import floor
import pickle


# --------- LOAD PATHS ---------
working_dir = osp.dirname(osp.abspath(__file__))
data_path = osp.join(working_dir, '..', '..', 'data')

# phase 1
p1_gctx_path = osp.join(data_path, 'phase1', 'GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx')
p1_sig_path = osp.join(data_path, 'phase1', 'GSE92742_Broad_LINCS_sig_info.txt')
p1_gene_path = osp.join(data_path, 'phase1', 'GSE92742_Broad_LINCS_gene_info.txt')
p1_pert_path = osp.join(data_path, 'phase1', 'GSE92742_Broad_LINCS_pert_info.txt')

# phase 2
p2_gctx_path = osp.join(data_path, 'phase2', 'GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx')
p2_sig_path = osp.join(data_path, 'phase2', 'GSE70138_Broad_LINCS_sig_info.txt')
p2_gene_path = osp.join(data_path, 'phase2', 'GSE92742_Broad_LINCS_gene_info.txt')
p2_pert_path = osp.join(data_path, 'phase2', 'GSE70138_Broad_LINCS_pert_info.txt')

label_path = osp.join(data_path, 'SIDER_PTs.csv')


if __name__ == '__main__':
#	p1_expression_df = parse(p1_gctx_path).data_df.transpose()
#	p1_signatures_df = pd.DataFrame.from_csv(p1_sig_path, sep='\t')
#	p1_gene_info_df = pd.DataFrame.from_csv(p1_gene_path, sep='\t')
#	p1_pert_info_df = pd.DataFrame.from_csv(p1_pert_path, sep='\t')
#
#	p2_expression_df = parse(p2_gctx_path).data_df.transpose()
#	p2_signatures_df = pd.DataFrame.from_csv(p2_sig_path, sep='\t')
#	p2_gene_info_df = pd.DataFrame.from_csv(p2_gene_path, sep='\t')
#	p2_pert_info_df = pd.DataFrame.from_csv(p2_pert_path, sep='\t')
#
#	label_df = pd.DataFrame.from_csv(label_path)
#	
#	p1_sig_list = p1_signatures_df.index.values.tolist()
#	p2_sig_list = p2_signatures_df.index.values.tolist()
#	
#	i = 0
#	for p2_sig in p2_sig_list:
#		if p2_sig in p1_sig_list:
#			print p2_sig
#			i+=1
#	print i
	
	p1_experiment_list = []
	for label in p1_set:
		keys = p1_signatures_df[p1_signatures_df["pert_id"] == label].index.values
		for k in keys:
			dose = p1_signatures_df.loc[k, "pert_idose"]
			time = p1_signatures_df.loc[k, "pert_itime"]
			cell = p1_signatures_df.loc[k, "cell_id"]
			name = label+"|"+dose+"|"+time+"|"+cell
			p1_experiment_list.append(name)
		
	p2_experiment_list = []
	for label in p2_set:
		keys = p2_signatures_df[p2_signatures_df["pert_id"] == label].index.values
		for k in keys:
			dose = p2_signatures_df.loc[k, "pert_idose"]
			time = p2_signatures_df.loc[k, "pert_itime"]
			cell = p2_signatures_df.loc[k, "cell_id"]
			name = label+"|"+dose+"|"+time+"|"+cell
			p2_experiment_list.append(name)