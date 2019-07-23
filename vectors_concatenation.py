import os
import sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
from collections import ChainMap

def extract_wp(document):
	wp_corpus_list = []
	with open(document) as input_f:
		for line in input_f:
			wp = line.strip().split()[0]
			wp_corpus_list.append(wp)
	return wp_corpus_list


def dict_from_vec(vec):
	D = defaultdict(list)
	with open(vec) as inputf:
		for line in inputf:
			line = line.strip().split(' ')
			wp = line[0]
			vectors = line[1]
			D[wp] = vectors
	return D


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-b','--build-folder', help='Build folder', required=True)
	parser.add_argument('-kg','--knowledge-graph', help='The knowledge-graph selected', required=True)
	parser.add_argument('-r','--relation-vectors',help='The word pair vectors from corpus', required=True)

	args = parser.parse_args()

	kg = pd.read_csv(args.knowledge_graph)
	kg['words_pair'] = kg['head'].map(str) + '__' + kg['tail'].map(str)

	wp_list = extract_wp(args.relation_vectors)
	print('wp_corpus vectors contains',len(wp_list),'words pairs')

	kg_selected = kg[kg['words_pair'].isin(wp_list)]
	print('loads',len(kg_selected['words_pair'].tolist()),'words pairs')
    
	row_num = len(set(kg_selected['words_pair']))
	column_num = len(set(kg_selected['relation']))
	binary_matrix = pd.DataFrame(np.zeros((row_num, column_num), dtype=np.int32),columns = sorted(set(kg_selected.relation)), index=set(kg_selected.words_pair))

	kg_groups = kg_selected.groupby('relation')['words_pair'].apply(list)
	group_list = kg_groups.index.tolist()

	r_count = 0
	for r in kg_groups:
		for wp in r:
			binary_matrix.at[wp, group_list[r_count]] = 1
		print('finish', r_count+1, 'of', len(group_list),'relation types')
		r_count += 1

	D_kg = defaultdict(list)
	for i in list(set(kg_selected.words_pair)):
    	i_int_ls = binary_matrix.loc[i]
    	i_str_ls = [str(i) for i in i_int_ls] 
    	D_kg[i] = i_str_ls

	print('wp_knowledge_graph contains',len(D_kg.keys()),'words_pair')

	D_corpus = dict_from_vec(args.relation_vectors)

	try:
		del D_corpus['515480']
	except KeyError:
		print("Key 'testing' not found")

	for key, value in D_corpus.items():
    	D_corpus[key] = '\t'.join(D_kg[key]) +'\t'+ value

	print('The dictionary contains',len(D_corpus.keys()),'words_pair')

	output_file=os.path.join(args.build_folder,'relation_vectors_1838d.vec')

	numb_wp = len(D_corpus.keys())

	with open(output_file,'w') as outf:
		outf.write(str(len(D_corpus.keys()))+' '+str(1838)+'\n')
		for k,v in D_corpus.items():
			outf.write(k +' '+ v + '\n')