import os
import sys
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import pandas as pd
import operator

def extract_pairs(document):
	pair_counter = defaultdict(int)

	linecount = 0
	for line in open(document,'r'):
		line = line.strip()
		linecount += 1
		if linecount % 10000 == 0:
			print('processing of', linecount, 'in', document)
		line_idx=defaultdict(int)
		tokens = line.split()
		wp_pairs=[
			(tokens[i] +'__'+ tokens[j]) 
			for i in range(len(tokens)) 
			for j in range(len(tokens)) 
			if 0 < abs(j-i) <= 10
		]
		for wp in wp_pairs:
			if wp in wp_set:
				pair_counter[wp] +=1 

	outf = open(document +'_wp_freq'+'.txt','w')
	ordered_frequencies = sorted(pair_counter.items(), key=lambda x:x[1], reverse=True)
	for a,b in ordered_frequencies:
		outf.write(a+'\t'+str(b)+'\n')
	outf.close()

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-c','--corpus-file', help='Corpus file', required=True)
	parser.add_argument('-b','--build-folder', help='Build folder', required=True)
	parser.add_argument('-t','--threads', help='Threads to use', required=False)
	parser.add_argument('-win','--window-size', help='Window size', required=True)
	parser.add_argument('-sw','--stopwords', help='List of stopwords (one per line)', required=True)
	parser.add_argument('-kg','--knowledge-graph', help='The knowledge-graph selected', required=True)

	args = parser.parse_args()
	workers=mp.cpu_count()
	print('Counting lines in original corpus')
	linecount=0
	with open(args.corpus_file) as f:
		for line in f:
			linecount+=1

	if not os.path.exists(args.build_folder):
		os.makedirs(args.build_folder)

	sents_per_split=round(linecount/workers)

	print('Source corpus has ',linecount,' lines')
	print('Splitting original corpus in ',workers,' files of ~',sents_per_split,' lines')
	linecount=0
	splitcount=0
	outf=open(os.path.join(args.build_folder,'split_'+str(splitcount))+'.txt','w')
	with open(args.corpus_file,'r') as f:
		for line in f:
			linecount+=1
			outf.write(line)
			if linecount % sents_per_split == 0:
				outf.close()
				splitcount+=1
				outf=open(os.path.join(args.build_folder,'split_'+str(splitcount)+'.txt'),'w')
				print('Saved split numb: ',splitcount,' of ',workers)
	
	
	kg = pd.read_csv(args.knowledge_graph)
	kg['words_pair'] = kg['head'].map(str) + '__' + kg['tail'].map(str)
	wp_list = kg['words_pair'].tolist()

	single_wp_list = []
	for wp in wp_list:
		w1,w2 = wp.split('__')[0],wp.split('__')[1]
		if '_' not in w1:
			if '_' not in w2:
				single_wp_list.append(wp)

	wp_set = set(single_wp_list)
	print('loaded',len(wp_set),'words_pairs')

	splits=[os.path.join(args.build_folder,i) for i in os.listdir(args.build_folder) if i.startswith('split')]
	P = mp.Pool(processes=workers)
	P.map(extract_pairs, splits)
	P.close()

	stopwords=[]
	if args.stopwords:
		stopwords=[line.strip() for line in open(args.stopwords)]
		print('Loaded ',len(stopwords),' stopwords')

	splited_wp_files = [os.path.join(args.build_folder, i) for i in os.listdir(args.build_folder) if i.startswith('split') and i.endswith('wp_freq.txt')]
	
	wp_freq_count = defaultdict(int)
	
	with open(os.path.join(args.build_folder,'wp_freq_count.txt'), 'w') as outfile:
		for fname in splited_wp_files:
			with open(fname,'r') as infile:
				for line in infile:
					line = line.strip()
					wp,freq = line.split('\t')[0], line.split('\t')[1]
					wp_freq_count[wp] += int(freq)
		
		filtered_wp_freq = {k: v for k, v in wp_freq_count.items() if v >= 5}
		filtered_wp_freq_dict = {k: v for k, v in filtered_wp_freq.items() if k.split('__')[0] not in stopwords and k.split('__')[1] not in stopwords}
		print(len(filtered_wp_freq_dict.keys()),'words pairs were collected')
		ordered_filtered_wp_freq_dict = sorted(filtered_wp_freq_dict.items(), key=lambda x: x[1], reverse=True)
		for k,v in ordered_filtered_wp_freq_dict:
			outfile.write(k + '\t' +str(v) + '\n')
