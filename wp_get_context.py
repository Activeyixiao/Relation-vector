import logging
from argparse import ArgumentParser
import os
import sys
from collections import defaultdict
import itertools
from datetime import datetime
import multiprocessing as mp

def ctx(corpus_path):
	### count lines
	print('Counting corpus lines')
	window_size = 10
	linecount=0
	for line in open(corpus_path,'r'):
		linecount+=1
	print('This file has ',linecount,' lines')
	### extract contexts
	corpus_filename=corpus_path.split('/')[-1]
	lc=0
	with open(os.path.join(output_folder,corpus_filename+'_MID_WINDOW='+str(MID_WINDOW)+'_SIDE_WINDOW='+str(SIDE_WINDOW)+'.ctx'),'w') as outf:
		for line in open(corpus_path,'r'):
			all_idx=defaultdict(int)
			tokens=line.strip().split()
			for idx,tok in enumerate(tokens):
				all_idx[tok] = idx

			wp_ls = [
				(tokens[i] +'__'+ tokens[j]) 
				for i in range(len(tokens)) 
				for j in range(len(tokens)) 
				if 0 < abs(j-i) <= window_size
			]
			for wp in wp_ls:
				if wp in conceptnet_set:
					source_word = wp.split('__')[0]
					target_word = wp.split('__')[1]
					sourceidx=all_idx[source_word]
					targetidx=all_idx[target_word]
			
					if sourceidx < targetidx:
						left_ctx=tuple(tokens[max(0,sourceidx-SIDE_WINDOW):sourceidx])
						mid_ctx=tuple(tokens[sourceidx:min(targetidx+1,len(tokens))][1:-1])
						right_ctx=tuple(tokens[targetidx:][1:SIDE_WINDOW])
									
						outf.write(source_word+'\t'+target_word+'\t'+'left1\t'+' '.join(left_ctx)+'\n')
						outf.write(source_word+'\t'+target_word+'\t'+'mid1\t'+' '.join(mid_ctx)+'\n')
						outf.write(source_word+'\t'+target_word+'\t'+'right1\t'+' '.join(right_ctx)+'\n')
					else:
						left_ctx2=tuple(tokens[max(0,targetidx-SIDE_WINDOW):targetidx])
						mid_ctx2=tuple(tokens[targetidx:min(sourceidx+1,len(tokens))][1:-1])
						right_ctx2=tuple(tokens[sourceidx:][1:SIDE_WINDOW])
						
						outf.write(source_word+'\t'+target_word+'\t'+'left2\t'+' '.join(left_ctx2)+'\n')
						outf.write(source_word+'\t'+target_word+'\t'+'mid2\t'+' '.join(mid_ctx2)+'\n')
						outf.write(source_word+'\t'+target_word+'\t'+'right2\t'+' '.join(right_ctx2)+'\n')
			lc+=1
			if lc % 1000 == 0:
				print('Done ',lc,' lines of ',linecount,' of file ',corpus_path,' | At time: ',datetime.now())


if __name__ == '__main__':

	parser = ArgumentParser() 
	parser.add_argument('-p','--pairs-file', help='Pairs file', required=True)
	parser.add_argument('-b','--build-folder', help='Folder where contexts will be saved', required=True)
	parser.add_argument('-mw','--mid-window', help='Mid word window (dont consider words more far apart)', required=True)
	parser.add_argument('-sw','--side-window', help='Side (left or right) window', required=True)

	args = parser.parse_args()

	pairs_file=args.pairs_file

	conceptnet_set = set([line.strip().split('\t')[0] for line in open(pairs_file,'r')])

	print('loading pairs')

	output_folder=args.build_folder

	splits=[os.path.join(args.build_folder,inf) for inf in os.listdir(args.build_folder) 
	if inf.startswith('split') 
	and inf.endswith('.txt') 
	and not 'wp' in inf]

	print('Processing files:')
	for i in splits:
		print(i)


	MID_WINDOW=int(args.mid_window)
	SIDE_WINDOW=int(args.side_window)

	workers=mp.cpu_count()
	p = mp.Pool(processes=workers)
	p.map(ctx,splits)
	p.close()