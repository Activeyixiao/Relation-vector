from argparse import ArgumentParser
import pickle
import gensim
import os
import sys
from collections import defaultdict
import numpy as np
import scipy

def load_embeddings(embeddings_path):
	print('Loading embeddings:',embeddings_path)
	try:
		model=gensim.models.Word2Vec.load(embeddings_path)
	except:
		try:
			model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
		except:
			try:
				model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
			except:
				sys.exit('Couldnt load embeddings')
	vocab=model.index2word
	dims=model.__getitem__(vocab[0]).shape[0]
	vocab=set(vocab)
	return model,vocab,dims

def insert_relation(relation_vectors,dims):
	if context_type == 'left1':
		relation_vector[0:dims,]=local_vector
	elif context_type == 'mid1':
		relation_vector[dims:dims*2]=local_vector
	elif context_type == 'right1':
		relation_vector[dims*2:dims*3]=local_vector
	elif context_type == 'left2':
		relation_vector[dims*3:dims*4]=local_vector
	elif context_type == 'mid2':
		relation_vector[dims*4:dims*5]=local_vector
	elif context_type == 'right2':
		relation_vector[dims*5:]=local_vector
	else:
		sys.exit('ERROR: Unknown context type')

def aggr_sent_bow(rel_type,relation,context_tokens,vocabwords,dimswords):
	# X, rels2ids and counts are global
	svec=np.zeros(dimswords)
	token_counter=0
	for context_word in context_tokens:
		if context_word in vocabwords:
			svec+=modelwords[context_word]
			token_counter+=1
	if token_counter > 0:
		if rel_type == 'left1':
			X[rels2ids[relation],0:dimswords]+=svec/token_counter
		if rel_type == 'mid1':
			X[rels2ids[relation],dimswords:dimswords*2]+=svec/token_counter
		if rel_type == 'right1':
			X[rels2ids[relation],dimswords*2:dimswords*3]+=svec/token_counter
		if rel_type == 'left2':
			X[rels2ids[relation],dimswords*3:dimswords*4]+=svec/token_counter
		if rel_type == 'mid2':
			X[rels2ids[relation],dimswords*4:dimswords*5]+=svec/token_counter
		if rel_type == 'right2':
			X[rels2ids[relation],dimswords*5:dimswords*6]+=svec/token_counter	
		counts[relation][rel_type]+=1

if __name__ == '__main__':

	parser = ArgumentParser()
	
	parser.add_argument('-wv','--word-vectors', help='Pretrained word vectors to vectorize relations', required=True)
	parser.add_argument('-p','--pairs-file', help='Pairs file, useful only for saving vectors in the same order', required=False)
	parser.add_argument('-b','--build-folder', help='Folder where relation vectors will be saved', required=True)

	args = parser.parse_args()

	embeddings_path = args.word_vectors
	embeddings_name=embeddings_path.split('/')[-1]
	pairs_file = args.pairs_file
	output_folder = args.build_folder

	modelwords,vocabwords,dimswords=load_embeddings(embeddings_path)

	all_relations_files=[os.path.join(args.build_folder,infile) for infile in os.listdir(args.build_folder) if infile.endswith('.ctx')]
	print('There are ',len(all_relations_files),' context files')

	print('Loading selected pairs')
	sorted_pairs = []
	for line in open(pairs_file,'r'):
		cols=line.strip().split('\t')[0]
		wp = cols.split('__')
		cent,contxt = wp[0],wp[1]
		sorted_pairs.append((cent,contxt))

	print('Loaded ',len(sorted_pairs),'pairs from ConceptNet')

	print('Getting relation->relation_id mapping')
	rels2ids={}
	relation_counter=0
	for idx,relation_file in enumerate(all_relations_files):
		print('Processing context file numb ',idx+1,' of ',len(all_relations_files))
		with open(relation_file,'r') as f:
			for line in f:
				cols=line.strip().split('\t')
				if len(cols) == 4:
					x,y,rel_type,context_tokens=cols[0],cols[1],cols[2],cols[3].split()
					r=(x,y)
					if not r in rels2ids:
						rels2ids[r]=relation_counter
						relation_counter+=1
	
	X=np.zeros((len(rels2ids), dimswords*6))
	print('Learning a ',X.shape,' embedding matrix')
	counts=defaultdict(lambda : defaultdict(int))
	for filedix,relation_file in enumerate(all_relations_files):
		with open(relation_file,'r') as f:
			line_idx=0
			for line in f:
				if line_idx % 10000 == 0:
						print('Aggregated vectors of ',line_idx,' lines | Done ',filedix, 'files of ',len(all_relations_files))
				cols=line.strip().split('\t')
				if len(cols) == 4:
					# context_tokens assumed to be tokenized already
					x,y,rel_type,context_tokens=cols[0],cols[1],cols[2],cols[3].split()
					relation=(x,y)
					if relation in rels2ids:
						aggr_sent_bow(rel_type,relation,context_tokens,vocabwords,dimswords)
				line_idx+=1

	for idx,relation in enumerate(rels2ids):
		if idx % 1000 == 0:
			print('Averaged ',idx,' of ',len(rels2ids),' relations')
		for ctx_type in counts[relation]:
			if ctx_type == 'left1':
				X[rels2ids[relation],0:dimswords]=X[rels2ids[relation],0:dimswords]/counts[relation][ctx_type]
			if ctx_type == 'mid1':
				X[rels2ids[relation],dimswords:dimswords*2]=X[rels2ids[relation],dimswords:dimswords*2]/counts[relation][ctx_type]
			if ctx_type == 'right1':
				X[rels2ids[relation],dimswords*2:dimswords*3]=X[rels2ids[relation],dimswords*2:dimswords*3]/counts[relation][ctx_type]
			if ctx_type == 'left2':
				X[rels2ids[relation],dimswords*3:dimswords*4]=X[rels2ids[relation],dimswords*3:dimswords*4]/counts[relation][ctx_type]
			if ctx_type == 'mid2':
				X[rels2ids[relation],dimswords*4:dimswords*5]=X[rels2ids[relation],dimswords*4:dimswords*5]/counts[relation][ctx_type]
			if ctx_type == 'right2':
				X[rels2ids[relation],dimswords*5:dimswords*6]=X[rels2ids[relation],dimswords*5:dimswords*6]/counts[relation][ctx_type]

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	rel_counter=0
	error_counter=0
	output_file=os.path.join(output_folder,'relation_vectors__pretrainedwv='+embeddings_name+'.vec')
	with open(output_file,'w') as outf:
		for a,b in sorted_pairs:
			relation=(a,b)
			if relation in rels2ids:
				relation_toprint=(a,b)
				vec=X[rels2ids[relation]]
				outf.write('__'.join(relation_toprint)+' '+'\t'.join([str(k) for k in vec])+'\n')
			else:
				error_counter+=1
			if rel_counter % 100000 == 0:
				print('Saved ',rel_counter,' relation vectors out of ',len(sorted_pairs),' relations')
				print('So far missed ',error_counter,' relations')
			rel_counter+=1