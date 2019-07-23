import numpy as np
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
from collections import defaultdict
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import regularizers,objectives
import gensim
import math


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
	vocab=model.vocab
	dims=model.vector_size
	vocab=set(vocab)
	return model,vocab,dims

def autoencoder_50d(x,words, y1, y2):

	model1_in = Input(shape=(1837,),name='orig_rel')
	model1_out = Dense(50, activation='relu', name='compressed')(model1_in)
	model1 = Model(model1_in, model1_out)
	model2_in = Input(shape=(600,),name='orig_words_in')
	model2_out = Dense(600, activation='relu', name='orig_words_out')(model2_in)
	model2 = Model(model2_in, model2_out)
	concatenated = concatenate([model1_out, model2_in],name='shared')

	con_vec = Dense(37, activation='softmax',name='conceptnet_vec')(concatenated)
	cor_vec = Dense(1800, activation='relu',name='corpus_vec')(concatenated)

	autoencoder = Model(input = [model1_in, model2_in], output = [con_vec, cor_vec])
	autoencoder.compile(optimizer=params['optimizer'],
						loss={'conceptnet_vec':'categorical_crossentropy','corpus_vec': 'mean_squared_error'},
						loss_weights={'conceptnet_vec': 1.0,'corpus_vec': 0.05})
	autoencoder.fit(x=[x, words],
					y={'conceptnet_vec':y1, 'corpus_vec':y2},
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True)
	return autoencoder

def build_classifier(rt):

	x_train_input = []
	x_train_kg = []
	x_train_corpus = []
	x_words = []
	y_train = []
	
	index = sorted(set(kg_selected.relation)).index(rt)
	for r in new_relvocab_train:
		if rt in D_rt[r]:
			y_train.append(1)
		else:
			y_train.append(0)

		rv = list(modelrels_train[r])
		del rv[index]
		x_train_input.append(np.array(rv))
		x_train_corpus.append(np.array(rv[37:]))
		x_train_kg.append(np.array(rv[:37]))

		w1 = r.split('__')[0]
		w2 = r.split('__')[1]
		if w1 in wordvocab:
			v1 = modelwords[w1]
		else:
			v1 = np.zeros(worddims)
		if w2 in wordvocab:
			v2 = modelwords[w2]
		else:
			v2 = np.zeros(worddims)
		c = np.concatenate([v1,v2])
		x_words.append(c)

	X_train_input = np.array(x_train_input)
	X_train_kg = np.array(x_train_kg)
	X_train_corpus = np.array(x_train_corpus)
	X_words = np.array(x_words)

	my_model = autoencoder_50d(X_train_input, X_words, X_train_kg, X_train_corpus)
	intermediate_layer_model = Model(inputs=my_model.input, outputs=my_model.get_layer('compressed').output)
	X_train=intermediate_layer_model.predict([X_train_input,X_words])
	Y_train = np.array(y_train)

	source = Input(shape=(50,),name='orig_rel')
	hidden = Dense(50, activation= 'tanh')(source)
	target = Dense(1, activation='sigmoid')(hidden)
	model = Model(input = source, output = target)
	model.compile(optimizer=params['optimizer'],loss='binary_crossentropy')
	model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, shuffle=True)
	return model


def prepare_test_vectors(rt):
	x_test_input = []
	x_test_kg = []
	x_test_corpus = []
	x_test_words = []
	y_test = []

	index = sorted(set(kg_selected.relation)).index(rt)
	for r in relvocab_test:
		if rt in D_rt[r]:
			y_test.append(1)
		else:
			y_test.append(0)
		rv = list(modelrels_test[r])
		del rv[index]
		x_test_input.append(np.array(rv))
		x_test_corpus.append(np.array(rv[37:]))
		x_test_kg.append(np.array(rv[:37]))

		w1 = r.split('__')[0]
		w2 = r.split('__')[1]
		if w1 in wordvocab:
			v1 = modelwords[w1]
		else:
			v1 = np.zeros(worddims)
		if w2 in wordvocab:
			v2 = modelwords[w2]
		else:
			v2 = np.zeros(worddims)
		c = np.concatenate([v1,v2])
		x_test_words.append(c)

	X_test_input = np.array(x_test_input)
	X_test_kg = np.array(x_test_kg)
	X_test_corpus = np.array(x_test_corpus)
	X_test_words = np.array(x_test_words)

	test_model = autoencoder_50d(X_test_input, X_test_words, X_test_kg, X_test_corpus)
	intermediate_layer_model = Model(inputs=test_model.input, outputs=test_model.get_layer('compressed').output)
	X_test=intermediate_layer_model.predict([X_test_input,X_test_words])
	Y_test = np.array(y_test)
	return X_test, Y_test

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-rv1','--relation-embeddings-train', help='Relation vectors', required=True)
	parser.add_argument('-rv2','--relation-embeddings-test', help='Relation vectors', required=True)
	parser.add_argument('-wv','--word-embeddings', help='Word embeddings', required=True)
	parser.add_argument('-kg','--knowledge-graph', help='Knowledge graph', required=True)
	parser.add_argument('-b','--build-folder', help='Output path where compressed embeddings are saved', required=True)

	args = parser.parse_args()
	path=args.build_folder
	if not os.path.exists(path):
		os.makedirs(path)
	
	modelrels_train,relvocab_train,reldims_train=load_embeddings(args.relation_embeddings_train)
	modelrels_test,relvocab_test,reldims_test=load_embeddings(args.relation_embeddings_test)
	modelwords,wordvocab,worddims=load_embeddings(args.word_embeddings)	
	

	new_relvocab_train = []
	for wp in relvocab_train:
		w1,w2 = wp.split('__')
		if w1 != 'nan' and w2 != 'nan':
			new_relvocab_train.append(wp)
	wp_ls = list(new_relvocab_train) + list(relvocab_test)

	kg = pd.read_csv('KG/en_words_pair.csv')
	kg['wp'] = kg['head'] +'__'+ kg['tail']
	kg_selected = kg[kg['wp'].isin(wp_ls)]
	kg_groups = kg_selected.groupby('wp')['relation'].apply(list)
	D_rt = kg_groups.T.to_dict()
	print('obtaion the dictionary D_rt in which each key is a wp and the values is a relation type')

	set_of_relations = ['DerivedFrom','FormOf','HasContext','AtLocation',
	'RelatedTo','EtymologicallyRelatedTo','Synonym','dbpedia','DistinctFrom','PartOf','IsA','SimilarTo','MannerOf','Antonym']
	
	epochs=20
	batch_size=1000
	l2_param=1e-08
	output_path=os.path.join(path,'50d_classifers_evalution'+'.txt') 
	params={'regularizer':l2_param,'epochs':epochs,'optimizer':'adagrad','loss':'mean_squared_error'}	

	from keras.layers import *
	from keras.models import *

	with open(output_path,'w') as outfile:
		for relation_type in set_of_relations:
			classifier = build_classifier(relation_type)
			test_vectors,test_labels = prepare_test_vectors(relation_type)
			predictions = classifier.predict(test_vectors)
			predictions_round = np.round(predictions)
			all_count = len(test_labels)

			accuracy_count = 0
			for i in range(all_count):
				if test_labels[i] == predictions_round[i]:
					accuracy_count += 1
			accurancy = accuracy_count/all_count
			print('the overall accurancy rate of ',relation_type,' is ',accurancy)

			true_positives = 0
			for i in range(all_count):
				if test_labels[i]==1:
					if predictions_round[i]==1:
						true_positives += 1
			print(true_positives, 'true positives')
						
			predicted_positives = np.sum(predictions_round)
			print(predicted_positives,' predicted positives')
			all_positives = np.sum(test_labels)
			print(all_positives, ' positives in test set')

			epsilon = 0.0000001
			precision_rate = true_positives/(predicted_positives + epsilon)
			recall_rate = true_positives/(all_positives+epsilon)
			f1_rate = 2*((recall_rate*precision_rate)/(recall_rate+precision_rate+epsilon))
			print(precision_rate)
			print(recall_rate)
			print(f1_rate)
			outfile.write(str(relation_type)+' '+str(precision_rate)+' '+str(recall_rate)+' '+str(f1_rate)+'\n')