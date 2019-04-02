import numpy as np
import pandas as pd

file = pd.read_csv('relation_wpair.csv')
f = open('relation_vectors__pretrainedwv=GoogleNews-vectors-negative300.bin.vec')
corpus = f.readlines()
corpus['wp'], corpus['vector'] = corpus['Vector'].str.split(' ', 1).str
corpus_ls = list(corpus['wp'])
corpus_ls.remove('341947')
file_ls = list(file['words_pair'])
intersect_ls = list(set(file_ls) & set(corpus_ls))
new_file = file.loc[file['words_pair'].isin(intersect_ls)]
new_corpus = corpus.loc[corpus['wp'].isin(intersect_ls)]
new_corpus = new_corpus.set_index('wp')
new_corpus = new_corpus.drop(['Vector'],axis = 1)
column_count = len(set(new_file.relation))
row_count = len(intersect_ls)
wp_matrix = pd.DataFrame(np.zeros((row_count, column_count), dtype=np.int32),columns = list(set(new_file.relation)), index=intersect_ls)
wp_groups = new_file.groupby('relation')['words_pair'].apply(list)
group_list = wp_groups.index.tolist()
count = 0
for r in wp_groups:
    for token in r:
        wp_matrix.at[token, group_list[count]] = 1
    print('finish one type of relation')
    count += 1

result = pd.concat([wp_matrix, new_corpus], axis=1, sort=False)
result.to_csv('concatenate_matrix')