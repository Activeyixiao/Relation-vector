import numpy as np
import pandas as pd

file = pd.read_csv('relaion_extraction.csv')
file.columns = ['relation', 'head', 'tail']

new_head = [i.split('/') for i in file['head']]
new_tail = [i.split('/') for i in file['tail']]
file['head'] = new_head
file['tail'] = new_tail

head_en = [[i[2]=='en'] for i in file['head']]
tail_en = [[i[2]=='en'] for i in file['tail']]

en_only = np.logical_and(head_en, tail_en)
en_file = file[en_only]

orgin_head = ['/'.join(i) for i in en_file['head']]
orgin_tail = ['/'.join(i) for i in en_file['tail']]

en_file['head'] = orgin_head
en_file['tail'] = orgin_tail
en_file.to_csv('en_relation.csv')

df = pd.read_csv('en_relation.csv')

file['relation'] = file['relation'].apply(lambda x: x.split('/')[2])
file['head'] = file['head'].apply(lambda x: x.split('/')[3])
file['tail'] = file['tail'].apply(lambda x: x.split('/')[3])

file['words_pair'] = file[['head', 'tail']].apply(lambda x: '__'.join(x), axis=1)
new_file = file[['relation','words_pair']]
