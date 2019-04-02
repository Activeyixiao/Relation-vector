import numpy as np
import pandas as pd
import csv

# Get 1837d relation vectors as plain txt form
with open('test.bin.vec','w') as out_put:
    with open ('relation_vectors_1837d.csv','r') as in_put:
        [out_put.write(" ".join(row)) for row in csv.reader(in_put)]
    out_put.close()

# Remove the undesired column names 
f =  open('relation_vectors__pretrainedwv=GoogleNews-vectors-negative300.bin.vec')
file = f.readlines()
file[0] = file[0][423:]
f.close()
w = open('relation_vectors__pretrained_GoogleNews-vectors-negative300.bin.vec','w')
w.writelines(file)
w.close()

df = pd.read_csv('relation_vectors_1837d.csv')
columns_ls = list(df)
columns_ls.remove('vector')
columns_ls.remove('Unnamed: 0')
new_df = df.drop(columns_ls, axis=1)
new_df.to_csv('relation_vector_1800d.csv', header = False, index = False)

with open('relation_vector_1800.bin.vec','w') as output_file:
    with open('relation_vector_1800d.csv','r') as input_file:
        [output_file.write(" ".join(row)) for row in csv.reader(input_file)]
    output_file.close()