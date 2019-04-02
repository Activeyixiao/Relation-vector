from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict 
import os
import multiprocessing as mp
import gensim

df = pd.read_csv('data/en_words_pair.csv')
df['words_pair'] = df['head'].map(str) + '__' + df['tail']
df.drop(['head','tail'], axis=1)

os.makedirs('build_dir')
workers=mp.cpu_count()

linecount=0
with open('testing/corpus') as f:
    for line in f:
        linecount+=1
sents_per_split=round(linecount/workers)

print('Source corpus has ',linecount,' lines')
print('Splitting original corpus in ',workers,' files of ~',sents_per_split,' lines')

linecount = 0
splitcount = 0
outf=open(os.path.join('build_dir','split_'+str(splitcount))+'.txt','w')
with open('all_wikipedia_no_POS.txt', 'r') as f:
    for line in f:
        linecount+=1
        outf.write(line)
        if linecount % sents_per_split == 0:
            outf.close()
            splitcount+=1
            outf = open(os.path.join('build_dir', 'split_'+str(splitcount)+'.txt'), 'w')
            print('saved split numb:', splitcount,'of', workers)

def extract_pairs(corpus):
    with open(corpus + '_triples.txt','w') as outf:
        max_dist = 10
        pair_counter = defaultdict(int)
        paircount = 0
        for line in open('data/word_pair.txt'):
            line=line.strip()
            w1,w2 = line.split('__')[0],line.split('__')[1]
            for corpus_line in open(corpus):
                sentences = sent_tokenize(corpus_line)
                for sentence in sentences:
                    sentence_tokens = word_tokenize(sentence)
                    if w1 in sentence_tokens and w2 in sentence_tokens:
                        dist = abs(sentence_tokens.index(w1) - sentence_tokens.index(w2))
                        if dist <= max_dist:
                            pair_counter[(w1,w2)]+=1
        ordered_frequencies = sorted(pair_counter.items(), key=lambda x:x[1], reverse=True)
        for a,b in ordered_frequencies:
            outf.write(a[0]+'__'+a[1]+'\t'+str(b)+'\n')

splits=[os.path.join('build_folder',i) for i in os.listdir('build_folder') if i.startswith('split')
workers=mp.cpu_count()
p = mp.Pool(processes=workers)
p.map(extract_pairs, splits) 

