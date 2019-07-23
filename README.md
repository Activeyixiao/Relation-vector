Relation-vector

This project aims at modelling relation types with the help of knowledge graph and text copus 

A Working Example:

1.

Getting words pairs from conceptnet and count the time of its occurrence in the text corpus and then return the frequency of the words pairs.

# python3 preprocess_wp/wp_from_corpus.py -c corpus/corpus.txt -b build_folder -win 10 -sw language_tool/english_stopwords.txt -kg KG/en_words_pair.csv

2.

After select the words pairs with high frequency,the contexts of the words pairs are extracted in the text corpus.

# python3 preprocess_wp/wp_get_context.py -p build_folder/wp_freq_count.txt -b build_folder -mw 5 -sw 5

3.

convert context into concatenation of word embeddings

# python3 preprocess_wp/vectorize.py -wv word_embedding/GoogleNews-vectors-negative300.bin -p build_folder/wp_freq_count.txt -b build_folder

4.

Encoding the words pairs in conceptnet as the 38 dimensional one-hot-representation which stipulate the occurance or absence of each relation type. 

# python3 preprocess_wp/vectors_concatenation.py -r build_folder/relation_vector_pretrained.vec -kg KG/en_words_pair.csv -b build_folder

5.

splitting the words_pair into train set and test sets based on knowledge graph partition

# python3 preprocess_wp/data_separation.py -r build_folder/relation_vectors_1838d.vec -b experiment_data

6.

building the neural network classifer for each relation type. 

# python3 preprocess_wp/new_neural_classifer.py -rv1 experiment_data/train_data.vec -rv2 experiment_data/test_data.vec -wv word_embedding/GoogleNews-vectors-negative300.bin -kg KG/en_words_pair.csv -b build_folder
