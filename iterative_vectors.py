# pulled content from iterative_vectors.ipynb
import contextlib
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet
import mmh3
import numpy as np
import contextlib
import numpy as np
import string
import nltk
import json

import spacy
import lemminflect
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
POS = ("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB")

def lemmatize(word): # Takes a word and uses the spacy lemmatizer to return the lemmatized form
    token = nlp(str(word))[0]
    lemma = token.lemma_
    inflections = {token._.inflect(pos) for pos in POS} # returns the inflection of the lemmatized token. (ex: run -> {'ran', 'run', 'runner', 'runnest', 'running', 'runs'} )
    return lemma, inflections

def tokenize(sentence): # Tokenizes a sentence and lemmatizes the words within
    tokenized = nlp(sentence.translate(str.maketrans('', '', string.punctuation)))
    return [token.lemma_ for token in tokenized if token.lemma_.lower() not in en_stopwords and wordnet.synsets(token.lemma_)] # disregards lemmatized token if it's in list of stopwords or not in english dictionary (wordnet)

nltk.download('stopwords')
nltk.download('wordnet')
en_stopwords = set(stopwords.words('english'))

with open('data/fairytales_iterative_vectors.json', 'r') as f:
    iterative_vectors = json.load(f)
with open('data/fairytales_word_tf-idfs.json', 'r') as f:
    tf_idfs = json.load(f)
with open('data/fairytales_word_bloom-filters.json', 'r') as f:
    bloom_filters = json.load(f)
with open('data/fairytales_tokenized.json', 'r') as f:
    tokenized_corpus = json.load(f)

def generate_vector(word, tokenized_sentence, bits, deltas):
    indices = [i for i, x in enumerate(tokenized_sentence) if x == word]
    instance_representation = np.zeros(bits)
    adjacent_words = 0
    
    for index in indices:
        for delta in deltas:
            if index + delta < 0:
                continue
            with contextlib.suppress(IndexError):
                adjacent_word = tokenized_sentence[index + delta]
                try:
                    tf_idf = tf_idfs[word][adjacent_word]
                except KeyError:
                    tf_idf = 0
                try:
                    instance_representation += np.array(iterative_vectors[adjacent_word]) * tf_idf
                except:
                    instance_representation += np.array(bloom_filters[adjacent_word]) * tf_idf 
                    # generate new bloom filter to represent word if vector is not found
                adjacent_words += 1
    return instance_representation, adjacent_words

def extract_vectors(word, deltas=None, bits=32):
    if deltas is None:
        deltas = [-4, -3, -2, -1, 1, 2, 3, 4]

    total_adjacent_words = 0
    representations = np.zeros(bits)

    for sentence in tokenized_corpus:
        if word in sentence:
            representation, adjacent_words = generate_vector(word, sentence, bits, deltas)
            representations += representation
            total_adjacent_words += adjacent_words
    iterative_vectors[word] = representations
    return representations / float(total_adjacent_words)

def store_encoding(word, fname, args):
    vector = list(extract_vectors(word, **args))
    
    with open(fname, 'r') as f:
        iterative_vectors = json.load(f)
    iterative_vectors[word] = vector
    with open(fname, 'w') as f:
        json.dump(iterative_vectors, f, indent=4)

if __name__ == '__main__':
    ITERATIONS = 50
    for i in range(ITERATIONS): 
        for word in tf_idfs.keys():
            print(f"iteration {i}, \"{word}\"")
            store_encoding(word, 'data/fairytales_iterative_vectors.json', {'deltas': [-4, -3, -2, -1, 1, 2, 3, 4], 'bits':32})