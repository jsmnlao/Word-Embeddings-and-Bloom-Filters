# pulled content from iterative_vectors.ipynb
import contextlib
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet
import numpy as np
import contextlib
import string
import nltk
import json
import copy
import spacy
import itertools
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

# with open('data/fairytales_word_tf-idfs.json', 'r') as f:
with open('data/fairytales_word_tf-idfs_drop.json', 'r') as f:
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
                    instance_representation += np.array(preassign_iterative_vectors[adjacent_word]) * tf_idf
                except Exception:
                    instance_representation += np.array(bloom_filters[adjacent_word]) * tf_idf
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
    return representations / float(total_adjacent_words)

def update_encoding(word, args):
    vector = extract_vectors(word, **args)
    iterative_vectors[word] = vector

def normalize_vector(): # dimensions sum to 1
    for word in iterative_vectors.keys():
        iterative_vectors[word] = list(iterative_vectors[word] / np.linalg.norm(iterative_vectors[word])) # normalized & list conversion

if __name__ == '__main__':
    ITERATIONS = 20
    iterative_vectors = {}
    for i in range(ITERATIONS): 
        preassign_iterative_vectors = copy.deepcopy(iterative_vectors) # generates a copy so everything is updated at the end
        for word in tf_idfs.keys():
            print(f"iteration {i}, \"{word}\"")
            update_encoding(word, {'deltas': [-4, -3, -2, -1, 1, 2, 3, 4], 'bits':32})
        normalize_vector()
        with open(f'data/iterative_vectors_dropped_tf-idfs/{i}.json', 'w+') as f: # save in separate files
            json.dump(iterative_vectors, f, indent=4)