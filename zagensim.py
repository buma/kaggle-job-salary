from data_io import (
    get_paths,
    read_column,
    #join_features,
    #label_encode_column_fit,
    #label_encode_column_fit_only,
    #label_encode_column_transform
)
from os.path import join as path_join
#import joblib
#import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#import joblib

#from gensim.corpora import TextCorpus
#from gensim.corpora import MmCorpus
#import re
import string
import logging
#from nltk.tokenize.regexp import WordPunctTokenizer
from nltk.corpus import stopwords
#from itertools import izip, repeat
import operator

from gensim.corpora.textcorpus import TextCorpus
#from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
paths = get_paths("Settings_loc5.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")
tmp_dir = path_join(data_dir, "tmp")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('test_miislita')

class Corpus_Column(TextCorpus):
    stoplist = set('for a of the and to in on'.split()).union(set(string.punctuation))

    def __init__(self, input=None, column=None):
        super(Corpus_Column, self).__init__(input)
        self.column = column

    def getstream(self):
        logger.info("getting stream")
        reader = read_column(self.input, "Title")
        return reader

    def get_texts(self):
        """
        Parse documents from the .cor file provided in the constructor. Lowercase
        each document and ignore some stopwords.

        .cor format: one document per line, words separated by whitespace.
        """
        #tokenizer = WordPunctTokenizer()
        #print CorpusCSV.stoplist
        #return self.getstream()
        for doc in self.getstream():
            yield [word for word in doc.lower().split()]
                    #if word not in CorpusMiislita.stoplist]
            #yield doc
            #yield [word for word in tokenizer.tokenize(clean_me(doc[7], doc[6]).lower())
                    #if word_ok(word)]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            logger.info("caching corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_texts())
        return self.length


stoplist = set('for a of the and to in'.split())
stoplist = stopwords.words('english')

#dictionary = Dictionary(title.lower().split() for title in read_column(paths["train_data_path"], "Title"))
#print dictionary

fname = paths["train_data_path"]

title_corpus = Corpus_Column(fname, "Title")
MmCorpus.serialize(path_join(cache_dir, "train_title_raw_corpus.pickle"), title_corpus)
title_corpus.dictionary.save(path_join(cache_dir, "train_title_raw_dic.pickle"))
print title_corpus
print title_corpus.dictionary
#print files.dictionary
id2token = title_corpus.dictionary.values()
i = 0
for k, v in sorted(title_corpus.dictionary.dfs.iteritems(), key=operator.itemgetter(1), reverse=True):
    if i < 100:
        print id2token[k], v
        i = i + 1


