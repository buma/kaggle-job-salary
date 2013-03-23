from data_io import (
    get_paths,
    read_column,
    #join_features,
    #label_encode_column_fit,
    #label_encode_column_fit_only,
    #label_encode_column_transform
)
from os.path import join as path_join
import re

#import string
import logging
from nltk.tokenize.regexp import WordPunctTokenizer
from nltk.corpus import stopwords
from itertools import izip, repeat
import operator
import joblib
from collections import Counter

from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
paths = get_paths("Settings_loc5.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")
tmp_dir = path_join(data_dir, "tmp")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('test_miislita')

regex = re.compile("^(.+;\s*)\n", re.MULTILINE)
regex_num = re.compile("\W(\d+)\W")
regex_hex = re.compile("\W([a-f0-9-]+)\W")
regex_punct = re.compile("^([:/\]\[\)\(,\.\}\{\?\!#=@'0-9]+)$")
english_stopwords = stopwords.words('english')
english_stopwords = dict(izip(english_stopwords, repeat(True, len(english_stopwords))))
#tokenizer = WordPunctTokenizer()

#cnt = Counter()


def word_ok(word):
    if word in english_stopwords:
        return False
    if regex_punct.match(word):
        return False
    #cnt[word] += 1
    return True


class Corpus_Column(TextCorpus):
    #stoplist = set('for a of the and to in on'.split()).union(set(string.punctuation))

    def __init__(self, input=None, column=None):
        self.column = column
        #self.input = input
        super(Corpus_Column, self).__init__(input)

    def getstream(self):
        logger.info("getting stream")
        reader = read_column(self.input, self.column)
        return reader

    def get_texts(self):
        """
        Parse documents from the .cor file provided in the constructor. Lowercase
        each document and ignore some stopwords.

        .cor format: one document per line, words separated by whitespace.
        """
        tokenizer = WordPunctTokenizer()
        #print CorpusCSV.stoplist
        #return self.getstream()
        for doc in self.getstream():
            #yield [word for word in doc.lower().split()]
                    #if word not in CorpusMiislita.stoplist]
            #yield doc
            yield [word for word in tokenizer.tokenize(doc.lower())
                   if word_ok(word)]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            logger.info("caching corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_texts())
        return self.length


#stoplist = set('for a of the and to in'.split())
#stoplist = stopwords.words('english')

#dictionary = Dictionary(title.lower().split() for title in read_column(paths["train_data_path"], "Title"))
#print dictionary

fname = paths["train_data_path"]


#title_corpus = Corpus_Column(fname, "Title")
#MmCorpus.serialize(path_join(cache_dir, "train_title_nltk_corpus.pickle"), title_corpus)
#title_corpus.dictionary.save(path_join(cache_dir, "train_title_nltk_dic.pickle"))
#print title_corpus
#print title_corpus.dictionary

#description_corpus = Corpus_Column(fname, "FullDescription")
#print len(description_corpus)
#for word in description_corpus.get_texts():
    #a = 5
#joblib.dump(cnt, path_join(cache_dir, "counter_train_desc_nltk"), compress=3)

cnt = joblib.load(path_join(cache_dir, "counter_train_desc_nltk"))

for word, freq in cnt.most_common(10): #[:-100:-1]:
    print word, freq
#MmCorpus.serialize(path_join(cache_dir, "train_desc_nltk_corpus.pickle1"), description_corpus)
#description_corpus.dictionary.save(path_join(cache_dir, "train_desc_nltk_dic.pickle"))
dicti = Dictionary.load(path_join(cache_dir, "train_desc_nltk_dic.pickle"))
#dicti = description_corpus.dictionary
print dicti
#print description_corpus
#print description_corpus.dictionary
#print files.dictionary

#id2token = dicti.id2token
i = 0
for k, v in sorted(dicti.dfs.items(), key=operator.itemgetter(1), reverse=True):
    if i < 10:
        print dicti[k], v, "ID:", k
        i = i + 1
k=0
print "printam token", k
print id2token[k], dicti.dfs[k], "ID:", k
#def num_appear(id):
    #total_sum = sum(
