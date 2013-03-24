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


fname = paths["train_data_path"]


def create_corpus(column, shortname, type_n):
    my_corpus = Corpus_Column(fname, column)
    dicti = my_corpus.dictionary
    dicti.filter_extremes()
    dicti.save(path_join(cache_dir, "%s_%s_nltk_filtered_dic.pickle" % (type_n, shortname)))
    MmCorpus.serialize(path_join(cache_dir, "%s_%s_nltk_filtered.corpus.mtx" % (type_n, shortname)), my_corpus, id2word=dicti)
    print my_corpus.dictionary
    print "50 most used in %s" % column
    i = 0
    for k, v in sorted(dicti.dfs.items(), key=operator.itemgetter(1), reverse=True):
        if i < 50:
            print dicti[k], v
            i = i + 1

#create_corpus("LocationRaw", "locraw", "train")
create_corpus("FullDescription", "desc", "train")
create_corpus("Title", "title", "train")

#description_corpus = Corpus_Column(fname, "FullDescription")
#dicti = description_corpus.dictionary
#dicti.filter_extremes()
#dicti.save(path_join(cache_dir, "train_desc_nltk_filtered_dic.pickle"))
#MmCorpus.serialize(path_join(cache_dir, "train_desc_nltk_filtered.corpus.mtx"), description_corpus, id2word=dicti)
#print description_corpus.dictionary
#print "50 most used"
#i = 0
#for k, v in sorted(dicti.dfs.items(), key=operator.itemgetter(1), reverse=True):
    #if i < 50:
        #print dicti[k], v
        #i = i + 1

#title_corpus = Corpus_Column(fname, "Title")
#dicti = title_corpus.dictionary
#dicti.filter_extremes()
#dicti.save(path_join(cache_dir, "train_title_nltk_filtered_dic.pickle"))
#MmCorpus.serialize(path_join(cache_dir, "train_title_nltk_filtered.corpus.mtx"), title_corpus, id2word=dicti)
#print title_corpus.dictionary
#print "50 most used"
#i = 0
#for k, v in sorted(dicti.dfs.items(), key=operator.itemgetter(1), reverse=True):
    #if i < 50:
        #print dicti[k], v
        #i = i + 1
