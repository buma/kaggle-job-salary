from data_io import (
    read_column,
    get_paths
)
from os.path import join as path_join
from sklearn.feature_extraction.text import CountVectorizer
import joblib


paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
#mem = Memory(cachedir=path_join(data_dir, "tmp"))
train_filename = paths["train_data_path"]
valid_filename = paths["valid_data_path"]
#count_vectorizer = mem.cache(CountVectorizer)

#creates word matrix for some columns
for column_name in ["Title", "FullDescription", "LocationRaw", "LocationNormalized"]:
    count_vector_titles = CountVectorizer(
        read_column(train_filename, column_name),
        max_features=200)
    titles = count_vector_titles.fit_transform(
        read_column(train_filename, column_name))
    joblib.dump(count_vector_titles.vocabulary_, path_join(
        cache_dir, column_name + "count_vectorizer_vocabulary"))
    joblib.dump(count_vector_titles.stop_words_, path_join(
        cache_dir, column_name + "count_vectorizer_stop_words"))
    print joblib.dump(titles, path_join(cache_dir, column_name + "_train_count_vector_matrix_max_f_200"))
    titles_valid = count_vector_titles.transform(
        read_column(valid_filename, column_name))
    print joblib.dump(titles_valid, path_join(cache_dir, column_name + "_valid_count_vector_matrix_max_f_200"))

#print titles

#counter = 0
#from collections import Counter
#times = Counter()
#for line in read_column(train_filename, "Category"):
    #times[line.lower().strip()]+=1
#print times.most_common(10)
#print list(times)
