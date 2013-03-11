#Based on kaggle sample code https://github.com/benhamner/JobSalaryPrediction.git
import csv
import json
import os
import pickle
from os.path import join as path_join
import joblib
import numpy as np


def read_column(filename, column_name):
    """returns generator with values in column_name in filename"""
    csv_file = csv.reader(open(filename, 'r'))
    header = csv_file.next()
    #print header
    if column_name not in header:
        raise Exception("Column name is not in header!")
    column_index = header.index(column_name)
    for line in csv_file:
        yield line[column_index]


def get_paths(filename="Settings.json"):
    paths = json.loads(open(filename).read())
    data_path = os.path.expandvars(paths["data_path"])
    for key in paths:
        paths[key] = os.path.join(data_path, os.path.expandvars(paths[key]))
    return paths


def save_model(model, model_name=None, mae=None):
    """Saves model in model_name.pickle file
    also creates model_name.txt with model parameters and
    mae value on validation set if provided"""
    if model_name is None:
        out_path = get_paths()["model_path"]
    else:
        filepath = os.path.join(get_paths()["data_path"], "models", model_name)
# Saves model parameters
        with open(filepath + '.txt', 'wb') as infofile:
            infofile.write(str(model))
            infofile.write("\n")
            if mae is not None:
                infofile.write("\nMAE validation: %f\n" % mae)
        out_path = filepath + ".pickle"

    pickle.dump(model, open(out_path, "w"))


def load_model(model_name=None):
    if model_name is None:
        in_path = get_paths()["model_path"]
    else:
        in_path = os.path.join(get_paths()["data_path"], "models", model_name + ".pickle")
    return pickle.load(open(in_path))

paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
memory = joblib.Memory(cachedir=cache_dir)


@memory.cache
def join_features(filename_pattern, column_names, data_dir):
    #filename = "%strain_count_vector_matrix_max_f_100"
    cache_dir = path_join(data_dir, "tmp")
    extracted = []
    print("Extracting features and training model")
    for column_name in column_names:  # ["Title", "FullDescription", "LocationRaw", "LocationNormalized"]:
        print "Extracting: ", column_name
        fea = joblib.load(path_join(cache_dir, filename_pattern % column_name))
        if hasattr(fea, "toarray"):
            extracted.append(fea.toarray())
        else:
            extracted.append(fea)
    if len(extracted) > 1:
        return np.concatenate(extracted, axis=1)
    else:
        return extracted[0]


#FIXME: fix function
def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)
