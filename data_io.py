#Based on kaggle sample code https://github.com/benhamner/JobSalaryPrediction.git
import csv
import json
import os
import pickle
from os.path import join as path_join
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


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


def save_model(model, model_name=None, mae=None, mae_cv=None):
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
            if mae_cv is not None:
                infofile.write("\nMAE CV: %s\n" % mae_cv)
            if parameters is not None:
                infofile.write("\nParameters: %s\n" % parameters)
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
prediction_dir = path_join(data_dir, "predictions")
memory = joblib.Memory(cachedir=cache_dir)

@memory.cache
def label_encode_column_fit(column_name, file_id="train_data_path"):
    le = LabelEncoder()
    transformation = le.fit_transform(list(read_column(paths[file_id], column_name)))
    #print "classes:", list(le.classes_)
    return le, transformation


@memory.cache
def label_encode_column_transform(le, column_name, file_id="valid_data_path"):
    return le.transform(list(read_column(paths[file_id], column_name)))


@memory.cache
def join_features(filename_pattern, column_names, data_dir, additional_features=[]):
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
    #import ipdb; ipdb.set_trace()
    additional_features = map(lambda x: np.reshape(np.array(x),(len(x),1)),additional_features)
    extracted.extend(additional_features)
    if len(extracted) > 1:
        return np.concatenate(extracted, axis=1)
    else:
        return extracted[0]


def fit_predict(model_name, features, salaries, validation_features, type_n="valid"):
    if model_name == "vowpall":
        print "Naredi vowpall"
        predictions = np.loadtxt(path_join(data_dir, "code", "from_fastml", "optional", "predictions_split_" + type_n + ".txt"))
    else:
        model = load_model(model_name)
        print model
        model.fit(features, salaries)
        predictions = model.predict(validation_features)
    joblib.dump(predictions, path_join(prediction_dir, model_name + "_prediction_" + type_n))


def load_predictions(model_name, type_n="valid"):
    return joblib.load(path_join(prediction_dir, model_name + "_prediction_" + type_n))


def write_submission(submission_name, prediction_name, unlog=False):
    paths = get_paths("Settings_submission.json")
    data_dir = paths["data_path"]
    prediction_path = path_join(data_dir, "predictions", prediction_name)
    submission_path = path_join(paths["submission_path"], submission_name)
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid_name = paths["valid_data_path"]
    valid = read_column(valid_name, "Id")
    predictions = joblib.load(prediction_path)
    if unlog:
        predictions = np.exp(predictions)
    rows = [x for x in zip(valid, predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)
