#Based on kaggle sample code https://github.com/benhamner/JobSalaryPrediction.git
import csv
import json
import os
import pandas as pd
import pickle


def read_column(filename, column_name):
    """returns generator with values in column_name in filename"""
    csv_file = csv.reader(open(filename, 'r'))
    header = csv_file.next()
    print header
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


def identity(x):
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = {"FullDescription": identity, "Title": identity, "LocationRaw": identity, "LocationNormalized": identity
              }


def get_train_df():
    train_path = get_paths()["train_data_path"]
    return pd.read_csv(train_path, converters=converters)


def get_valid_df():
    valid_path = get_paths()["valid_data_path"]
    return pd.read_csv(valid_path, converters=converters)


def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))


def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))


def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)
