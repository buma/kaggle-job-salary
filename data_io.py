#Based on kaggle sample code https://github.com/benhamner/JobSalaryPrediction.git
import csv
import json
import os
import pickle
from os.path import join as path_join
from os.path import isfile
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import mmread
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from scipy.sparse import coo_matrix, hstack, vstack
try:
    import cloud
    on_cloud = cloud.running_on_cloud()
except Exception as e:
    on_cloud = False


def read_column(filename, column_name):
    """returns generator with values in column_name in filename"""
    csv_file = csv.reader(open(filename, 'r'))
    header = csv_file.next()
    #print header
    if column_name not in header:
        raise Exception("Column '%s' is not in header! Header: %s" % (column_name, ",".join(header)))
    column_index = header.index(column_name)
    for line in csv_file:
        yield line[column_index]


def get_paths(filename="Settings.json"):
    paths = json.loads(open(filename).read())
    data_path = os.path.expandvars(paths["data_path"])
    for key in paths:
        paths[key] = os.path.join(data_path, os.path.expandvars(paths[key]))
    return paths


def save_model(model, model_name=None, mae=None, mae_cv=None, parameters=None):
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
def label_encode_column_fit(column_name, file_id="train_data_path", type_n="train"):
    le = LabelEncoder()
    transformation = le.fit_transform(list(read_column(paths[file_id], column_name)))
    #print "classes:", list(le.classes_)
    return le, transformation

@memory.cache
def label_encode_column_fit_only(column_name, file_id="train_data_path", type_n="train"):
    le = LabelEncoder()
    le.fit(list(read_column(paths[file_id], column_name)))
    #print "classes:", list(le.classes_)
    return le


@memory.cache
def label_encode_column_transform(le, column_name, file_id="valid_data_path", type_n="valid"):
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
    filepath = path_join(prediction_dir, model_name + "_prediction_" + type_n)
    #print "path:", filepath
    if isfile(filepath):
        print model_name + "_prediction_" + type_n + " already exists"
        return
    else:
        print model_name, "doing fit_predict"
    if model_name.startswith("vowpall"):
        print "Naredi vowpall"
        predictions = np.loadtxt(path_join(data_dir, "code", "from_fastml", "optional", "predictions_split_" + type_n + ".txt"))
    else:
        model = load_model(model_name)
        print model
        model.fit(features, salaries)
        predictions = model.predict(features)
        joblib.dump(predictions, path_join(prediction_dir, model_name + "_train_prediction_" + type_n))
        predictions = model.predict(validation_features)
    joblib.dump(predictions, filepath)


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


class DataIO(object):

    def __init__(self, paths_name, cache=True):
        self.paths_name = paths_name
        self.paths = self._get_paths()
        if not on_cloud and cache:
            memory = joblib.Memory(cachedir=self.cache_dir)
            self.get_le_features = memory.cache(self.get_le_features)
            self.get_features = memory.cache(self.get_features)
        self.is_log = False

    def _get_paths(self):
        paths = json.loads(open(self.paths_name).read())
        data_path = os.path.expandvars(paths["data_path"])
        for key in paths:
            paths[key] = os.path.join(data_path, os.path.expandvars(paths[key]))
        self.data_dir = paths["data_path"]
        self.cache_dir = paths["cache_dir"]
        self.prediction_dir = paths["prediction_dir"]
        self.models_dir = paths["models_dir"]
        self.submission_path = paths["submission_path"]
        return paths

    def _check_type_n(self, type_n):
        if type_n == "train":
            file_id = "train_data_path"
        elif type_n == "train_and_valid":
            file_id = "train_and_valid"
        elif type_n == "train_full":
            if "train_full_data_path" in self.paths:
                file_id = "train_full_data_path"
            else:
                file_id = "train_data_path"
        elif type_n == "valid" or type_n == "valid_full":
            file_id = "valid_data_path"
        elif type_n == "testno" or type_n == "valid_classes" or type_n == "train_classes":
            file_id = "uknown"
        else:
            raise ValueError("Unknown type_n: %s" % type_n)
        return file_id

    def get_le_features(self, columns, type_n):
        file_id = self._check_type_n(type_n)

        le_features = map(lambda x: self.label_encode_column_fit(
            x, file_id=file_id, type_n=type_n), columns)
        return le_features

    def make_counts(self, preprocessor, short_id, column_names, type_n, type_v):
        #count_vector_titles = CountVectorizer(
            #read_column(train_filename, column_name),
            #max_features=200)
        file_id = self._check_type_n(type_n)
        valid_file_id = self._check_type_n(type_v)
        name = "%s_%s_%s_%s"
        for column_name in column_names:
            vocabulary_path = path_join(self.cache_dir, name % (column_name, type_n, short_id, "vocabulary"))
            stop_words_path = path_join(self.cache_dir, name % (column_name, type_n, short_id, "stop_words"))
            valid_path = path_join(self.cache_dir, name % (column_name, type_v, short_id, "matrix"))
            cur_preprocessor = clone(preprocessor)
            print "Working on %s" % column_name
            if isfile(vocabulary_path) and isfile(stop_words_path):
                print "vocabulary exists"
                vocabulary = joblib.load(vocabulary_path)
                stop_words = joblib.load(stop_words_path)
                cur_preprocessor.set_params(vocabulary=vocabulary)
                cur_preprocessor.set_params(stop_words=stop_words)
            else:
                print "Fitting train"
                cur_preprocessor.set_params(input=self.read_column(file_id, column_name))
                titles = cur_preprocessor.fit_transform(self.read_column(file_id, column_name))
                joblib.dump(cur_preprocessor.vocabulary_, vocabulary_path)
                joblib.dump(cur_preprocessor.stop_words_, stop_words_path)
                print joblib.dump(titles, path_join(self.cache_dir, name % (column_name, type_n, short_id, "matrix")))
            if not isfile(valid_path):
                print "Fitting valid"
                titles_valid = cur_preprocessor.transform(
                    self.read_column(valid_file_id, column_name))
                print joblib.dump(titles_valid, valid_path)

    def join_features(self, filename_pattern, column_names, additional_features=[], sparse=False):
        #filename = "%strain_count_vector_matrix_max_f_100"
        extracted = []
        print("Extracting features and training model")
        for column_name in column_names:  # ["Title", "FullDescription", "LocationRaw", "LocationNormalized"]:
            print "Extracting: ", column_name
            fea = joblib.load(path_join(self.cache_dir, filename_pattern % column_name))
            print fea.shape
            if not sparse and hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        #import ipdb; ipdb.set_trace()
        #print map(len, additional_features)
        # changes arrays in additional features to numpy arrays with shape(len(x), 1)
        additional_features = map(lambda x: np.reshape(np.array(x), (len(x), 1)), additional_features)
        if sparse:
            additional_features = map(coo_matrix, additional_features)
        extracted.extend(additional_features)
        if len(extracted) > 1:
            if sparse:
                return hstack(extracted)
            else:
                return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

    def get_features(self, columns, type_n, le_features):
        file_id = self._check_type_n(type_n)
        extra_features = map(lambda (le, name): self.label_encode_column_transform(le, name, file_id=file_id, type_n=type_n), zip(le_features, columns))
        return extra_features

    def read_column(self, filename_or_path, column_name):
        """returns generator with values in column_name in filename"""
        if filename_or_path in self.paths:
            filename = self.paths[filename_or_path]
        elif isfile(filename_or_path):
            filename = filename_or_path
        else:
            raise Exception("filename_or_path: '%s' not found" % filename_or_path)
        csv_file = csv.reader(open(filename, 'r'))
        header = csv_file.next()
        #print header
        if column_name not in header:
            raise Exception("Column '%s' is not in header! Header: %s" % (column_name, ",".join(header)))
        column_index = header.index(column_name)
        for line in csv_file:
            yield line[column_index]

    def label_encode_column_fit_transform(self, column_name, file_id="train_data_path", type_n="train"):
        """Returns LabelEncoder and transformation for column

        Parameters
        ----------
        column_name: string
            Which column
        file_id: string
            Id of file in paths. Gets filepath
        type_n: string
            train, test, valid. Same file id gets different files in different types. For caching
        """
        le = LabelEncoder()
        transformation = le.fit_transform(list(self.read_column(file_id, column_name)))
        #print "classes:", list(le.classes_)
        return le, transformation

    def label_encode_column_fit(self, column_name, file_id="train_data_path", type_n="train"):
        le = LabelEncoder()
        le.fit(list(self.read_column(file_id, column_name)))
        #print "classes:", list(le.classes_)
        return le

    def label_encode_column_transform(self, le, column_name, file_id="valid_data_path", type_n="valid"):
        return le.transform(list(self.read_column(file_id, column_name)))

    def read_gensim_corpus(self, filename):
        """Reads corpus in MMX format and returns Sparse scipy matrix

        It assumes the corpus is in cache directory"""
        corpus_csc = mmread(open(path_join(self.cache_dir, filename), "r"))
        return corpus_csc

    def get_salaries(self, type_n, log=False):
        file_id = self._check_type_n(type_n)
        salaries = np.array(list(self.read_column(file_id, "SalaryNormalized"))).astype(np.float64)
        if log:
            self.is_log = True
            salaries = np.log(salaries)
        return salaries

    def compare_valid_pred(self, valid, pred):
        if self.is_log:
            mexp = np.exp
        else:
            mexp = lambda x: x
        print mexp(valid[1:10])
        print mexp(pred[1:10])

    def error_metric(self, y_true, y_pred):
        if self.is_log:
            mexp = np.exp
        else:
            mexp = lambda x: x
        return mean_absolute_error(mexp(y_true), mexp(y_pred))
    #TODO: Why this doesn't work?
        #if self.is_log:
            #def log_mean_absolute_error(y_true, y_pred):
                #return mean_absolute_error(np.exp(y_true), np.exp(y_pred))
            #return log_mean_absolute_error
        #else:
            #def local_mean_absolute_error(y_true, y_pred):
                #return mean_absolute_error(y_true, y_pred)
            #return local_mean_absolute_error

    def save_prediction(self, model_name, predictions, type_n):
        self._check_type_n(type_n)
        if on_cloud:
            joblib.dump(predictions, model_name + "_prediction_" + type_n, compress=5)
            cloud.bucket.put(model_name + "_prediction_" + type_n, prefix="prediction")
        else:
            joblib.dump(predictions, path_join(self.prediction_dir, model_name + "_prediction_" + type_n), compress=5)

    def load_model(self, model_name):
        in_path = path_join(self.models_dir, model_name + ".pickle")
        return pickle.load(open(in_path))

    def save_model(self, model, model_name, mae=None, mae_cv=None, parameters=None):
        """Saves model in model_name.pickle file
        also creates model_name.txt with model parameters and
        mae value on validation set if provided

        :param mae MAE validation float
        :param mae_cv MAE CV string
        :param parameters model parameters string"""
        if model_name is None:
            raise ValueError("Model name is empty!")
        else:
            filepath = path_join(self.models_dir, model_name)
            with open(filepath + '.txt', 'wb') as infofile:
                infofile.write(str(model))
                infofile.write("\n")
                if mae is not None:
                    infofile.write("\nMAE validation: %f\n" % mae)
                if mae_cv is not None:
                    infofile.write("\nMAE CV: %s\n" % mae_cv)
                if parameters is not None:
                    infofile.write("\nParameters: %s\n" % (parameters, ))
            if on_cloud:
                cloud.bucket.put(filepath + '.txt', prefix="models")
            out_path = filepath + ".pickle"

            pickle.dump(model, open(out_path, "w"))
            if on_cloud:
                cloud.bucket.put(out_path, prefix="models")

    def get_prediction(self, prediction_name=None, model_name=None, type_n=None):
        if prediction_name is not None:
            filename = prediction_name
        elif model_name is not None and type_n is not None:
            filename = model_name + "_prediction_" + type_n
        return joblib.load(path_join(self.prediction_dir, filename))

    def write_submission(self, submission_name, predictions=None, prediction_name=None, model_name=None, type_n=None, unlog=False):
        submission_file_path = path_join(self.submission_path, submission_name)
        writer = csv.writer(open(submission_file_path, "w"), lineterminator="\n")
        valid = self.read_column("valid_data_path", "Id")
        if predictions is None:
            predictions = self.get_prediction(prediction_name=prediction_name,
                                                model_name=model_name,
                                                type_n=type_n)
            if unlog:
                predictions = np.exp(predictions)
        else:
            if self.is_log:
                predictions = np.exp(predictions)
        rows = [x for x in zip(valid, predictions.flatten())]
        writer.writerow(("Id", "SalaryNormalized"))
        writer.writerows(rows)
