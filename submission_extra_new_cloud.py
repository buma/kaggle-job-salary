from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
from os.path import join as path_join
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import cloud
import os

tfidf_columns = ["Title", "FullDescription", "LocationRaw"]
dio = DataIO("Settings.json")

vectorizer = TfidfVectorizer(
    max_features=200,
    norm='l1',
    smooth_idf=True,
    sublinear_tf=False,
    use_idf=True
)
short_id = "tfidf_200f_l1"
type_n = "train"
type_v = "valid"
dio.make_counts(vectorizer, short_id, tfidf_columns, "train", "valid")
columns = ["Category", "ContractTime", "ContractType"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)
features = dio.join_features("%s_" + type_n + "_" + short_id + "_matrix",
                             tfidf_columns,
                             extra_features)
validation_features = dio.join_features("%s_" + type_v + "_" + short_id + "_matrix",
                                        tfidf_columns,
                                        extra_valid_features)

print features.shape
print validation_features.shape
run = raw_input("OK (Y/N)?")
print run
if run != "Y":
    os.exit()

files = joblib.dump(features, "train_200f_noNorm_categoryTimeType_tfidfl1_features_jl", compress=9)
files1 = joblib.dump(validation_features, "train_200f_noNorm_categoryTimeType_tfidfl1_valid_features_jl", compress=9)
files.extend(files1)

print files

run = raw_input("OK (Y/N)?")
print run
if run != "Y":
    os.exit()

for file_name in files:
    cloud.volume.sync(file_name, "my-vol-job:")

def tfidf_cloud(n_trees):
    dio = DataIO("/data/Settings_cloud.json")
    submission = False
    min_samples_split = 2
    param = """Normal count vector with max 200. New submission which is repeatable.
    and nicer

    count_vector_titles = TfidfVectorizer(
        read_column(train_filename, column_name),
        max_features=200, norm='l1', smooth_idf=True, sublinear_tf=False, use_idf=True)
    """

    if submission:
        type_n = "train_full"
        type_v = "valid_full"
    else:
        type_n = "train"
        type_v = "valid"



#features = dio.join_features("%s_" + type_n + "_tfidf_matrix_max_f_200",
                                #["Title", "FullDescription", "LocationRaw"],
                                #extra_features)
#validation_features = dio.join_features("%s_" + type_v + "_tfidf_matrix_max_f_200",
                                            #["Title", "FullDescription", "LocationRaw"],
                                            #extra_valid_features)

#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_features", features)
#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_valid_features", validation_features)
    def load(filename):
        return joblib.load(path_join("/data", filename))

    features = load("train_200f_noNorm_categoryTimeType_tfidfl1_features_jl")
    validation_features = load("train_200f_noNorm_categoryTimeType_tfidfl1_valid_features_jl")

    print "features", features.shape
    print "valid features", validation_features.shape

#salaries = dio.get_salaries(type_n, log=True)
#if not submission:
        #valid_salaries = dio.get_salaries(type_v, log=True)

#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_salaries", salaries)
#np.save("train_200f_noNorm_categoryTimeType_tfidfl2_valid_salaries", valid_salaries)

#joblib.dump(salaries, "train_200f_noNorm_categoryTimeType_tfidfl2_salaries_jl", compress=5)
#joblib.dump(valid_salaries, "train_200f_noNorm_categoryTimeType_tfidfl2_valid_salaries_jl", compress=5)

#TODO: valid salaries so narobe dumpane

    salaries = load("train_200f_noNorm_categoryTimeType_tfidfl2_salaries_jl")
    valid_salaries = load("train_200f_noNorm_categoryTimeType_tfidfl2_valid_salaries_jl")
    dio.is_log = True

    print salaries.shape


    name = "ExtraTree_min_sample%d_%dtrees_200f_noNorm_categoryTimeType_tfidfl1_new_log" % (min_samples_split, n_trees)
    print name
    #dio.save_prediction("testni", np.array([1,2,3]), type_n="testno")
    classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                    verbose=2,
                                    n_jobs=4, # 2 jobs on submission / 4 on valid test
                                    oob_score=False,
                                    min_samples_split=min_samples_split,
                                    random_state=3465343)

    #dio.save_model(classifier, "testni_model", 99.)
    classifier.fit(features, salaries)
    predictions = classifier.predict(validation_features)
    if submission:
        dio.save_prediction(name, predictions, type_n=type_v)
        dio.write_submission(name + ".csv", predictions=predictions)
    else:
        dio.compare_valid_pred(valid_salaries, predictions)
        metric = dio.error_metric
        mae = metric(valid_salaries, predictions)
        print "MAE validation: ", mae
        dio.save_model(classifier, name, mae)
        dio.save_prediction(name, predictions, type_n=type_v)
##oob_predictions = classifier.oob_prediction_
##mae_oob = mean_absolute_error(salaries, oob_predictions)
##print "MAE OOB: ", mae_oob
        #classifier1 = ExtraTreesRegressor(n_estimators=n_trees,
                                            #verbose=1,
                                            #n_jobs=4,
                                            #oob_score=False,
                                            #min_samples_split=min_samples_split,
                                            #random_state=3465343)
        #scores = cross_val_score(classifier1, features, salaries, cv=3, score_func=metric, verbose=1)
        #print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #dio.save_model(classifier, name, mae, mae_cv=mae_cv, parameters=param)
#jid = cloud.call(tfidf_cloud, 10, _type='f2', _cores=4, _env='sklearnPrecise', _vol="my-vol-job", _label="10 trees l1")
#jid = cloud.call(tfidf_cloud, 20, _type='f2', _cores=4, _env='sklearnPrecise', _vol="my-vol-job")
#print jid
#jid = cloud.call(tfidf_cloud, 30, _type='f2', _cores=4, _env='sklearnPrecise', _vol="my-vol-job")
#print jid
#jid = cloud.call(tfidf_cloud, 40, _type='f2', _cores=4, _env='sklearnPrecise', _vol="my-vol-job")
#print jid
