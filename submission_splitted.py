from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

dio = DataIO("Settings.json")
submission = False
n_trees = 10
min_samples_split = 2
param = """Normal count vector with max 200. New submission which is repeatable.
 and nicer

count_vector_titles = TfidfVectorizer(
    read_column(train_filename, column_name),
    max_features=200, norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
 """

if submission:
    type_n = "train_full"
    type_v = "valid_full"
else:
    type_n = "train"
    type_v = "valid"


vectorizer = CountVectorizer(
    max_features=200,
)
short_id = "count_200f"
tfidf_columns = ["Title", "FullDescription", "LocationRaw"]
#dio.make_counts(vectorizer, short_id, tfidf_columns, type_n, type_v)


columns = ["Category", "ContractTime", "ContractType"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)

#features = dio.join_features("%s_" + type_n + "_count_vector_matrix_max_f_200",
                             #["Title", "FullDescription", "LocationRaw"],
                             #extra_features)
#validation_features = dio.join_features("%s_" + type_v + "_count_vector_matrix_max_f_200",
                                        #["Title", "FullDescription", "LocationRaw"],
                                        #extra_valid_features).astype(np.int64)
features = dio.join_features("%s_" + type_n + "_" + short_id + "_matrix",
                             tfidf_columns,
                             extra_features)
validation_features = dio.join_features("%s_" + type_v + "_" + short_id + "_matrix",
                                        tfidf_columns,
                                        extra_valid_features)

salaries = dio.get_salaries(type_n, log=True).astype(np.int64)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=True)


par = " classed from 0-11500 then 4 classes to 100 000 and to end NoNormal classTypeTime"
def encode_salaries(salaries, bins):
    bin_edges = np.linspace(11500.0, 100000, bins + 1, endpoint=True)
    #hist, bin_edges = np.histogram(salaries, bins)
    print np.diff(bin_edges)
    idxs = np.searchsorted(bin_edges, salaries, side="right")
    return idxs




#salaries_enc = encode_salaries(salaries, 4)
#valid_salaries_enc = encode_salaries(valid_salaries, 4)


print salaries.shape
metric = dio.error_metric




for bins in [4]: #range(10,15):
    n_trees = 10
    #salaries_enc = encode_salaries(salaries, bins)
    #valid_salaries_enc = encode_salaries(valid_salaries, bins)
    salaries_enc = dio.get_prediction(model_name="randomForest_tfidf_titleFullLoc_bin4", type_n="train_classes")
    valid_salaries_enc = dio.get_prediction(model_name="randomForest_tfidf_titleFullLoc_bin4", type_n="valid_classes")
    par = " classed from 0-11500 then %d classes to 100 000 and to end NoNormal classTypeTime salaries and valid predicted with randomForest_tfidf_titleFullLoc_bin4" % (bins,)
    name = "ExtraTree_min_sample%d_%dtrees_200f_noNorm_categoryTimeType_count_rf10_%dsplit_new_log" % (min_samples_split, n_trees, bins)
    print name
    num_classes = salaries_enc.max()
    print "classes:", num_classes

    def predict(class_id):
        print "predicting: ", class_id
        salaries_idx = np.where(salaries_enc == class_id)
        valid_idx = np.where(valid_salaries_enc == class_id)

        if len(salaries_idx[0]) == 0 or len(valid_idx[0]) == 0:
            return [], None

        classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                        verbose=0,
                                        n_jobs=4, # 2 jobs on submission / 4 on valid test
                                        oob_score=False,
                                        min_samples_split=min_samples_split,
                                        random_state=3465343)

        print features[salaries_idx[0], :].shape
        print salaries[salaries_idx].shape
        classifier.fit(features[salaries_idx[0], :], salaries[salaries_idx])
        predictions_part = classifier.predict(validation_features[valid_idx[0]])
        return predictions_part, valid_idx
    predictions = np.zeros_like(valid_salaries)
    for cur_class_id in range(num_classes + 1):
        predictions_part, idx = predict(cur_class_id)
        if idx is not None:
            predictions[idx] = predictions_part
            print "Part MAE: ", metric(valid_salaries[idx], predictions_part)
    if submission:
        dio.save_prediction(name, predictions, type_n=type_v)
        dio.write_submission(name + ".csv", predictions=predictions)
    else:
        dio.compare_valid_pred(valid_salaries, predictions)
        metric = dio.error_metric
        mae = metric(valid_salaries, predictions)
        print "MAE validation: ", mae
        dio.save_model(ExtraTreesRegressor(), name, mae)
        dio.save_prediction(name, predictions, type_n=type_v)
#oob_predictions = classifier.oob_prediction_
#mae_oob = mean_absolute_error(salaries, oob_predictions)
#print "MAE OOB: ", mae_oob
        #classifier1 = ExtraTreesRegressor(n_estimators=n_trees,
                                            #verbose=1,
                                            #n_jobs=3,
                                            #oob_score=False,
                                            #min_samples_split=min_samples_split,
                                            #random_state=3465343)
        #scores = cross_val_score(classifier1, features, salaries, cv=3, score_func=metric, verbose=1)
        #print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        #dio.save_model(classifier, name, mae_cv=mae_cv, parameters=param)
