from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score

dio = DataIO("Settings.json", cache=False)
submission = False
n_trees = 10
min_samples_split = 2

if submission:
    type_n = "train_full"
    type_v = "valid_full"
else:
    type_n = "train"
    type_v = "valid"


columns = ["Category", "ContractTime", "ContractType"]
le_features = dio.get_le_features(columns, "train_full")
extra_features = dio.get_features(columns, type_n, le_features)
extra_valid_features = dio.get_features(columns, type_v, le_features)
features = dio.join_features("%s_" + type_n + "_count_vector_matrix_max_f_200",
                             ["Title", "FullDescription", "LocationRaw"],
                             extra_features)
validation_features = dio.join_features("%s_" + type_v + "_count_vector_matrix_max_f_200",
                                        ["Title", "FullDescription", "LocationRaw"],
                                        extra_valid_features)
print "features", features.shape
print "valid features", validation_features.shape

salaries = dio.get_salaries(type_n, log=True)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=True)

print salaries.shape
for n_trees in [20, 30, 40]:
    name = "ExtraTree_min_sample%d_%dtrees_200f_noNorm_categoryTimeType_new_log" % (min_samples_split, n_trees)
    print name
    classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                    verbose=2,
                                    n_jobs=2,
                                    oob_score=False,
                                    min_samples_split=min_samples_split,
                                    random_state=3465343)

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

