from data_io import (
    get_paths,
    read_column,
    save_model,
    join_features,
    label_encode_column_fit,
    label_encode_column_transform
)
from os.path import join as path_join
#import joblib
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
import joblib


def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

le_category, category_train = label_encode_column_fit("Category")
category_valid = label_encode_column_transform(le_category, "Category")

le_contractTime, contractTime_train = label_encode_column_fit("ContractTime")
contractTime_valid = label_encode_column_transform(le_contractTime, "ContractTime")

le_contractType, contractType_train = label_encode_column_fit("ContractType")
contractType_valid = label_encode_column_transform(le_contractType, "ContractType")


features = join_features("%strain_count_vector_matrix_max_f_100", #train_tfidf_matrix_max_f_200
                         ["Title", "FullDescription", "LocationRaw"],
                         data_dir,
                         [contractTime_train, contractType_train, category_train])
validation_features = join_features("%svalid_count_vector_matrix_max_f_100",#valid_tfidf_matrix_max_f_200
                                    ["Title", "FullDescription", "LocationRaw"],
                                    data_dir,
                                    [contractTime_valid, contractType_valid, category_valid])
print "features", features.shape
print "valid features", validation_features.shape


salaries = np.array(list(read_column(paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
salaries = np.log(salaries)
print salaries.shape
#classifier = RandomForestRegressor(n_estimators=10,
                                   #verbose=2,
                                   #n_jobs=1,
                                   #oob_score=True,
                                   #min_samples_split=30,
                                   #random_state=3465343)
for n_trees in range(30,51,10):
    for min_samples_split in [2, 30]:
        print n_trees
        #name = "ExtraTree_min_sample%d_%dtrees_200f_noNorm_categoryTimeType_log" % (min_samples_split, n_trees)
        name = "adaBoost_decision_%dtrees_100f_noNorm_categoryTimeType_log" % (n_trees)
        print name
        classifier = AdaBoostRegressor(n_estimators=n_trees, random_state=3465343)

        classifier.fit(features, salaries)
        predictions = classifier.predict(validation_features)
        print valid_salaries[1:10]
        print np.exp(predictions[1:10])
        mae = mean_absolute_error(valid_salaries, np.exp(predictions))
        print "MAE validation: ", mae
        save_model(classifier, name, mae)
        #joblib.dump(predictions, path_join(prediction_dir, name + "_prediction_valid"))
        #oob_predictions = classifier.oob_prediction_
        #mae_oob = mean_absolute_error(salaries, oob_predictions)
        #print "MAE OOB: ", mae_oob
        classifier = AdaBoostRegressor(n_estimators=n_trees, random_state=3465343)
        scores = cross_val_score(classifier, features, salaries, cv=3, score_func=log_mean_absolute_error, verbose=1)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        save_model(classifier, name, mae, mae_cv)
        break
    #break
