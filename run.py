from data_io import (
    get_paths,
    read_column,
    save_model,
    join_features,
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


paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")

features = join_features("%strain_count_vector_matrix_max_f_100",
        ["Title", "FullDescription", "LocationRaw", "LocationNormalized"],
        data_dir)
validation_features = join_features("%svalid_count_vector_matrix_max_f_100",
        ["Title", "FullDescription", "LocationRaw", "LocationNormalized"],
        data_dir)
print "features", features.shape
print "valid features", validation_features.shape
salaries = np.array(list(read_column(paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
print salaries.shape
#classifier = RandomForestRegressor(n_estimators=10,
                                   #verbose=2,
                                   #n_jobs=1,
                                   #oob_score=True,
                                   #min_samples_split=30,
                                   #random_state=3465343)
for n_trees in range(30,51,10):
    print n_trees
    name = "ExtraTree_min_samplesdef_%dtrees" % n_trees
    print name
    classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                    verbose=2,
                                    n_jobs=1,
                                    oob_score=False,
                                    #min_samples_split=1,
                                    random_state=3465343)
#classifier = SGDRegressor(random_state=3465343, verbose=0, n_iter=50)
#classifier = LinearRegression()
    classifier.fit(features, salaries)
    predictions = classifier.predict(validation_features)
    print valid_salaries[1:10]
    print predictions[1:10]
    mae = mean_absolute_error(valid_salaries, predictions)
    print "MAE validation: ", mae
    save_model(classifier, name, mae)
    #oob_predictions = classifier.oob_prediction_
    #mae_oob = mean_absolute_error(salaries, oob_predictions)
    #print "MAE OOB: ", mae_oob
    #break
