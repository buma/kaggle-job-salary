from data_io import (
    get_paths,
    read_column,
    save_model,
    label_encode_column_fit,
    label_encode_column_fit_only,
    label_encode_column_transform
)
from os.path import join as path_join
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib


def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


paths = get_paths("Settings_loc5.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")


names = ["Category", "ContractTime", "ContractType", "Loc1", "Loc2", "Loc3", "Loc4", "Loc5", "Company", "SourceName"]
le_features = map(lambda x: label_encode_column_fit_only(
    x, file_id="train_full_data_path", type_n="train_full"), names)

features = map(lambda (le, name): label_encode_column_transform(le, name, file_id="train_data_path", type_n="train"), zip(le_features, names))

description_length = map(len, read_column(paths["train_data_path"], "FullDescription"))
title_length = map(len, read_column(paths["train_data_path"], "Title"))

features.append(description_length)
features.append(title_length)


#le_features, features = zip(*features_les)

validation_features = map(lambda (le, name): label_encode_column_transform(le, name, file_id="valid_data_path", type_n="valid"), zip(le_features, names))

description_length = map(len, read_column(paths["valid_data_path"], "FullDescription"))
title_length = map(len, read_column(paths["valid_data_path"], "Title"))

validation_features.append(description_length)
validation_features.append(title_length)

names.extend(["Description_length", "Title_Length"])

features = np.vstack(features).T
validation_features = np.vstack(validation_features).T


print "features", features.shape
print "valid features", validation_features.shape


salaries = np.array(list(read_column(
    paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.array(list(read_column(
    paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
salaries = np.log(salaries)
valid_salaries = np.log(valid_salaries)
print salaries.shape
#for n_clusters in [8, 10, 15, 20, 25, 30]:
#for n_clusters in range(50, 201, 25):
for n_trees in range(100,151,25):
    for min_samples_split in [2, 10, 15, 20, 25, 30]:
        #name = "Dectree_depth-3_AllLocations_log" # % (n_clusters)
        #name = "ExtraTreesRegressor_%dtrees_split%d_notex_log" % (n_trees, min_samples_split)
        name = "sgd_regressor_default_%d_notext_log" % (n_trees,)
        print name
        #kmeans = KMeans(
            #random_state=3465343,
            #n_clusters=n_clusters,
            #n_jobs=-1,
            #verbose=0)
        #classifier = Pipeline(steps=[('knn', kmeans), ('tree', DecisionTreeRegressor())])
        #classifier = DecisionTreeRegressor(max_depth=3, random_state=3465343)
        #classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                        #verbose=2,
                                        #n_jobs=-1,
                                        #oob_score=False,
                                        #min_samples_split=min_samples_split,
                                        #random_state=3465343)

        clf = SGDRegressor(random_state=3465343, verbose=1, n_iter=n_trees)
        classifier = Pipeline(steps=[('scale', StandardScaler()), ('sgd', clf)])
        classifier.fit(features, salaries)
        #import StringIO
        #with open("iris.dot", 'w') as f:
            #f = export_graphviz(classifier, out_file=f)
        predictions = classifier.predict(validation_features)
        print np.exp(valid_salaries[1:10])
        print np.exp(predictions[1:10])
        mae = log_mean_absolute_error(valid_salaries, predictions)
        print "MAE validation: ", mae
        #importances = classifier.feature_importances_
        #indices = np.argsort(importances)[::-1]
        #num_feat = len(names)

## Print the feature ranking
        #print "Feature ranking:"

        #for f in xrange(len(indices)):
            #print "%d. feature %d %s (%f)" % (f + 1, indices[f], names[indices[f]], importances[indices[f]])

## Plot the feature importances of the forest
        #import pylab as pl
        #pl.figure()
        #pl.title("Feature importances")
        #pl.bar(xrange(len(indices)), importances[indices],
                #color="r", align="center")
        #pl.xticks(xrange(len(indices)), indices)
        #pl.xlim([-1, len(indices)])
        #pl.show()
        #a=5/0
        save_model(classifier, name, mae)
        #joblib.dump(predictions, path_join(prediction_dir, name + "_prediction_valid"))
        #kmeans = KMeans(
            #random_state=3465343,
            #n_clusters=n_clusters,
            #n_jobs=-1,
            #verbose=0)
        #classifier = Pipeline(steps=[('knn', kmeans), ('tree', DecisionTreeRegressor())])
        #classifier = DecisionTreeRegressor(max_depth=3, random_state=3465343)
        #classifier = ExtraTreesRegressor(n_estimators=n_trees,
                                        #verbose=1,
                                        #n_jobs=-1,
                                        #oob_score=False,
                                        #min_samples_split=min_samples_split,
                                        #random_state=3465343)
        clf = SGDRegressor(random_state=3465343, verbose=0, n_iter=n_trees)
        classifier = Pipeline(steps=[('scale', StandardScaler()), ('sgd', clf)])
        scores = cross_val_score(classifier, features, salaries, cv=5, score_func=log_mean_absolute_error, verbose=1)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        save_model(classifier, name, mae, mae_cv, parameters=",".join(names))
        break
    ##break
