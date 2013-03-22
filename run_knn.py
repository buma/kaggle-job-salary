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


#le_features, features = zip(*features_les)

validation_features = map(lambda (le, name): label_encode_column_transform(le, name, file_id="valid_data_path", type_n="valid"), zip(le_features, names))

features = np.vstack(features).T
validation_features = np.vstack(validation_features).T


print "features", features.shape
print "valid features", validation_features.shape


salaries = np.array(list(read_column(
    paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.array(list(read_column(
    paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
#salaries = np.log(salaries)
#valid_salaries = np.log(valid_salaries)
print salaries.shape
for n_clusters in [8, 10, 15, 20, 25, 30]:
        name = "Knn_linreg_%dcenters_categoryTimeTypeLoc2CompanySourcename" % (n_clusters)
        print name
        kmeans = KMeans(
            random_state=3465343,
            n_clusters=n_clusters,
            n_jobs=-1,
            verbose=0)
        classifier = Pipeline(steps=[('knn', kmeans), ('linear', LinearRegression())])

        classifier.fit(features, salaries)
        predictions = classifier.predict(validation_features)
        print valid_salaries[1:10]
        print predictions[1:10]
        mae = mean_absolute_error(valid_salaries, predictions)
        print "MAE validation: ", mae
        save_model(classifier, name, mae)
        #joblib.dump(predictions, path_join(prediction_dir, name + "_prediction_valid"))
        kmeans = KMeans(
            random_state=3465343,
            n_clusters=n_clusters,
            n_jobs=-1,
            verbose=0)
        classifier = Pipeline(steps=[('knn', kmeans), ('linear', LinearRegression())])
        scores = cross_val_score(classifier, features, salaries, cv=3, score_func=mean_absolute_error, verbose=1)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        mae_cv = "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        save_model(classifier, name, mae, mae_cv, parameters=",".join(names))
        #break
    ##break
