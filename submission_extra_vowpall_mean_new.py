from data_io import (
    get_paths,
    write_submission,
    load_predictions
)
from os.path import join as path_join
#import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
import joblib

paths = get_paths("Settings_submission.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))



model_names = ["ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log",
 "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log",
 "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new",
 "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidfl2_new_log",
 "vowpall_submission",
 "vowpall_loc5"]
#model_names = [model2, model4]
#model_names = [model1, model6, model4]


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)
#fit_predict(model5)

#fit_predict(model4, features, salaries, validation_features, type_n="test_subm")


all_model_predictions = []
for model_name in model_names:
    #fit_predict(model_name, features, salaries, validation_features, type_n="test_subm")
    model_predictions = load_predictions(model_name, type_n="valid_full")
    if not model_name.endswith("log") and not model_name.startswith("vowpall"):
        model_predictions = np.log(model_predictions)
    print "modelp", model_predictions.shape
    #print "%s\nMAE: %f\n" % (model_name, log_mean_absolute_error(np.log(valid_salaries), model_predictions))
    all_model_predictions.append(model_predictions)
predictions = np.vstack(all_model_predictions).T
predictions = np.exp(predictions)
#predictions = np.random.randint(0,5, size=(10,3))
print predictions.shape
print predictions[1:10, :]


#classifier = LinearRegression()
#classifier.fit(predictions, salaries)
#result = classifier.predict(validation_features)
result = predictions.mean(axis=1)
model_name = "-".join(model_names)
model_name = "vowpal_loc5-extra30_40log-extra40-extra40tfidf2log-mean"
joblib.dump(result, path_join(prediction_dir, model_name + "_prediction"))

write_submission(model_name + ".csv", path_join(prediction_dir, model_name + "_prediction"), unlog=False)
