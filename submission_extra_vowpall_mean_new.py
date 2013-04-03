from data_io import DataIO
#from os.path import join as path_join
#import joblib
import numpy as np
dio = DataIO("Settings_submission.json")
submission = True
if submission:
    type_n = "train_full"
    type_v = "test_full"
else:
    type_n = "train"
    type_v = "valid"


model_names = [
    "ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log",
    "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log",
    "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new",
    "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidfl2_new_log",
    "vowpall_submission",
    "vowpall_loc5"
]
#model_names = [model2, model4]
#model_names = [model1, model6, model4]


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)
#fit_predict(model5)

#fit_predict(model4, features, salaries, validation_features, type_n="test_subm")

model_name = "predictions_submit_test.txt"
predictions = np.loadtxt(path_join(dio.data_dir, "code", "from_fastml", "optional", model_name))
dio.save_prediction("vowpall_submission", predictions, type_v)

model_name = "predictions_submit_test_loc5.txt"
predictions = np.loadtxt(path_join(dio.data_dir, "code", "from_fastml", "optional", model_name))
dio.save_prediction("vowpall_loc5", predictions, type_v)

all_model_predictions = []
for model_name in model_names:
    #fit_predict(model_name, features, salaries, validation_features, type_n="test_subm")
    model_predictions = dio.get_prediction(model_name=model_name, type_n="test_full")
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


result = predictions.mean(axis=1)
model_name = "-".join(model_names)
model_name = "vowpal_loc5-extra30_40log-extra40-extra40tfidf2log-mean-test"
dio.save_prediction(model_name, result, type_v)

dio.write_submission(model_name + ".csv", result)

