from data_io import (
    get_paths,
    read_column,
    load_model,
    join_features,
    label_encode_column_fit,
    label_encode_column_transform,
    load_predictions,
    fit_predict,
    write_submission
)
from os.path import join as path_join
#import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


paths = get_paths("Settings_submission.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

le_category, category_train = label_encode_column_fit("Category")
category_valid = label_encode_column_transform(le_category, "Category")

le_contractTime, contractTime_train = label_encode_column_fit("ContractTime")
contractTime_valid = label_encode_column_transform(le_contractTime, "ContractTime")

le_contractType, contractType_train = label_encode_column_fit("ContractType")
contractType_valid = label_encode_column_transform(le_contractType, "ContractType")
features = join_features("%s_train_full_count_vector_matrix_max_f_200", #train_tfidf_matrix_max_f_200
#features = join_features("%s_train_tfidf_matrix_max_f_200",
                         ["Title", "FullDescription", "LocationRaw"],
                         data_dir,
                         [contractTime_train, contractType_train, category_train])
#for column_name in ["Title", "FullDescription", "LocationRaw"]:
    #vocabulary = joblib.load(path_join(cache_dir, column_name + "count_vectorizer_vocabulary"))
    #stop_words = joblib.load(path_join(cache_dir, column_name + "count_vectorizer_stop_words"))

    #count_vector_titles = CountVectorizer(max_features=200, vocabulary=vocabulary, stop_words=stop_words)
    #titles_valid = count_vector_titles.transform(
        #read_column(paths["test_data_path"], column_name))
    #print joblib.dump(titles_valid, path_join(cache_dir, column_name + "_test_count_vector_matrix_max_f_200"))
validation_features = join_features("%s_valid_full_count_vector_matrix_max_f_200",#valid_tfidf_matrix_max_f_200
#validation_features = join_features("%s_valid_tfidf_matrix_max_f_200",
                                    ["Title", "FullDescription", "LocationRaw"],
                                    data_dir,
                                    [contractTime_valid, contractType_valid, category_valid])
print "features", features.shape
print "valid features", validation_features.shape


salaries = np.array(list(read_column(paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
#valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
salaries = np.log(salaries)
print salaries.shape
#valid_salaries = np.log(valid_salaries)
#print valid_salaries.shape

model1 = "ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log"
model2 = "vowpall_submission"
model3 = "Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log"
model4 = "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
model5 = "Random_forest_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
#model_names = [model2, model4]
model_names = [model1, model2, model4, model5]


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)
#fit_predict(model5)

#fit_predict(model4, features, salaries, validation_features, type_n="test_subm")


all_model_predictions = []
for model_name in model_names:
    fit_predict(model_name, features, salaries, validation_features, type_n="test_subm")
    model_predictions = load_predictions(model_name, type_n="test_subm")
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
model_name = "vowpal-extra20_40-random40-mean"
joblib.dump(result, path_join(prediction_dir, model_name + "_prediction"))

write_submission("vowpal-extra20_40-random40-mean.csv", path_join(prediction_dir, model_name + "_prediction"), unlog=False)
