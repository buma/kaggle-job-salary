from data_io import (
    get_paths,
    read_column,
    save_model,
    load_model,
    join_features,
    label_encode_column_fit,
    label_encode_column_transform,
    load_predictions,
    fit_predict,
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
from itertools import combinations, chain
import joblib


def log_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


paths = get_paths("Settings.json")
data_dir = paths["data_path"]
cache_dir = path_join(data_dir, "tmp")
prediction_dir = path_join(data_dir, "predictions")

#le_category, category_train = label_encode_column_fit("Category")
#category_valid = label_encode_column_transform(le_category, "Category")

#le_contractTime, contractTime_train = label_encode_column_fit("ContractTime")
#contractTime_valid = label_encode_column_transform(le_contractTime, "ContractTime")

#le_contractType, contractType_train = label_encode_column_fit("ContractType")
#contractType_valid = label_encode_column_transform(le_contractType, "ContractType")
#features = join_features("%s_train_count_vector_matrix_max_f_200", #train_tfidf_matrix_max_f_200
##features = join_features("%s_train_tfidf_matrix_max_f_200",
                         #["Title", "FullDescription", "LocationRaw"],
                         #data_dir,
                         #[contractTime_train, contractType_train, category_train])
#validation_features = join_features("%s_valid_count_vector_matrix_max_f_200",#valid_tfidf_matrix_max_f_200
##validation_features = join_features("%s_valid_tfidf_matrix_max_f_200",
                                    #["Title", "FullDescription", "LocationRaw"],
                                    #data_dir,
                                    #[contractTime_valid, contractType_valid, category_valid])
#print "features", features.shape
#print "valid features", validation_features.shape


#salaries = np.array(list(read_column(paths["train_data_path"], "SalaryNormalized"))).astype(np.float64)
valid_salaries = np.array(list(read_column(paths["valid_data_path"], "SalaryNormalized"))).astype(np.float64)
#salaries = np.log(salaries)
#print salaries.shape
#valid_salaries = np.log(valid_salaries)
print valid_salaries.shape

model1 = "ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log"
model2 = "vowpall"
model3 = "Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log"
model_names = [model1, model2, model3]


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)



model1_predictions = load_predictions(model1)
print mean_absolute_error(valid_salaries, model1_predictions)
model2_predictions = load_predictions(model2)
print mean_absolute_error(valid_salaries, model2_predictions)
model3_predictions = load_predictions(model3)
print mean_absolute_error(valid_salaries, model3_predictions)
predictions = np.vstack([model1_predictions, model2_predictions, model3_predictions, np.log(valid_salaries)]).T
predictions = np.exp(predictions)
#predictions = np.random.randint(0,5, size=(10,3))
print predictions.shape
print predictions[1:10,:]
for average_index in chain(combinations([0,1,2],2), combinations([0,1,2],3)):
    print average_index
    print model_names[average_index[0]], model_names[average_index[1]]
    my_prediction = predictions[:,average_index]
    #print my_prediction[1:10,:]
    mean_pred = my_prediction.mean(axis=1)
    mae = mean_absolute_error(mean_pred, valid_salaries)
    print "MAE:", mae
#(0, 1)
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log vowpall
#MAE: 5925.75752661
#(0, 2)
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log
#MAE: 6373.82572206
#(1, 2)
#vowpall Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log
#MAE: 6157.26021497
#(0, 1, 2)
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log vowpall
#MAE: 5889.15272898

