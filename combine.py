from data_io import (
    get_paths,
    read_column,
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
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
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
model4 = "ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log"
model_names = [model1, model2, model3, model4]


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)

all_model_predictions = []
for model_name in model_names:
    model_predictions = load_predictions(model_name)
    #print model_predictions[0]
    if not model_name.endswith("log") and not model_name.startswith("vowpall"):
        model_predictions = np.log(model_predictions)
    #if model_name.startswith("vowpall"):
        #model_predictions = np.log(model_predictions)
    #print model_predictions[0]
    print "%s\nMAE: %f\n" % (model_name, mean_absolute_error(valid_salaries, np.exp(model_predictions)))
    all_model_predictions.append(model_predictions)
predictions = np.vstack(all_model_predictions).T
predictions = np.exp(predictions)
#predictions = np.random.randint(0,5, size=(10,3))
print predictions.shape
print predictions[1:10, :]
indexes = range(0, len(model_names))


def print_index(index):
    names = map(lambda x: model_names[x], index)
    return "\n ".join(names)
best_average = (10000,(0))
best_classifier = best_average
for num in range(2, len(model_names) + 1):
    for average_index in combinations(indexes, num):
        print average_index
        print print_index(average_index)
        my_prediction = predictions[:, average_index]
        #print my_prediction[1:10,:]
        mean_pred = my_prediction.mean(axis=1)
        mae = mean_absolute_error(mean_pred, valid_salaries)
        print "MAE:", mae
        if best_average[0] > mae:
            best_average = (mae, average_index)
        classifier = LinearRegression()
        #classifier = RidgeCV(loss_func=mean_absolute_error)
        #classifier.fit(my_prediction, valid_salaries)
        #alpha = classifier.alpha_
        #classifier = Ridge(alpha=alpha)
        scores = cross_val_score(classifier, my_prediction, valid_salaries, cv=5, score_func=mean_absolute_error, verbose=0, n_jobs=-1)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        if best_classifier[0] > scores.mean():
            best_classifier = (scores.mean(), average_index)
print "best average:", best_average[0], print_index(best_average[1])
print "best classifier:", best_classifier[0], print_index(best_classifier[1])
#
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

#Linear regression (0,1,2):
#Accuracy: 5834.14 (+/- 29.86)

#Ridge Linear je isto
#best average: 5766.06198285 ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log, vowpall, ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log
#best classifier: 5778.35931012 vowpall, ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log

#best average: 5766.06198285 ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log, vowpall, ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log
#best classifier: 5778.35931012 vowpall, ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_log

#best average: 5694.24220595 ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log
 #vowpall
 #vowpall_loc5
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidf_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidf1_log
#best classifier: 5691.47252485 ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_log
 #vowpall
 #Random_forest_min_sample2_20trees_200f_noNorm_categoryTimeType_log
 #Random_forest_min_sample2_40trees_200f_noNorm_categoryTimeType_log
 #vowpall_loc5
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidf_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidf1_log
