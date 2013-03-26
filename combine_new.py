from data_io import DataIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error
from itertools import combinations
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge

dio = DataIO("Settings.json")
submission = False
n_trees = 10
min_samples_split = 2
param = """Normal count vector with max 200. New submission which is repeatable.
 and nicer

 """

if submission:
    type_n = "train_full"
    type_v = "valid_full"
else:
    type_n = "train"
    type_v = "valid"

#columns = ["Category", "ContractTime", "ContractType"]
#le_features = dio.get_le_features(columns, "train_full")
#extra_features = dio.get_features(columns, type_n, le_features)
#extra_valid_features = dio.get_features(columns, type_v, le_features)
#features = dio.join_features("%s_" + type_n + "_tfidf_matrix_max_f_200",
                             #["Title", "FullDescription", "LocationRaw"],
                             #extra_features)
#validation_features = dio.join_features("%s_" + type_v + "_tfidf_matrix_max_f_200",
                                        #["Title", "FullDescription", "LocationRaw"],
                                        #extra_valid_features)
#print "features", features.shape
#print "valid features", validation_features.shape

#salaries = dio.get_salaries(type_n, log=True)
if not submission:
    valid_salaries = dio.get_salaries(type_v, log=False)
print valid_salaries.shape

model_names = []
for n_trees in [20, 30, 40]:
    name = "ExtraTree_min_sample2_%dtrees_200f_noNorm_categoryTimeType_new_log" % (n_trees)
    model_names.append(name)
    name = "ExtraTree_min_sample2_%dtrees_200f_noNorm_categoryTimeType_new" % (n_trees)
    model_names.append(name)
    name = "ExtraTree_min_sample2_%dtrees_200f_noNorm_categoryTimeType_tfidfl2_new_log" % (n_trees)
    model_names.append(name)
model_names.append("vowpall")
model_names.append("vowpall_loc5")


#fit_predict(model2)
#fit_predict(model1)
#fit_predict(model3)
#fit_predict(model6, "", "", "")

all_model_predictions = []
for model_name in model_names:
    model_predictions = dio.get_prediction(model_name=model_name, type_n=type_v)
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
best_average = (10000, (0))
best_classifier = best_average
for num in range(2, len(model_names) + 1):
    for average_index in combinations(indexes, num):
        #print average_index
        #print print_index(average_index)
        my_prediction = predictions[:, average_index]
        #print my_prediction[1:10,:]
        mean_pred = my_prediction.mean(axis=1)
        mae = mean_absolute_error(mean_pred, valid_salaries)
        #print "MAE:", mae
        if best_average[0] > mae:
            best_average = (mae, average_index)
        classifier = LinearRegression()
        #classifier = RidgeCV(loss_func=mean_absolute_error)
        #classifier.fit(my_prediction, valid_salaries)
        #alpha = classifier.alpha_
        #classifier = Ridge(alpha=alpha)
        scores = cross_val_score(classifier, my_prediction, valid_salaries, cv=5, score_func=mean_absolute_error, verbose=0, n_jobs=-1)
        #print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
        if best_classifier[0] > scores.mean():
            best_classifier = (scores.mean(), average_index)
print "best average:", best_average[0], "\n",  print_index(best_average[1])
print "best classifier:", best_classifier[0], "\n",  print_index(best_classifier[1])
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

 #best average: 5713.96940131
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new
 #vowpall
 #vowpall_loc5
#best classifier: 5715.27766624
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_new
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new
 #vowpall
 #vowpall_loc5

#best average: 5699.8652553
#ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_tfidfl2_new_log
 #vowpall
 #vowpall_loc5
#best classifier: 5700.57319137
#ExtraTree_min_sample2_20trees_200f_noNorm_categoryTimeType_tfidfl2_new_log
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_new
 #ExtraTree_min_sample2_30trees_200f_noNorm_categoryTimeType_tfidfl2_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new_log
 #ExtraTree_min_sample2_40trees_200f_noNorm_categoryTimeType_new
 #vowpall
 #vowpall_loc5
